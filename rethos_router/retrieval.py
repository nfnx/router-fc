"""Local SkillRet embeddings and a content-addressed cosine index."""
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
from importlib.metadata import version

from .common import ConfigurationError, RouterError, integer, read_json, resolve_path, strict_json

LOG = logging.getLogger(__name__)
QUERY_PREFIX = 'Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery: '
REPRESENTATION = 'skillret-name-description-full-skillmd-last-token-l2-v1'


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    description: str
    body: str

    def document(self):
        return ' | '.join((self.name.strip(), self.description.strip(), self.body.strip()))


@dataclass(frozen=True)
class Candidate:
    skill: Skill
    score: float


def load_pool(registry_path, root, scope='all'):
    registry = read_json(registry_path)
    if not isinstance(registry, dict) or not isinstance(registry.get('skills_path'), str):
        raise ConfigurationError('Skill registry must specify skills_path')
    pool_path = resolve_path(registry['skills_path'], root)
    try:
        raw = pool_path.read_bytes()
        records = [strict_json(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f'Cannot read skill pool: {pool_path}') from exc
    ids = set()
    skills = []
    for record in records:
        if not isinstance(record, dict):
            raise ConfigurationError('Skill pool records must be JSON objects')
        sid, name = record.get('id'), record.get('name')
        if not isinstance(sid, str) or not sid.strip() or sid in ids or not isinstance(name, str) or not name.strip():
            raise ConfigurationError('Skill IDs must be nonempty and unique; names must be nonempty')
        ids.add(sid)
        if scope == 'research_core' and record.get('research_role') != scope:
            continue
        description = record.get('description') or ''
        body = record.get('skill_md') or record.get('body') or ''
        if not isinstance(description, str) or not isinstance(body, str):
            raise ConfigurationError('Skill text fields must be strings')
        skills.append(Skill(sid, name, description, body))
    if not skills:
        raise ConfigurationError('The selected skill pool is empty')
    return skills, hashlib.sha256(raw).hexdigest()


class SkillRetEncoder:
    """Lazy local-only model; query and document representations stay identical."""

    def __init__(self, settings, root):
        self.path = resolve_path(settings['model_path'], root)
        self.max_length = integer(settings['max_length'], 'embedding.max_length', minimum=32, maximum=32_768)
        self.batch_size = integer(settings['batch_size'], 'embedding.batch_size', maximum=64)
        self.device_setting = settings.get('device', 'auto')
        if self.device_setting not in ('auto', 'cpu', 'cuda'):
            raise ConfigurationError('Embedding device must be auto, cpu or cuda')
        self.expected_weight = settings.get('weight_sha256')
        self._fingerprint = None
        self._model = self._tokenizer = None
        self._lock = threading.Lock()

    @property
    def fingerprint(self):
        if self._fingerprint is None:
            import torch
            self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if self.device_setting == 'auto' else self.device_setting
            self.dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
            files = sorted(p for p in self.path.iterdir() if p.is_file() and
                           (p.suffix in ('.json', '.safetensors', '.model') or p.name in ('merges.txt', 'vocab.txt')))
            if not files or not (self.path / 'model.safetensors').is_file():
                raise ConfigurationError(f'Local SkillRet weights not found: {self.path}')
            hashes = {}
            for file in files:
                h = hashlib.sha256()
                with file.open('rb') as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        h.update(block)
                hashes[file.name] = h.hexdigest()
            if self.expected_weight and hashes['model.safetensors'] != self.expected_weight:
                raise ConfigurationError('SkillRet weight hash differs from the configured snapshot')
            self._fingerprint = {
                'files': hashes, 'max_length': self.max_length, 'dtype': str(self.dtype),
                'representation': REPRESENTATION, 'query_prefix': QUERY_PREFIX,
                'torch': torch.__version__, 'transformers': version('transformers'),
                'attention': 'windows-explicit-kv-sdpa' if os.name == 'nt' and self.device == 'cuda' else 'sdpa',
            }
        return self._fingerprint

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        _ = self.fingerprint
        if os.name == 'nt' and self.device == 'cuda':
            # Same adaptation as scripts/evaluate_local_routers.py: native GQA can
            # fall back to huge math-attention allocations on Windows/PyTorch 2.7.
            # Transformers' explicit KV-repeat path enables memory-efficient SDPA.
            import transformers.integrations.sdpa_attention as sdpa
            sdpa.use_gqa_in_sdpa = lambda attention_mask, key: False
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        self._tokenizer = AutoTokenizer.from_pretrained(self.path, padding_side='left', local_files_only=True, trust_remote_code=False)
        self._tokenizer.padding_side = 'left'
        self._tokenizer.truncation_side = 'right'
        self._model = AutoModel.from_pretrained(
            self.path, dtype=self.dtype, attn_implementation='sdpa',
            local_files_only=True, trust_remote_code=False,
        ).to(self.device).eval()

    def encode(self, texts, *, query=False):
        import numpy as np
        import torch
        with self._lock:
            self._load()
            values = [QUERY_PREFIX + text if query else text for text in texts]
            result = []
            for start in range(0, len(values), self.batch_size):
                inputs = self._tokenizer(values[start:start + self.batch_size], padding=True, truncation=True,
                                         max_length=self.max_length, return_tensors='pt').to(self.device)
                with torch.inference_mode():
                    hidden = self._model(**inputs, use_cache=False).last_hidden_state[:, -1, :]
                    vectors = torch.nn.functional.normalize(hidden.float(), dim=-1).cpu().numpy()
                if not np.isfinite(vectors).all():
                    raise RouterError('SkillRet returned non-finite embeddings')
                result.append(vectors)
                if not query:
                    LOG.info('SkillRet index: %s/%s', min(start + self.batch_size, len(values)), len(values))
            return np.concatenate(result)


class SkillIndex:
    def __init__(self, skills, vectors, encoder):
        import numpy as np
        self.skills, self.encoder = tuple(skills), encoder
        self.vectors = np.asarray(vectors, dtype=np.float32)
        if (self.vectors.ndim != 2 or len(self.vectors) != len(skills) or self.vectors.shape[1] == 0 or
                not np.isfinite(self.vectors).all()):
            raise ConfigurationError('Invalid skill index dimensions or values')
        norms = np.linalg.norm(self.vectors, axis=1)
        if not np.isfinite(norms).all() or (norms < 1e-8).any():
            raise ConfigurationError('Skill index contains invalid vector norms')
        self.vectors = self.vectors / norms[:, None]

    @classmethod
    def load_or_build(cls, skills, pool_sha256, encoder, cache_path, *, rebuild=False):
        import numpy as np
        cache_path = Path(cache_path)
        identity = {'schema_version': 1, 'pool_sha256': pool_sha256,
                    'skill_ids': [s.id for s in skills], 'encoder': encoder.fingerprint}
        if not rebuild and cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as cache:
                    if strict_json(str(cache['metadata'].item())) == identity:
                        return cls(skills, cache['vectors'].copy(), encoder)
                LOG.info('Skill index snapshot changed; rebuilding')
            except (ValueError, OSError, KeyError, ConfigurationError):
                LOG.warning('Skill index is invalid; rebuilding')
        index = cls(skills, encoder.encode([s.document() for s in skills]), encoder)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=cache_path.name + '.', suffix='.tmp', dir=cache_path.parent)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                np.savez_compressed(stream, metadata=json.dumps(identity, sort_keys=True), vectors=index.vectors)
            os.replace(temporary, cache_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return index

    def search(self, request, top_k):
        import numpy as np
        query = np.asarray(self.encoder.encode([request], query=True), dtype=np.float32)
        if query.shape != (1, self.vectors.shape[1]) or not np.isfinite(query).all():
            raise RouterError('Invalid query embedding')
        norm = np.linalg.norm(query[0])
        if not np.isfinite(norm) or norm < 1e-8:
            raise RouterError('Query embedding has an invalid norm')
        scores = self.vectors @ (query[0] / norm)
        ranks = np.argsort(-scores, kind='stable')[:top_k]
        return [Candidate(self.skills[int(i)], float(scores[i])) for i in ranks]
