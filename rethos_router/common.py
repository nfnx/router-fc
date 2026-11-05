"""Configuration and strict JSON helpers. No skill execution or environment probing."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESSES = frozenset({'claude', 'codex', 'dsh'})
MODELS = frozenset({'deepseek', 'glm'})
TASK_TYPES = frozenset({'corpus_retrieval', 'deep_research', 'research_report', 'scientific_execution'})


class RouterError(Exception):
    """A routing request cannot be completed; do not invent a successful route."""


class ConfigurationError(RouterError):
    pass


class ReviewError(RouterError):
    pass


class RequestError(RouterError):
    pass


def strict_json(text):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('Duplicate JSON key')
            out[key] = value
        return out

    def constant(value):
        raise ValueError('Non-finite JSON value')

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def read_json(path):
    try:
        return strict_json(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f'Cannot read JSON configuration: {path}') from exc


def resolve_path(value, root=ROOT):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def integer(value, name, minimum=1, maximum=1_000_000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ConfigurationError(f'{name} must be an integer in [{minimum}, {maximum}]')
    return value


def positive_number(value, name):
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f'{name} must be positive and finite')
    return value


def load_config(path=None, root=ROOT):
    config = read_json(path or Path(root) / 'config/request_router.json')
    if not isinstance(config, dict) or config.get('schema_version') != 1:
        raise ConfigurationError('Unsupported router configuration version')
    for key in ('skill_pool_registry', 'agent_registry', 'embedding', 'retrieval', 'review', 'fallback'):
        if key not in config:
            raise ConfigurationError(f'Missing router configuration field: {key}')
    for key in ('skill_pool_registry', 'agent_registry'):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ConfigurationError(f'{key} must be a nonempty path')
    for key in ('embedding', 'retrieval', 'review', 'fallback'):
        if not isinstance(config[key], dict):
            raise ConfigurationError(f'{key} must be a JSON object')
    for key in ('model_path', 'index_path'):
        if not isinstance(config['embedding'].get(key), str) or not config['embedding'][key].strip():
            raise ConfigurationError(f'embedding.{key} must be a nonempty path')
    integer(config.get('max_request_chars'), 'max_request_chars', maximum=1_000_000)
    retrieval = config['retrieval']
    integer(retrieval.get('top_k'), 'retrieval.top_k', maximum=100)
    integer(retrieval.get('max_selected_skills'), 'retrieval.max_selected_skills', minimum=0, maximum=10)
    if retrieval['max_selected_skills'] > retrieval['top_k']:
        raise ConfigurationError('max_selected_skills cannot exceed top_k')
    integer(retrieval.get('candidate_body_chars'), 'candidate_body_chars', minimum=256, maximum=200_000)
    integer(retrieval.get('total_candidate_body_chars'), 'total_candidate_body_chars', minimum=256, maximum=500_000)
    if retrieval['total_candidate_body_chars'] < retrieval['top_k'] * 256:
        raise ConfigurationError('Total candidate body budget must allow at least 256 characters per candidate')
    if config.get('skill_scope') not in ('all', 'research_core'):
        raise ConfigurationError('skill_scope must be all or research_core')
    config['_root'] = str(Path(root).resolve())
    return config
