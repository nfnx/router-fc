"""Offline, reproducible retrieval evaluation on cleaned SKILLRET research tests.

Weights stay local. No task execution, generation, training or test-label tuning.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/skillret-research-clean-v1.1'
PROMPTS = {
    'qwen': 'Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery: ',
    'skillret': 'Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery: ',
    'skillrouter': 'Instruct: Given a coding task description, retrieve the most relevant skill document that would help an agent complete the task\nQuery:',
    'r3': 'Instruct: Given a user request, retrieve the agent skill that solves it.\nQuery: ',
}
INSTRUCTIONS = {
    'qwen': 'Given a skill search query, judge whether the skill document is relevant and useful for the query',
    'skillret': 'Given a skill search query, judge whether the skill document is relevant and useful for the query',
    'skillrouter': 'Given a coding task description, judge whether the skill document is relevant and useful for completing the task',
    'r3': 'Given a user request, retrieve the agent skill that solves it.',
}

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_views():
    views = {}
    for name, directory in [('extended', DATA / 'data'), ('research_only', DATA / 'research_only/data')]:
        skills = read_jsonl(directory / 'skills/test.jsonl')
        queries = read_jsonl(directory / 'queries/test.jsonl')
        ids = {s['id'] for s in skills}
        qrels = read_jsonl(directory / 'qrels/test.jsonl')
        gold = {q['id']: {r['skill_id'] for r in qrels if r['query_id'] == q['id'] and r['relevance'] > 0} for q in queries}
        assert len(ids) == len(skills)
        for q in queries:
            assert gold[q['id']] == set(q['skill_ids']) and gold[q['id']] <= ids
        views[name] = {'skills': skills, 'queries': queries, 'gold': gold, 'hashes': {f: digest(directory / f / 'test.jsonl') for f in ['skills', 'queries', 'qrels']}}
    return views

def doc_text(skill):
    return ' | '.join((skill.get('name', '').strip(), (skill.get('description') or '').strip(), (skill.get('skill_md') or skill.get('body') or '').strip()))

def metrics(ranking, gold):
    assert len(ranking) == len(set(ranking))
    out = {'hit@1': float(ranking[0] in gold)}
    for k in [1, 5, 10, 20]:
        found = len(set(ranking[:k]) & gold)
        out[f'recall@{k}'] = found / len(gold)
        out[f'completeness@{k}'] = float(found == len(gold))
    out['ndcg@10'] = sum(1 / math.log2(i + 2) for i, sid in enumerate(ranking[:10]) if sid in gold) / sum(1 / math.log2(i + 2) for i in range(min(10, len(gold))))
    out['mrr@10'] = next((1 / (i + 1) for i, sid in enumerate(ranking[:10]) if sid in gold), 0.0)
    return out

def aggregate(rows):
    keys = rows[0]['metrics']
    result = {key: float(np.mean([r['metrics'][key] for r in rows])) for key in keys}
    a = np.array([r['metrics']['ndcg@10'] for r in rows])
    rng = np.random.default_rng(42)
    means = a[rng.integers(0, len(a), (2000, len(a)))].mean(1)
    result['ndcg@10_bootstrap95'] = np.quantile(means, [.025, .975]).tolist()
    for key in ['query_encode_s', 'search_s', 'rerank_s']:
        vals = np.array([r.get(key, 0.0) for r in rows])
        result[key + '_mean'] = float(vals.mean())
    total = np.array([sum(r.get(k, 0.0) for k in ['query_encode_s', 'search_s', 'rerank_s']) for r in rows])
    result['online_ms_median'] = float(np.median(total) * 1000)
    result['online_ms_p95'] = float(np.quantile(total, .95) * 1000)
    result['queries'] = len(rows)
    return result

def sync():
    torch.cuda.synchronize()

def configure_attention():
    # PyTorch 2.7 Windows has efficient SDPA but no Flash Attention. Its native
    # GQA path falls back to the slow math/cuDNN kernels. Reuse Transformers'
    # existing explicit KV-repeat branch, then select efficient SDPA only.
    import transformers.integrations.sdpa_attention as sdpa
    sdpa.use_gqa_in_sdpa = lambda attention_mask, key: False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

def unload(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()

def load_model(repo, causal=False, quantized=False):
    path = ROOT / 'models' / repo.replace('/', '--')
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side='left')
    cls = AutoModelForCausalLM if causal else AutoModel
    kwargs = dict(local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 'cuda:0'})
    if quantized:
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = cls.from_pretrained(path, **kwargs).eval()
    return model, tokenizer

@torch.inference_mode()
def encode(model, tokenizer, texts, max_length, batch_size=2):
    vectors = []
    truncated = 0
    token_count = 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        raw = tokenizer(batch, add_special_tokens=True, return_length=True)
        truncated += sum(n > max_length for n in raw['length'])
        token_count += sum(raw['length'])
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to('cuda')
        hidden = model(**inputs, use_cache=False).last_hidden_state[:, -1, :]
        vectors.append(torch.nn.functional.normalize(hidden.float(), p=2, dim=1).cpu().numpy())
    vectors = np.concatenate(vectors)
    assert np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
    return vectors, {'truncated': truncated, 'texts': len(texts), 'original_tokens': token_count}

class Reranker:
    def __init__(self, repo, family, max_length):
        self.model, self.tokenizer = load_model(repo, causal=True)
        self.family, self.max_length = family, max_length
        self.yes = self.tokenizer.convert_tokens_to_ids('yes')
        self.no = self.tokenizer.convert_tokens_to_ids('no')
        self.truncated, self.pairs = 0, 0
        self.prefix = self.tokenizer.encode('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n', add_special_tokens=False)
        self.suffix = self.tokenizer.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', add_special_tokens=False)
        if family == 'r3':
            marker = '__R3_DOCUMENT_BOUNDARY__'
            rendered = self.tokenizer.apply_chat_template([{'role': 'system', 'content': INSTRUCTIONS['r3']}, {'role': 'query', 'content': 'warmup'}, {'role': 'document', 'content': marker}], tokenize=False)
            self.r3_suffix = self.tokenizer.encode(rendered.split(marker)[1], add_special_tokens=False)

    @torch.inference_mode()
    def score(self, query, docs):
        tokenized = []
        for doc in docs:
            if self.family == 'r3':
                # Preserve the released R3 tokenizer's own chat template, including final whitespace.
                ids = self.tokenizer.apply_chat_template([{'role': 'system', 'content': INSTRUCTIONS['r3']}, {'role': 'query', 'content': query}, {'role': 'document', 'content': doc}], tokenize=True, return_dict=False)
                # Preserve the assistant suffix if a long document is truncated.
                tail = self.r3_suffix
                assert ids[-len(tail):] == tail, 'R3 chat template suffix changed'
                if len(ids) > self.max_length:
                    self.truncated += 1
                    ids = ids[:self.max_length - len(tail)] + tail
            else:
                sep = '\n\n' if self.family == 'skillrouter' else '\n'
                text = f'<Instruct>: {INSTRUCTIONS[self.family]}{sep}<Query>: {query}{sep}<Document>: {doc}'
                middle = self.tokenizer.encode(text, add_special_tokens=False)
                limit = self.max_length - len(self.prefix) - len(self.suffix)
                self.truncated += int(len(middle) > limit)
                ids = self.prefix + middle[:limit] + self.suffix
            tokenized.append(ids)
            self.pairs += 1
        scores = np.empty(len(tokenized), dtype=np.float32)
        order = sorted(range(len(tokenized)), key=lambda i: len(tokenized[i]))
        for start in range(0, len(tokenized), 8):
            indices = order[start:start + 8]
            inputs = self.tokenizer.pad({'input_ids': [tokenized[i] for i in indices]}, padding=True, return_tensors='pt').to('cuda')
            # Compute only the final token's vocabulary logits, as in SKILLRET's upstream scorer.
            hidden = self.model.model(**inputs, use_cache=False).last_hidden_state[:, -1:, :]
            logits = self.model.lm_head(hidden)[:, 0, :].float()
            scores[indices] = (logits[:, self.yes] - logits[:, self.no]).cpu().numpy()
        assert np.isfinite(scores).all(), 'Non-finite reranker logits'
        return np.array(scores)

def run(args):
    assert torch.cuda.is_available(), 'CUDA required for this configuration'
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    configure_attention()
    outdir = ROOT / args.output
    outdir.mkdir(parents=True, exist_ok=True)
    views = load_views()
    previous_protocol = outdir / 'protocol.json'
    previous = {}
    if previous_protocol.exists():
        previous = json.loads(previous_protocol.read_text(encoding='utf-8'))
        assert previous['max_length'] == args.max_length, 'Use a new output directory when changing context length'
        assert previous['data_hashes'] == {k: v['hashes'] for k, v in views.items()}, 'Dataset changed during a resumed run'
        assert previous['script_sha256'] == digest(Path(__file__)), 'Evaluator changed; use a new output directory'
    config = json.loads((ROOT / 'config/local_models.json').read_text())
    manifest = json.loads((ROOT / 'models/MANIFEST.json').read_text())
    if previous_protocol.exists():
        for repo, item in previous['models'].items():
            assert manifest[repo]['revision'] == item['revision'], 'A cached model revision changed'
    dump(outdir / 'protocol.json', {**previous, 'attention_backend': 'efficient SDPA only, explicit KV repeat for GQA', 'max_length': args.max_length, 'batch_size': 2, 'rerank_batch_size': 8, 'rerank_batching': 'sort pairs by token length within query, then restore original pair order', 'rerank_top_n': 20, 'dtype': 'bfloat16', '8b_dtype': 'NF4 double quantization with BF16 compute', 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(), 'seed': 42, 'query_prompts': PROMPTS, 'rerank_instructions': INSTRUCTIONS, 'data_hashes': {k: v['hashes'] for k, v in views.items()}, 'models': {k: {'revision': v['revision'], 'license': v['license']} for k, v in manifest.items()}, 'script_sha256': digest(Path(__file__))})
    for family in args.methods.split(','):
        print(f'BEGIN {family}', flush=True)
        fam = 'skillret' if family == 'skillret8b' else family
        emb_path = outdir / f'{family}_embedding.json'
        if family == 'bm25' and emb_path.exists():
            print('CACHED bm25', flush=True)
            continue
        if not emb_path.exists():
            if family == 'bm25':
                from rank_bm25 import BM25Okapi
                embedding_results = {}
                for viewname, view in views.items():
                    t = time.perf_counter()
                    index = BM25Okapi([re.findall(r'\w+', doc_text(s).lower()) for s in view['skills']])
                    index_s = time.perf_counter() - t
                    rows = []
                    for q in view['queries']:
                        t = time.perf_counter()
                        scores = index.get_scores(re.findall(r'\w+', q['query'].lower()))
                        order = np.argsort(-scores, kind='stable')
                        ranking = [view['skills'][i]['id'] for i in order]
                        elapsed = time.perf_counter() - t
                        rows.append({'query_id': q['id'], 'ranking': ranking, 'metrics': metrics(ranking, view['gold'][q['id']]), 'search_s': elapsed})
                    embedding_results[viewname] = {'rows': rows, 'metrics': aggregate(rows), 'index_s': index_s}
                dump(emb_path, embedding_results)
                continue
            model, tokenizer = load_model(config[family]['embedding'], quantized=family == 'skillret8b')
            encode(model, tokenizer, ['warmup'], args.max_length, 1)
            embedding_results = {}
            for viewname, view in views.items():
                docs = [doc_text(s) for s in view['skills']]
                sync(); t = time.perf_counter()
                dvec, doc_stats = encode(model, tokenizer, docs, args.max_length, 1 if family == 'skillret8b' else 2)
                sync(); index_s = time.perf_counter() - t
                rows, query_vectors, query_stats = [], [], []
                for q in view['queries']:
                    sync(); t = time.perf_counter()
                    qvec, stats = encode(model, tokenizer, [PROMPTS[fam] + q['query']], args.max_length, 1)
                    sync(); encode_s = time.perf_counter() - t
                    t = time.perf_counter()
                    scores = qvec[0] @ dvec.T
                    order = np.argsort(-scores, kind='stable')
                    ranking = [view['skills'][i]['id'] for i in order]
                    search_s = time.perf_counter() - t
                    rows.append({'query_id': q['id'], 'ranking': ranking, 'metrics': metrics(ranking, view['gold'][q['id']]), 'query_encode_s': encode_s, 'search_s': search_s})
                    query_vectors.append(qvec[0]); query_stats.append(stats)
                embedding_results[viewname] = {'rows': rows, 'metrics': aggregate(rows), 'index_s': index_s, 'document_tokens': doc_stats, 'query_truncated': sum(s['truncated'] for s in query_stats)}
                np.savez(outdir / f'{family}_{viewname}_vectors.npz', documents=dvec, queries=np.array(query_vectors))
                print(f'{family}/{viewname} {embedding_results[viewname]["metrics"]}', flush=True)
            del model; gc.collect(); torch.cuda.empty_cache()
            dump(emb_path, embedding_results)
        else:
            embedding_results = json.loads(emb_path.read_text(encoding='utf-8'))
        if family == 'qwen':
            run_skillsight(views, embedding_results, outdir)
        rerank_path = outdir / f'{family}_reranked.json'
        if not rerank_path.exists():
            rr = Reranker(config[family]['reranker'], fam, args.max_length)
            rr.score('warmup', ['warmup']); sync()
            results = {}
            for viewname, view in views.items():
                skills = {s['id']: s for s in view['skills']}
                rows = []
                for i, (q, base) in enumerate(zip(view['queries'], embedding_results[viewname]['rows'])):
                    sync(); t = time.perf_counter()
                    scores = rr.score(q['query'], [doc_text(skills[sid]) for sid in base['ranking'][:20]])
                    sync(); elapsed = time.perf_counter() - t
                    indices = np.argsort(-scores, kind='stable')
                    ranking = [base['ranking'][j] for j in indices] + base['ranking'][20:]
                    rows.append({**base, 'ranking': ranking, 'metrics': metrics(ranking, view['gold'][q['id']]), 'rerank_s': elapsed, 'top20_scores_before_sort': scores.tolist()})
                    if i % 10 == 0:
                        print(f'{family}/{viewname} rerank {i+1}/{len(view["queries"])} ({elapsed:.2f}s/query)', flush=True)
                        dump(outdir / f'{family}_{viewname}_partial.json', rows)
                results[viewname] = {'rows': rows, 'metrics': aggregate(rows), 'index_s': embedding_results[viewname]['index_s']}
            results['_reranker_token_stats'] = {'pairs_including_warmup': rr.pairs, 'truncated': rr.truncated}
            del rr; gc.collect(); torch.cuda.empty_cache()
            dump(rerank_path, results)
        print(f'COMPLETE {family}', flush=True)

def run_skillsight(views, embedding_results, outdir):
    path = outdir / 'skillsight_embedding.json'
    if path.exists():
        return
    sys.path.insert(0, str(ROOT / 'external'))
    from SkillSight.retriever import SkillSightRetriever
    from SkillSight.constants import VARIANT_SKILLSIGHT
    results = {}
    for name, view in views.items():
        vectors = np.load(outdir / f'qwen_{name}_vectors.npz')
        t = time.perf_counter()
        retriever = SkillSightRetriever([doc_text(s) for s in view['skills']], vectors['documents'], [s['id'] for s in view['skills']], candidate_k=300, lexical_beta=1.0)
        index_s = time.perf_counter() - t
        rows = []
        for i, (q, base) in enumerate(zip(view['queries'], embedding_results[name]['rows'])):
            t = time.perf_counter()
            result = retriever.score([q['query']], vectors['queries'][i:i+1], final_k=len(view['skills']))[VARIANT_SKILLSIGHT]
            ranking = [view['skills'][j]['id'] for j in result.top_indices[0]]
            elapsed = time.perf_counter() - t
            rows.append({**base, 'ranking': ranking, 'metrics': metrics(ranking, view['gold'][q['id']]), 'search_s': elapsed})
        results[name] = {'rows': rows, 'metrics': aggregate(rows), 'index_s': index_s + embedding_results[name]['index_s'], 'background_rank': retriever.stats.background_rank}
    dump(path, results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', default='bm25,qwen,skillret,skillrouter,r3,skillret8b')
    parser.add_argument('--max-length', type=int, default=4096)
    parser.add_argument('--output', default='runs/research-test-20261029')
    run(parser.parse_args())
