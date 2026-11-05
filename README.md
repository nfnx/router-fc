# RethosAI-Router-Dev
Agent router for Rethos.ai agent framework

## Text request routing

The request router implements:

`user text -> SkillRet cosine top-K -> ONE LLM intent + skill check -> historical harness/model policy -> JSON`

Successful CLI and HTTP responses contain **only** `skills`, `harness`, and
`model`. `skills` is an array of stable skill registry IDs (default zero or one),
so same-name implementations remain distinct. Harness values are `codex`,
`claude` (Claude Code), and `dsh` (DeepSeek Harness); model values are the execution
provider aliases `deepseek` and `glm`. The check LLM is configured separately.

```json
{"skills": [], "harness": "claude", "model": "deepseek"}
```

Configuration: [`config/request_router.json`](config/request_router.json).
Output schema: [`config/request_router.output.schema.json`](config/request_router.output.schema.json).
The default pool is the current **734-record** pool; set `skill_scope` to
`research_core` to restrict it to 653 records. Existing pool files, local
SkillRet weights, and the evaluation registry must be present. Large model,
data and generated index files remain outside Git.

Use the existing `.venv` with the local requirements below. For a new environment,
install a suitable PyTorch build first, then install this project editable with
`python -m pip install -e ".[embedding]"`. The optional console command
`rethos-router` is equivalent to `python -m rethos_router`. Run from the repository
checkout; configuration paths resolve relative to its root.

```powershell
# Optional prebuild: local embeddings only, no check-model API request.
.venv\Scripts\python.exe -m rethos_router build-index

# Set the key in this process; it is never written into router configuration.
$env:ROUTER_CHECK_API_KEY = '<your-key>'
.venv\Scripts\python.exe -m rethos_router route --request '调研空间转录组去卷积方法并比较证据'

# Alternatively read UTF-8 text from a file, or read text from stdin.
.venv\Scripts\python.exe -m rethos_router route --request-file request.txt

# Persistent process: reuse the embedding model and index across requests.
.venv\Scripts\python.exe -m rethos_router serve --host 127.0.0.1 --port 8787
```

HTTP request: `POST /route`, `Content-Type: application/json`, body:

```json
{"request": "对h5ad数据进行质控、聚类并生成UMAP图"}
```

Python callers can use `from rethos_router import Router`, initialize once with
`router = Router.from_config()`, then call `router.route(request_text)`.

Check API settings support DeepSeek and GLM Chat Completions JSON output.
Override defaults with `ROUTER_CHECK_PROVIDER` (`deepseek` or `glm`),
`ROUTER_CHECK_BASE_URL` (base URL **before** `/chat/completions`),
`ROUTER_CHECK_MODEL`, and `ROUTER_CHECK_API_KEY`. Without the generic key,
`DEEPSEEK_API_KEY` or `GLM_API_KEY` is used for the selected check provider.
Direct API defaults are independent of the historical execution model IDs.
The DeepSeek check disables thinking; GLM-5.3 uses low reasoning effort (that
model does not support disabling reasoning). Provider-specific reasoning
parameters are configurable in `review.providers`. GLM accounts requiring the
Coding Plan endpoint should override the base URL accordingly.
For a compatible gateway, explicitly configure its base URL, model and key.
Remote endpoints require HTTPS; localhost HTTP supports local gateways/tests.
Provider documentation: [DeepSeek JSON output](https://api-docs.deepseek.com/guides/json_mode/),
[GLM API quick start](https://docs.z.ai/guides/overview/quick-start).

Routing behavior:

- Original request text is embedded directly, without a preceding LLM rewrite.
  Documents use `name | description | SKILL.md`, left padding, final-token pooling,
  L2 normalization and cosine similarity. The default context limit is 4096 tokens.
  The old experimental `.25` threshold was tuned on manually summarized queries;
  it is **not** applied to this raw-query pipeline.
- Top-5 candidates are reviewed in one non-streaming LLM request. Candidate
  bodies are bounded head/tail excerpts with an explicit truncation marker.
  The same response identifies task type, explicit speed/provider preferences,
  and selected candidate IDs. Candidate text is treated as untrusted data.
- The program validates the review and selects a pair using
  `config/research_eval/agent_registry.v0.1.json`. The four task types are
  `corpus_retrieval`, `deep_research`, `research_report`, `scientific_execution`.
  Defaults use existing `quality_first` priority lists; an explicit speed request
  uses `latency_first`. A provider constraint is preserved. When the latency
  shortlist excludes that provider, same-task baseline common-cohort timing
  selects the fastest measured configuration within the requested provider.
- `mixed` uses its primary task if identified. `other` or an unresolved mixed
  request uses the configured default `claude + deepseek`; an explicit GLM
  constraint uses its configured GLM fallback. This is a default, not evidence
  of performance on unseen task types. A no-skill selection remains valid.
- The historical registry is provisional, with small, single-run cohorts and
  some substitute judging. Default pair selection uses no-skill baseline
  policies. Skill-specific exploratory overrides are not automatically enabled.
  No cross-benchmark aggregate or unmeasured cost score is invented.
- Index metadata binds pool bytes/ordered IDs, weight/tokenizer hashes and
  embedding settings. A stale index rebuilds automatically. This validates the
  router's own retrieval cache, not selected skills' execution environments.
- No selected skill, harness, dependency installer or runtime-readiness check
  is executed. Invalid/blank/truncated review output is an error, with no automatic
  repair or retry. Successful requests make exactly **one** check-LLM HTTP call.
  Failures produce a nonzero CLI exit with stderr diagnostics, or an HTTP error
  with a plain-text body; no successful three-field result is fabricated.
  CLI loading/progress logs go to stderr, never mixed with result JSON.

Focused checks (no external inference, no changes to existing benchmark runs):

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p test_request_router.py -v
```

## Local skill-router evaluation

The local benchmark compares BM25, Qwen3, SkillRet, SkillRouter, R3, and
SkillSight on the cleaned research subsets. Source checkouts are under
`external/`, weights under `models/`, and the isolated Python environment under
`.venv/`. These large local files are excluded from Git. Exact upstream commits
are in `config/deployment_sources.json`; model revisions and file hashes are in
`models/MANIFEST.json`.

Run from this project directory in PowerShell:

```powershell
# Validate downloaded sources and model hashes.
.venv\Scripts\python.exe scripts/verify_local_deployment.py

# Evaluate original test splits and generate the report. Completed method
# artifacts are reused only when the evaluator and data hashes match.
.\scripts\run_local_eval.ps1
```

Results: `reports/research-test-20261029.md` and
`runs/research-test-20261029/summary.json`. The run directory also retains full
rankings, per-query metrics, timing, embeddings, and the evaluation protocol.
The extended test contains 69 queries / 125 candidate skills, and the strict
research view contains 22 queries / 64 skills. The strict view's queries are
included in the extended view. Neither view uses the training split for scores.

The default context cap is 4096 tokens. All 0.6B models use BF16 on CUDA; the
8B embedding model uses NF4 double quantization to fit the local 8GB GPU.
Reranking uses each retriever's top 20. This is a local retrieval benchmark,
not end-to-end execution of the retrieved skills or a deployed HTTP service.

For a fresh setup, create `.venv`, install the CUDA 12.8 PyTorch wheel and
`config/requirements-local.txt`, clone the repositories listed in the source
manifest to their matching directories, then run:

```powershell
# Set these only if your network requires the Windows system proxy.
$taskProxyUri = [System.Net.WebRequest]::DefaultWebProxy.GetProxy([uri]'https://huggingface.co')
$env:HTTPS_PROXY = $taskProxyUri.AbsoluteUri
$env:HTTP_PROXY = $taskProxyUri.AbsoluteUri
python scripts/download_router_models.py
```

Downloads support resumable byte ranges and verify original Hugging Face
LFS/Git hashes. `config/requirements-local.lock.txt` records the measured run's
full package versions. Use a new `-Output` directory when changing evaluator
code, data, or context length; do not mix cached predictions across protocols.
