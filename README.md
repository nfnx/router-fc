# RethosAI-Router-Dev
Agent router for Rethos.ai agent framework

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
