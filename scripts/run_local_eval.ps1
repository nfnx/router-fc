param(
    [string]$Methods = 'bm25,qwen,skillret,skillrouter,r3,skillret8b',
    [string]$Output = 'runs/research-test-20261029'
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv/Scripts/python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONHASHSEED = '0'
$env:HF_HUB_OFFLINE = '1'
Push-Location $taskRoot
try {
    & $taskPython scripts/verify_router_metrics.py
    if ($LASTEXITCODE -ne 0) { throw 'Metric/data validation failed' }
    & $taskPython -u scripts/evaluate_local_routers.py --methods $Methods --output $Output
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed; inspect logs before resuming' }
    & $taskPython scripts/report_router_eval.py --run $Output
    if ($LASTEXITCODE -ne 0) { throw 'Report validation failed' }
    & $taskPython scripts/plot_router_eval.py
    if ($LASTEXITCODE -ne 0) { throw 'Plot generation failed' }
} finally {
    Pop-Location
}
