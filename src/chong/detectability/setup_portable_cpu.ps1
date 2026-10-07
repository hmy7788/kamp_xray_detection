# Run inside a freshly extracted portable snapshot. No automatic training.
param([string]$PythonExe = '')
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$env:PYTHONUTF8 = '1'
$venv = Join-Path $repo 'runs/chong/local/work/portable-venv-cpu'
$python = Join-Path $venv 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    if ($PythonExe) {
        & $PythonExe -m venv $venv
    } else {
        & py -3.13 -m venv $venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.13 is required. Install it or pass -PythonExe with its executable path.' }
}
& $python (Join-Path $PSScriptRoot 'restore_portable.py') --verify
if ($LASTEXITCODE -ne 0) { throw 'Portable file verification failed.' }
& $python -m pip install -r (Join-Path $PSScriptRoot 'requirements-cpu.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& (Join-Path $PSScriptRoot 'prepare.ps1')
if ($LASTEXITCODE -ne 0) { throw 'Data path preparation failed.' }
$manifest = Get-Content -LiteralPath (Join-Path $repo 'runs/chong/06_deployment_compare/work/standalone_inputs_portable.json') -Raw -Encoding UTF8 | ConvertFrom-Json
& $python (Join-Path $PSScriptRoot 'onnx_cpu.py') --model (Join-Path $repo 'runs/chong/06_deployment_compare/exports/fp32/baseline_640.onnx') --image $manifest.images[0].path --threads 1 --threshold 0.42
if ($LASTEXITCODE -ne 0) { throw 'CPU inference smoke check failed.' }
Write-Output 'CPU setup complete. One validation image was processed; no training or test evaluation was run.'
