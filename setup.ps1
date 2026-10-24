# Vaani setup: creates .venv, installs dependencies, downloads models.
# Run from the project folder:  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "==> Creating virtual environment (Python 3.12)" -ForegroundColor Cyan
py -3.12 -m venv .venv
$py = ".\.venv\Scripts\python.exe"

Write-Host "==> Installing dependencies (onnxruntime-gpu + CUDA libs is ~1 GB)" -ForegroundColor Cyan
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt

Write-Host "==> Checking GPU" -ForegroundColor Cyan
& $py -c "import onnxruntime as o; print('Providers:', o.get_available_providers())"

Write-Host "==> Downloading Parakeet + Silero models" -ForegroundColor Cyan
& $py -m vaani --download

if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "==> Pulling cleanup model (gemma3:4b)" -ForegroundColor Cyan
    ollama pull gemma3:4b
} else {
    Write-Host "Ollama not found. Cleanup will be skipped until you install it from https://ollama.com and run: ollama pull gemma3:4b" -ForegroundColor Yellow
}

Write-Host "`nDone. Start Vaani with run.bat" -ForegroundColor Green
