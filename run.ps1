# Allow running this script: in PowerShell once: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
$ErrorActionPreference = "Stop"

# 1) install uv if missing
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "Installing uv (local, no admin)..."
  $install = Invoke-WebRequest -UseBasicParsing https://astral.sh/uv/install.ps1
  Invoke-Expression $install.Content
  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

# 2) create local venv if missing
if (-not (Test-Path ".venv")) {
  uv venv --python 3.11
}

# 3) install deps
uv pip install -r requirements.txt

# 4) prefill local LLM defaults (safe even if Ollama isn't installed)
if (-not $env:OLLAMA_HOST) { $env:OLLAMA_HOST = "http://127.0.0.1:11434" }
if (-not $env:OLLAMA_MODEL) { $env:OLLAMA_MODEL = "llama3.1:8b" }

# 5) run (localhost only)
uv run streamlit run app.py --server.address=127.0.0.1 --server.port=8501
