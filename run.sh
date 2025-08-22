#!/usr/bin/env bash
set -euo pipefail

# resolve to script directory
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 1) install uv if missing
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (local, no admin)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"

# 2) create a local venv (in this folder) if missing
if [ ! -d ".venv" ]; then
  uv venv --python 3.11
fi

# 3) install deps into the local venv (cached; fast next time)
uv pip install -r requirements.txt

# 4) prefill local LLM defaults (safe even if Ollama isn't installed)
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"

# 5) run the app (localhost only)
uv run streamlit run app.py --server.address=127.0.0.1 --server.port=8501
