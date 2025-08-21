#!/usr/bin/env bash
set -e
docker compose up -d --build
python - <<'PY'
import time, webbrowser; time.sleep(2); webbrowser.open("http://localhost:8501")
PY
echo "Running at http://localhost:8501"
