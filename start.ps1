docker compose up -d --build
Start-Sleep -Seconds 2
Start-Process "http://localhost:8501"