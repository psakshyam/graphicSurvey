FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
COPY app/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY app /app
CMD ["streamlit","run","/app/app.py","--server.address=0.0.0.0","--server.port=8501"]
