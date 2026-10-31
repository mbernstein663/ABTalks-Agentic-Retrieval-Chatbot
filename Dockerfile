# ---------- Builder ----------
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---------- Final ----------
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /install /usr/local

# Dependencies are now available to normal Python
RUN python -m spacy download en_core_web_lg

# Pre-cache tokenizer so Kubernetes doesn't download it at startup
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]