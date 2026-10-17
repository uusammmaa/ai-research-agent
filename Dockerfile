FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN useradd --create-home --uid 10001 app

# Dependencies first so code changes don't invalidate the install layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY config.py main.py ./
COPY agent ./agent

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

# Keys and tracing settings come from the environment at run time, e.g.
#   docker run --env-file .env -p 8000:8000 ai-research-agent
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
