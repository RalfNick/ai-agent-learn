FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/phase-5-production/01-fastapi-backend

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY phase-5-production/01-fastapi-backend/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY phase-5-production/01-fastapi-backend/app ./app
COPY docs /app/docs
COPY phase-2-rag/05-rag-benchmark/outputs /app/phase-2-rag/05-rag-benchmark/outputs
COPY phase-3-frameworks/02-agentic-rag-langgraph/outputs /app/phase-3-frameworks/02-agentic-rag-langgraph/outputs
COPY phase-4-advanced/03-memory-system /app/phase-4-advanced/03-memory-system
COPY phase-4-advanced/04-multi-agent-patterns /app/phase-4-advanced/04-multi-agent-patterns
COPY phase-4-advanced/05-agent-runtime-integration /app/phase-4-advanced/05-agent-runtime-integration

RUN mkdir -p .memory

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
