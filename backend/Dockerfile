# Local image for the API, worker and model server. Phase 11 turns this into a slimmer
# multi-stage build.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STORAGE_DIR=/data/uploads \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

# `--build-arg EXTRAS=local` adds local models (bge-m3 embeddings, cross-encoder reranker).
ARG EXTRAS=""

# Heavy model dependencies get their own layer so code changes do not reinstall them.
# CPU-only PyTorch: the default Linux wheels bundle several GB of CUDA libraries.
RUN if echo "$EXTRAS" | grep -q local; then \
        pip install torch --index-url https://download.pytorch.org/whl/cpu \
        && pip install "sentence-transformers>=4,<6"; \
    fi

COPY pyproject.toml ./
COPY app ./app
RUN if [ -n "$EXTRAS" ]; then pip install ".[$EXTRAS]"; else pip install .; fi

COPY alembic.ini ./

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data/uploads "$HF_HOME" \
    && chown -R app:app /data/uploads /home/app/.cache
USER app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
