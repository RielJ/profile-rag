# The profile-rag API for a deployed demo: FastAPI, fastembed, and an
# OpenAI-compatible answer model (Groq, OpenRouter). No Ollama, no Pi.
#
# The image holds code and embedding weights only. The corpus is not in it:
# the chunks and vectors live in Postgres, filled from your machine with
# `just index` (docs/DEPLOY.md). Secrets come from the host's environment.
#
#   docker build -t profile-rag-api .
#   docker run --rm -p 7860:7860 --env-file deploy.env profile-rag-api

FROM python:3.14-slim

# backend/pyproject.toml requires uv >= 0.13.6.
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv

# Hugging Face Spaces runs the container as user 1000.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    FASTEMBED_CACHE_PATH=/app/models \
    PATH=/app/.venv/bin:$PATH

# Dependencies first, so a code change does not reinstall them.
COPY backend/pyproject.toml backend/uv.lock backend/README.md backend/
RUN uv sync --directory backend --frozen --no-dev --no-install-project

COPY backend/src backend/src
RUN uv sync --directory backend --frozen --no-dev

# Download the embedding weights now, so the server never needs the network
# for them. The name must match vectorstore/models.py.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('nomic-ai/nomic-embed-text-v1.5')" \
    && chown -R app:app /app/models

USER app

# The deployed defaults. Set the rest on the host (docs/DEPLOY.md):
# DATABASE_URL, PROFILE_RAG_LLM_API_KEY, PROFILE_RAG_ACCESS_KEY, and the limits.
ENV PROFILE_RAG_LLM=openai \
    PROFILE_RAG_EMBEDDER=fastembed \
    PORT=7860

EXPOSE 7860

# --proxy-headers is off: the client address comes from the portfolio proxy's
# X-Visitor-IP header, trusted only with the access key (api/access.py).
CMD ["sh", "-c", "exec uvicorn profile_rag.api.app:app --host 0.0.0.0 --port ${PORT}"]
