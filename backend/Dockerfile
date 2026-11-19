# One image for both the api and the worker (they only differ in the start command).
# Stages: "base" installs the libraries, "dev" adds the test tools, "prod" adds the code.

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app
RUN pip install --no-cache-dir "uv==0.11.*"
WORKDIR /app
# Install the libraries first, exactly as pinned in uv.lock, so Docker can reuse this layer
# when only the code changes. The virtual environment lives in /opt/venv, outside /app,
# so mounting your code over /app in dev does not hide it.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

# Used by compose.dev.yaml: the code is mounted from your laptop, so it is not copied in.
FROM base AS dev
RUN uv sync --locked
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Used on the server.
FROM base AS prod
COPY . .
RUN useradd --create-home --uid 1000 mechai \
    && mkdir -p /data/files && chown mechai /data/files
USER mechai
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
