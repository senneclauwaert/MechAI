# One image for both the api and the worker (they only differ in the start command).
# Stages: "base" installs the libraries, "dev" adds the test tools, "prod" adds the code.

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app
WORKDIR /app
# Install only the libraries from pyproject.toml (not our own code), so Docker can reuse
# this layer when only the code changes. The code itself is found through PYTHONPATH.
COPY pyproject.toml ./
RUN mkdir app && touch app/__init__.py \
    && pip install . && pip uninstall -y mechai-backend \
    && rm -rf app build *.egg-info

# Used by compose.dev.yaml: the code is mounted from your laptop, so it is not copied in.
FROM base AS dev
RUN mkdir app && touch app/__init__.py \
    && pip install ".[dev]" && pip uninstall -y mechai-backend \
    && rm -rf app build *.egg-info
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Used on the server.
FROM base AS prod
COPY . .
RUN useradd --create-home --uid 1000 mechai \
    && mkdir -p /data/files && chown mechai /data/files
USER mechai
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
