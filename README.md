# MechAi

Repair help for the mechanics at Supercube: a chatbot that answers with cited handbook pages,
plus kart history, issues, stock and to-dos.

Authors: Nehad and Senne

## What runs where

| Container | Built from | What it does |
| --- | --- | --- |
| `db` | `pgvector/pgvector:pg16` (no Dockerfile) | PostgreSQL 16 with pgvector, one schema per module |
| `api` | `backend/Dockerfile` | FastAPI. Runs the migrations at start |
| `worker` | `backend/Dockerfile` (same image) | Takes jobs from `core.jobs` (embeddings, ingestion, sync) |
| `web` | `web/Dockerfile` | Builds the React app and serves it with nginx; forwards `/api` to `api` |
| `frontend` | `node:22-alpine` (dev only) | The Vite dev server with hot reload |

Two compose files:

- `compose.yaml` is the real setup, as it runs on the server.
- `compose.dev.yaml` only holds what is different on your laptop: your code mounted into the
  containers, automatic reload, the Vite dev server and the database port open.

## Start on your laptop

You need Git and Docker Desktop. [uv](https://docs.astral.sh/uv/) and Node 22 are only needed
to run things outside Docker (and so your editor knows the libraries).

```bash
cp .env.example .env          # then fill in MISTRAL_API_KEY (or leave it empty for the fake AI)
docker compose -f compose.yaml -f compose.dev.yaml up --build
```

| Open | What |
| --- | --- |
| http://localhost:5173 | The app (Vite dev server) |
| http://localhost:8000/api/docs | The API docs, where you can try every endpoint |
| `localhost:5432` | The database (user, password and name from `.env`) |

Save a Python file and the api and worker restart by themselves. Save a frontend file and the
browser updates. Stop everything with `Ctrl+C`, or `docker compose -f compose.yaml -f compose.dev.yaml down`
(add `-v` to also delete the database).

Tip: to type less, set `COMPOSE_FILE=compose.yaml;compose.dev.yaml` (Windows) or
`COMPOSE_FILE=compose.yaml:compose.dev.yaml` (macOS, Linux) in your terminal, then `docker compose up` is enough.

## Try the RAG playground and the worker

1. Open http://localhost:8000/api/docs.
2. `POST /api/v1/knowledge/playground` with a title and a piece of text. This saves it as a chunk
   and puts an `embed_text` job in the queue.
3. The worker picks up the job within 2 seconds and stores the embedding
   (watch it with `docker compose -f compose.yaml -f compose.dev.yaml logs -f worker`).
4. `GET /api/v1/knowledge/search?q=...` runs the hybrid search (vector + full text, merged with
   RRF) and shows the rank from each search, so you can see which one found what.

Without a `MISTRAL_API_KEY` the fake AI client is used: it makes simple word-count vectors, so
the search still works roughly and nothing costs money. With a key, `mistral-embed` is used.

To test the worker on its own: `POST /api/v1/dev/jobs` with `{"type": "echo", "payload": {"msg": "hi"}}`
(or `"type": "fail"` to see the retries). The `/dev` endpoints only exist when `ENVIRONMENT=dev`.

## Common commands

Run these inside the api container, so you don't need Python on your laptop:

```bash
docker compose -f compose.yaml -f compose.dev.yaml exec api pytest          # tests
docker compose -f compose.yaml -f compose.dev.yaml exec api ruff check .    # lint
docker compose -f compose.yaml -f compose.dev.yaml exec api ruff format .   # format
docker compose -f compose.yaml -f compose.dev.yaml exec api mypy app        # type check

# after changing a models.py: create a migration, then check the new file in migrations/versions
docker compose -f compose.yaml -f compose.dev.yaml exec api alembic revision --autogenerate -m "short description"
docker compose -f compose.yaml -f compose.dev.yaml exec api alembic upgrade head
```

## Python libraries with uv

The backend uses [uv](https://docs.astral.sh/uv/) instead of pip. `pyproject.toml` lists the
libraries, `uv.lock` pins their exact versions, so everyone (and the server) gets the same ones.

```bash
cd backend
uv sync                      # creates .venv with Python 3.12 and all libraries (also for VS Code)
uv add <library>             # add a library: updates pyproject.toml and uv.lock
uv add --dev <tool>          # add a development-only tool (tests, linters)
uv run pytest                # run something inside .venv without activating it
```

After `uv add`, commit both `pyproject.toml` and `uv.lock`, and rebuild the containers with
`docker compose -f compose.yaml -f compose.dev.yaml up --build`.

The frontend uses npm the same way: `package.json` lists the libraries, `package-lock.json`
pins them, `npm install <library>` adds one.

## Folder structure

```txt
backend/
  app/
    main.py          FastAPI app, one router per module
    worker.py        the job loop (same image as the api, other command)
    config.py        settings from .env
    llm.py           the only file that talks to Mistral (chat, embed, transcribe) + a fake client
    core/            database, job queue, errors, logging, health endpoint
    modules/
      knowledge/     handbook tables + hybrid search (copy this folder for a new module)
  migrations/        Alembic migrations
  tests/
frontend/            React + TypeScript + Vite
  src/
    components/ui/   shared buttons, cards, ...
    features/        one folder per feature
    i18n/            nl, fr, en
    lib/api.ts       one helper for every API call
web/                 nginx config + Dockerfile that builds the frontend
compose.yaml         the server setup
compose.dev.yaml     laptop overrides
```

## On the server

```bash
cp .env.example .env          # set ENVIRONMENT=prod, a strong POSTGRES_PASSWORD, VERSION, the API key
docker compose up -d
```

Only `web` opens a port. HTTPS with Supercube's certificate is added in the deployment task.
