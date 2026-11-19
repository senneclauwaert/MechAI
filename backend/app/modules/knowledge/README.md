# knowledge module

Owns the handbook data: documents, pages, sections and chunks (schema `knowledge`), and the
hybrid search over the chunks.

**Now:** a playground to try RAG.
- `POST /api/v1/knowledge/playground` with a title and a text: saves it as a chunk and queues an
  `embed_text` job. The worker embeds it with Mistral (or the fake client without an API key).
- `GET /api/v1/knowledge/search?q=...`: hybrid search (vector top 20 + full-text top 20, merged
  with RRF) and returns the best 5 chunks with both ranks, so you can see which search found what.

**Later:** PDF ingestion (parse, sections, chunks of about 300 tokens, page images).

## Files

| File | What goes in it |
| --- | --- |
| `models.py` | The tables (SQLAlchemy) |
| `schemas.py` | Request and response shapes (Pydantic) |
| `service.py` | The logic. The only file other modules may call |
| `router.py` | The endpoints. Thin, only calls `service.py` |
| `errors.py` | Error codes, all starting with `KNOWLEDGE_` |

## Copy this module for a new feature

1. Copy this folder to `app/modules/<feature>/` and rename the classes and the error prefix.
2. Change `models.py`, import it in `migrations/env.py`, then run
   `alembic revision --autogenerate -m "<feature>"` and check the generated file.
3. Add the router in `app/main.py` (one line).
