# knowledge module

Owns the handbook data: documents, pages, sections and chunks (schema `knowledge`), and the
hybrid search over the chunks.

**Now:** a playground to try RAG.
- `POST /api/v1/knowledge/playground` (dev only, like the upload) with a title and a text: saves it as a chunk and queues an
  `embed_text` job. The worker embeds it with Mistral (or the fake client without an API key).
- `GET /api/v1/knowledge/search?q=...`: hybrid search (vector top 20 + full-text top 20, merged
  with RRF) and returns the best 5 chunks with both ranks, so you can see which search found what.

**For the assistant** (called from Python, no endpoint):
- `retrieve_sections(db, query, keywords, part_numbers, *, use_embeddings=True)` returns a
  `Retrieval`: the best 20 chunks with their ranks and similarity, the parent sections to give to
  the answer step, `best_similarity` and `embedding_error`. It only searches current handbook text
  (`source_type = handbook`, not superseded); Playground text never counts. `query` is a full
  English sentence for the vector search (top 20 by cosine distance, `similarity = 1 - distance`;
  the refusal threshold is measured on this score, never on the RRF score). The full-text search
  (top 20) ORs the keywords, because `websearch_to_tsquery` would AND them, and chunks with one of
  the part numbers (max 10, max 50 characters) come first. Both lists are merged with RRF; the
  sections are kept in order of their best chunk, at most `ANSWER_SECTIONS` of them, and a section
  that would pass `ANSWER_SOURCE_TOKENS` is skipped (the first one is always kept). A section is
  never cut. It raises `KNOWLEDGE_NOTHING_EMBEDDED` when there is no handbook chunk (or none
  embedded with the current model). If the query embedding fails, the text hits are still returned
  and `embedding_error` holds the code. With `use_embeddings=False`, or with `AI_ENABLED=false`,
  no AI client is created and there is no vector search (text hits only).
  `best_similarity` is the highest similarity among chunks of the sections that are really
  returned. A similarity that is not a number (a zero vector) counts as no score (`None`), and
  that chunk is not a vector hit. NUL and other control characters are replaced by a space in
  the query, the keywords and the part numbers; when the keywords hold no usable word, the words
  of the query are used. Search terms are never logged (the database engine hides parameters).
  The vector search needs pgvector 0.8.0 or newer (`hnsw.iterative_scan`); `compose.yaml` pins it.
- `external_text_allowed(llm)`: true for the fake client, else only when `ALLOW_EXTERNAL_EMBEDDINGS`
  and `AI_ENABLED` are both `true`. Embedding uses it, and so does the assistant before it sends
  handbook text to the chat model.

**Handbook ingestion** (dev only until #26 adds the admin check; refused when `ENVIRONMENT=prod`):
- `POST /api/v1/knowledge/documents`: multipart with `file` (a PDF), `title` (1-300), `revision`
  (optional, max 50) and `kart_model` (optional, max 100). Returns `201` for a new file, or `200`
  with `already_ingested: true` for a file that is known. The size limit is `MAX_PDF_MB`; the
  file is checked by its first bytes (`%PDF-`), never by its name or content type. It is saved as
  `knowledge/<document id>/source.pdf` on the files volume.
- `GET /api/v1/knowledge/documents/{document_id}/status`: `status` (`queued`, `running`, `done`,
  `failed`), `error_code` when failed, and the number of pages, sections and chunks saved.
- The worker job `ingest_document` (payload `{"document_id": ...}`) reads the PDF with Docling,
  saves the pages (text and PNG), sections and chunks, and embeds the chunks, in one transaction.
  Before it parses anything it checks two settings: `ALLOW_EXTERNAL_EMBEDDINGS` (default `false`)
  and `MAX_PDF_PAGES` (default 500, checked by the worker; the api never opens the PDF). Every embedding goes through
  `embed_chunks`, which only lets the real Mistral client work when `ALLOW_EXTERNAL_EMBEDDINGS`
  and `AI_ENABLED` are both `true` (handbook text may go to Mistral only after Supercube's written
  OK); the fake client always works. Page images are at most 6000 px on the longest side.

**Re-upload rule** (SHA-256 of the file): the same file again returns the existing document and
queues nothing, unless its job failed: then a new job is queued for the same document. Two uploads
of the same file at the same moment end up with one document (the database refuses the second).

**Error messages** (the app shows the translation of the code):

| Code | nl | fr | en |
| --- | --- | --- | --- |
| `KNOWLEDGE_DOCUMENT_NOT_FOUND` | Dit document bestaat niet. | Ce document n'existe pas. | This document does not exist. |
| `KNOWLEDGE_PARSE_FAILED` | De PDF kon niet gelezen worden. | Le PDF n'a pas pu etre lu. | The PDF could not be read. |
| `KNOWLEDGE_NOT_A_PDF` | Alleen PDF-bestanden kunnen geupload worden. | Seuls les fichiers PDF peuvent etre televerses. | Only PDF files can be uploaded. |
| `KNOWLEDGE_FILE_TOO_LARGE` | Het bestand is groter dan de toegelaten grootte. | Le fichier depasse la taille autorisee. | The file is larger than the allowed size. |
| `KNOWLEDGE_UPLOAD_NOT_ALLOWED` | Je mag geen documenten uploaden. | Vous n'avez pas le droit de televerser des documents. | You are not allowed to upload documents. |
| `KNOWLEDGE_STATUS_NOT_ALLOWED` | Je mag de verwerkingsstatus niet bekijken. | Vous n'avez pas le droit de voir l'etat du traitement. | You are not allowed to see the ingestion status. |
| `KNOWLEDGE_TOO_MANY_PAGES` | De PDF heeft te veel pagina's. | Le PDF a trop de pages. | The PDF has more than the allowed number of pages. |
| `KNOWLEDGE_EXTERNAL_EMBEDDINGS_OFF` | Tekst naar de AI-aanbieder sturen staat uit (ALLOW_EXTERNAL_EMBEDDINGS en AI_ENABLED). | L'envoi de texte au fournisseur d'IA est desactive (ALLOW_EXTERNAL_EMBEDDINGS et AI_ENABLED). | Sending text to the AI provider is off (needs ALLOW_EXTERNAL_EMBEDDINGS and AI_ENABLED). |
| `KNOWLEDGE_PLAYGROUND_NOT_ALLOWED` | Je mag de playground niet gebruiken. | Vous n'avez pas le droit d'utiliser le playground. | You are not allowed to use the playground. |
| `STORAGE_FILE_MISSING` | Dit bestand bestaat niet. | Ce fichier n'existe pas. | This file does not exist. |
| `STORAGE_PATH_NOT_ALLOWED` | Deze bestandslocatie is niet toegelaten. | Cet emplacement de fichier n'est pas autorise. | This file location is not allowed. |

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
