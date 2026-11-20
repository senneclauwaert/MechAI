# knowledge module

Owns the handbook data: documents, pages, sections and chunks (schema `knowledge`), and the
hybrid search over the chunks.

**Now:** a playground to try RAG.
- `POST /api/v1/knowledge/playground` (dev only, like the upload) with a title and a text: saves it as a chunk and queues an
  `embed_text` job. The worker embeds it with Mistral (or the fake client without an API key).
- `GET /api/v1/knowledge/search?q=...`: hybrid search (vector top 20 + full-text top 20, merged
  with RRF) and returns the best 5 chunks with both ranks, so you can see which search found what.

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
  and `MAX_PDF_PAGES` (default 500, also checked at upload). Every embedding goes through
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
