"""Error codes of the knowledge module. Every code starts with KNOWLEDGE_."""

from app.core.errors import AppError


def document_not_found() -> AppError:
    return AppError("KNOWLEDGE_DOCUMENT_NOT_FOUND", "This document does not exist.", 404)


def parse_failed() -> AppError:
    return AppError("KNOWLEDGE_PARSE_FAILED", "The PDF could not be read.", 422)


def not_a_pdf() -> AppError:
    return AppError("KNOWLEDGE_NOT_A_PDF", "Only PDF files can be uploaded.", 415)


def file_too_large(max_mb: int) -> AppError:
    return AppError("KNOWLEDGE_FILE_TOO_LARGE", f"The file is larger than {max_mb} MB.", 413)


def upload_not_allowed() -> AppError:
    return AppError("KNOWLEDGE_UPLOAD_NOT_ALLOWED", "You are not allowed to upload documents.", 403)


def status_not_allowed() -> AppError:
    return AppError(
        "KNOWLEDGE_STATUS_NOT_ALLOWED", "You are not allowed to see the ingestion status.", 403
    )


def too_many_pages(max_pages: int) -> AppError:
    return AppError("KNOWLEDGE_TOO_MANY_PAGES", f"The PDF has more than {max_pages} pages.", 413)


def external_embeddings_off() -> AppError:
    return AppError(
        "KNOWLEDGE_EXTERNAL_EMBEDDINGS_OFF",
        "Sending text to the AI provider is off (needs ALLOW_EXTERNAL_EMBEDDINGS and AI_ENABLED).",
        409,
    )


def playground_not_allowed() -> AppError:
    return AppError(
        "KNOWLEDGE_PLAYGROUND_NOT_ALLOWED", "You are not allowed to use the playground.", 403
    )


def chunk_not_found() -> AppError:
    return AppError("KNOWLEDGE_CHUNK_NOT_FOUND", "This text piece does not exist.", 404)


def nothing_embedded() -> AppError:
    return AppError(
        "KNOWLEDGE_NOTHING_EMBEDDED",
        "No text has been embedded yet. Add text and let the worker run first.",
        409,
    )
