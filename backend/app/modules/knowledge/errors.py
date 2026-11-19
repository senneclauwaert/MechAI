"""Error codes of the knowledge module. Every code starts with KNOWLEDGE_."""

from app.core.errors import AppError


def chunk_not_found() -> AppError:
    return AppError("KNOWLEDGE_CHUNK_NOT_FOUND", "This text piece does not exist.", 404)


def nothing_embedded() -> AppError:
    return AppError(
        "KNOWLEDGE_NOTHING_EMBEDDED",
        "No text has been embedded yet. Add text and let the worker run first.",
        409,
    )
