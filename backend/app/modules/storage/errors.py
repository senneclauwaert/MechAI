"""Error codes of the storage module. Every code starts with STORAGE_."""

from app.core.errors import AppError


def file_missing() -> AppError:
    return AppError("STORAGE_FILE_MISSING", "This file does not exist.", 404)


def path_not_allowed() -> AppError:
    return AppError("STORAGE_PATH_NOT_ALLOWED", "This file location is not allowed.", 400)
