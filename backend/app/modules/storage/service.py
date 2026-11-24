"""Save and open files on the files volume. The only file other modules may call.

Folder and name are always built by our code, never taken from the client.
"""

from pathlib import Path

from app.config import get_settings
from app.modules.storage.errors import file_missing, path_not_allowed


def _full_path(relative_path: str) -> Path:
    """Resolve the path and refuse anything outside the files folder."""
    root = Path(get_settings().files_dir).resolve()
    full = (root / relative_path).resolve()
    if not full.is_relative_to(root):
        raise path_not_allowed()
    return full


def save_file(folder: str, name: str, data: bytes) -> str:
    """Write the file (overwriting an existing one) and return its relative path."""
    relative_path = f"{folder}/{name}"
    full = _full_path(relative_path)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(data)
    return relative_path


# Read a file back; raises STORAGE_FILE_MISSING when it is not there.
def open_file(relative_path: str) -> bytes:
    full = _full_path(relative_path)
    if not full.is_file():
        raise file_missing()
    return full.read_bytes()
