# storage module

Owns the files volume (`FILES_DIR`): handbook pages, photos and other files.

**Now:** a minimal service, no tables and no endpoints.
- `save_file(folder, name, data)` writes `<files_dir>/<folder>/<name>`, creates the folders,
  overwrites an existing file and returns the relative path `"<folder>/<name>"`.
- `open_file(relative_path)` returns the bytes, or raises `STORAGE_FILE_MISSING` (404).
- Both refuse a path outside `files_dir` (`STORAGE_PATH_NOT_ALLOWED`, 400): no `..`, no absolute
  paths. Folder and name are always built by our code, never taken from the client.

**Later (#27):** `models.py`, `schemas.py` and `router.py` for uploads (type allow-list and size
limit) and for serving files.

## Files

| File | What goes in it |
| --- | --- |
| `service.py` | The logic. The only file other modules may call |
| `errors.py` | Error codes, all starting with `STORAGE_` |
