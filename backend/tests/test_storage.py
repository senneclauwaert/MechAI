import pytest

from app.config import get_settings
from app.core.errors import AppError
from app.modules.storage import service


@pytest.fixture(autouse=True)
def files_dir(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "files_dir", str(tmp_path))


def test_save_then_open_returns_same_bytes() -> None:
    path = service.save_file("pages/doc1", "p1.png", b"abc")
    assert path == "pages/doc1/p1.png"
    assert service.open_file(path) == b"abc"


def test_saving_twice_overwrites() -> None:
    service.save_file("pages", "p1.png", b"old")
    path = service.save_file("pages", "p1.png", b"new")
    assert service.open_file(path) == b"new"


def test_open_missing_file() -> None:
    with pytest.raises(AppError) as error:
        service.open_file("pages/nope.png")
    assert error.value.code == "STORAGE_FILE_MISSING"
    assert error.value.status == 404


@pytest.mark.parametrize("bad_path", ["../x", "/etc/passwd"])
def test_path_outside_files_dir_is_refused(bad_path: str) -> None:
    with pytest.raises(AppError) as error:
        service.open_file(bad_path)
    assert error.value.code == "STORAGE_PATH_NOT_ALLOWED"
    assert error.value.status == 400


def test_save_outside_files_dir_is_refused() -> None:
    with pytest.raises(AppError) as error:
        service.save_file("..", "x", b"abc")
    assert error.value.code == "STORAGE_PATH_NOT_ALLOWED"
