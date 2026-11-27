import pytest

from logo_gen import config, library, paths


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway project folder: its own database, settings, output and .env."""
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(paths, "SETTINGS_FILE", tmp_path / "data" / "settings.json")
    monkeypatch.setattr(paths, "DB_FILE", tmp_path / "data" / "library.sqlite3")
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(paths, "UPLOADS_DIR", tmp_path / "output" / "uploads")
    monkeypatch.setattr(library, "_conn", None)
    monkeypatch.setattr(config, "_settings", None)
    paths.ensure_dirs()
    yield tmp_path
    if library._conn is not None:
        library._conn.close()
