"""Settings bootstrap — the walk-up ``.env`` discovery."""

from pathlib import Path

from app.config import _find_env_file


def test_find_env_file_walks_up_to_nearest(tmp_path):
    (tmp_path / "deep" / "nested").mkdir(parents=True)
    (tmp_path / ".env").write_text("REDIS_PASSWORD=from-root", encoding="utf-8")

    found = _find_env_file(tmp_path / "deep" / "nested")

    assert found is not None
    assert Path(found) == tmp_path / ".env"


def test_find_env_file_returns_none_when_absent(tmp_path):
    assert _find_env_file(tmp_path) is None
