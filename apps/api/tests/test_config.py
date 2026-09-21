"""Settings bootstrap — the walk-up ``.env`` discovery + CSV list fields."""

from pathlib import Path

from app.config import Settings, _find_env_file


def test_find_env_file_walks_up_to_nearest(tmp_path):
    (tmp_path / "deep" / "nested").mkdir(parents=True)
    (tmp_path / ".env").write_text("POSTGRES_PASSWORD=from-root", encoding="utf-8")

    found = _find_env_file(tmp_path / "deep" / "nested")

    assert found is not None
    assert Path(found) == tmp_path / ".env"


def test_find_env_file_nearest_wins(tmp_path):
    (tmp_path / "deep").mkdir()
    (tmp_path / ".env").write_text("POSTGRES_PASSWORD=from-root", encoding="utf-8")
    (tmp_path / "deep" / ".env").write_text("POSTGRES_PASSWORD=from-deep", encoding="utf-8")

    found = _find_env_file(tmp_path / "deep")

    assert Path(found) == tmp_path / "deep" / ".env"


def test_find_env_file_returns_none_when_absent(tmp_path):
    assert _find_env_file(tmp_path) is None


def test_cors_origins_accept_csv_string():
    settings = Settings(cors_allow_origins="http://a:1, http://b:2")
    assert settings.cors_allow_origins == ["http://a:1", "http://b:2"]


def test_cors_origins_accept_json_string():
    settings = Settings(cors_allow_origins='["http://a:1"]')
    assert settings.cors_allow_origins == ["http://a:1"]


def test_cors_origins_default_list_untouched():
    assert "http://localhost:3000" in Settings().cors_allow_origins


def test_allowed_file_types_accept_csv_string():
    assert Settings(allowed_file_types="pdf,txt").allowed_file_types == ["pdf", "txt"]
