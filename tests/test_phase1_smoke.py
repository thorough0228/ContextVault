"""Phase 1 smoke tests.

These orchestrate the per-app test suites via subprocess so each app's
``app/`` package resolves without colliding with the sibling app's
package of the same name. They also statically validate the
docker-compose file and the .env.example.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = REPO_ROOT / "infrastructure" / "docker-compose.yml"
PY = sys.executable  # the active interpreter (conda env in this container)


def _run(cmd, cwd=None, timeout=600):
    """Run ``cmd`` (list of str) and return CompletedProcess; raise on failure."""
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("LC_ALL", "C.UTF-8")
    return subprocess.run(
        cmd,
        cwd=cwd or str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------


def test_env_example_has_no_real_secrets():
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    for marker in ("AKIA", "BEGIN PRIVATE KEY", "ghp_"):
        assert marker not in text, f".env.example contains forbidden token: {marker}"
    assert "change-me-locally" in text


def test_docker_compose_yaml_parses():
    import yaml

    data = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    services = set(data["services"].keys())
    required = {"postgres", "redis", "minio", "minio-init", "api", "web", "worker"}
    missing = required - services
    assert not missing, f"missing services in compose: {missing}"


def test_directory_layout():
    expected = [
        "apps/api/app/main.py",
        "apps/api/tests/test_health.py",
        "apps/worker/app/celery_app.py",
        "apps/worker/app/worker_tasks.py",
        "apps/worker/tests/test_tasks.py",
        "apps/web/src/app/layout.tsx",
        "apps/web/src/app/login/page.tsx",
        "apps/web/src/app/register/page.tsx",
        "apps/web/src/app/dashboard/page.tsx",
        "apps/web/src/lib/api-client.ts",
        "apps/web/src/components/health-badge.tsx",
        "packages/shared/python/contextvault_shared/constants.py",
        "packages/shared/ts/index.ts",
        "infrastructure/docker-compose.yml",
        "tests/test_phase1_smoke.py",
        "docs/phase1.md",
        "README.md",
        ".env.example",
    ]
    for rel in expected:
        assert (REPO_ROOT / rel).exists(), f"missing {rel}"


# ---------------------------------------------------------------------------
# Per-app test orchestration (subprocess isolates the import namespaces)
# ---------------------------------------------------------------------------


def test_api_tests_pass():
    result = _run([PY, "-m", "pytest", "tests/", "-q"], cwd=str(REPO_ROOT / "apps" / "api"))
    assert result.returncode == 0, (
        f"API tests failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_worker_tests_pass():
    result = _run([PY, "-m", "pytest", "tests/", "-q"], cwd=str(REPO_ROOT / "apps" / "worker"))
    assert result.returncode == 0, (
        f"Worker tests failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_evals_offline_suite_pass():
    """Offline eval suites (deterministic retrieval regression, streaming
    contract, robustness mechanics, harness self-tests). The golden
    retrieval metrics run against the production embedding API (paid/
    rate-limited) so it is excluded here and run manually or via
    `pytest evals/suites -m online` style invocations — see
    evals/README.md."""
    result = _run(
        [PY, "-m", "pytest", "evals/suites", "-m", "offline",
         "-k", "not test_retrieval_metrics", "-q"],
        cwd=str(REPO_ROOT / "evals"),
        timeout=1800,
    )
    assert result.returncode == 0, (
        f"Eval offline suites failed.\nSTDOUT:\n{result.stdout[-3000:]}\n"
        f"STDERR:\n{result.stderr[-2000:]}"
    )


def test_frontend_build_succeeds():
    npm = shutil.which("npm")
    if npm is None:
        pytest.skip("npm not available in this environment")
    web_dir = REPO_ROOT / "apps" / "web"
    if not (web_dir / "node_modules").exists():
        install = _run([npm, "install", "--no-audit", "--no-fund"], cwd=str(web_dir))
        assert install.returncode == 0, install.stderr[-2000:]
    result = _run([npm, "run", "build"], cwd=str(web_dir))
    out = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, (
        f"next build failed.\nSTDOUT:\n{(result.stdout or '')[-2000:]}\n"
        f"STDERR:\n{(result.stderr or '')[-2000:]}"
    )
    assert "/dashboard" in out
    assert "/login" in out
    assert "/register" in out


# ---------------------------------------------------------------------------
# End-to-end in-process contract check
# ---------------------------------------------------------------------------


def test_shared_python_constants_consistent():
    """Importing each app's constants via subprocess proves the shared
    module imports cleanly under both runtimes."""
    code = (
        "from contextvault_shared import API_PREFIX, HEALTH_PATH, APP_NAME;"
        "assert API_PREFIX == '/api/v1';"
        "assert HEALTH_PATH == '/api/v1/health';"
        "assert APP_NAME == 'contextvault';"
        "print('shared-ok')"
    )
    shared_dir = REPO_ROOT / "packages" / "shared" / "python"
    result = _run(
        [PY, "-c", code],
        cwd=str(shared_dir),
    )
    assert result.returncode == 0, (
        f"shared import failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "shared-ok" in result.stdout


def test_env_example_is_valid_env_format():
    """Make sure every key=value pair parses; nothing more."""
    env = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    parsed = {}
    for line in env.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        parsed[key.strip()] = value.strip()
    assert "POSTGRES_PASSWORD" in parsed
    assert "JWT_SECRET" in parsed
    assert "S3_SECRET_KEY" in parsed
    # All three must be the placeholder
    for k in ("POSTGRES_PASSWORD", "JWT_SECRET", "S3_SECRET_KEY"):
        assert parsed[k] == "change-me-locally", f"{k} must be placeholder, got {parsed[k]!r}"