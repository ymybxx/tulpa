import json
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[1]
GOVERNANCE_FILES = (
    Path("LICENSE"),
    Path("CONTRIBUTING.md"),
    Path("CODE_OF_CONDUCT.md"),
    Path("SECURITY.md"),
    Path("CHANGELOG.md"),
    Path("docs/versioning.md"),
)
GITHUB_FILES = (
    Path(".github/workflows/ci.yml"),
    Path(".github/workflows/windows-desktop.yml"),
    Path(".github/dependabot.yml"),
    Path(".github/ISSUE_TEMPLATE/bug_report.yml"),
    Path(".github/ISSUE_TEMPLATE/feature_request.yml"),
    Path(".github/ISSUE_TEMPLATE/config.yml"),
    Path(".github/pull_request_template.md"),
)
REPOSITORY_URL = "https://github.com/ymybxx/tulpa"


@pytest.mark.parametrize("path", GOVERNANCE_FILES)
def test_required_open_source_file_exists(path: Path) -> None:
    assert (PROJECT_ROOT / path).is_file()


def test_package_declares_public_metadata() -> None:
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert project["name"] == "tulpa"
    assert project["readme"] == "README.md"
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert project["urls"]["Repository"] == REPOSITORY_URL
    assert "Programming Language :: Python :: 3.11" in project["classifiers"]
    assert not any(
        classifier.startswith("License ::") for classifier in project["classifiers"]
    )


def test_desktop_package_declares_public_metadata() -> None:
    package = json.loads(
        (PROJECT_ROOT / "desktop/pet/package.json").read_text(encoding="utf-8")
    )

    assert package["name"] == "tulpa"
    assert package["license"] == "MIT"
    assert package["repository"]["url"] == f"git+{REPOSITORY_URL}.git"


@pytest.mark.parametrize("path", GITHUB_FILES)
def test_required_github_file_exists(path: Path) -> None:
    assert (PROJECT_ROOT / path).is_file()


def test_ci_runs_python_and_desktop_checks() -> None:
    content = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    for required in (
        "uv sync --locked --dev",
        "ruff format --check",
        "ruff check",
        "ty check",
        "pytest -q",
        "FISH_MEMORY_TEST_DATABASE_URL",
        "tests/integration",
        "npm ci",
        "npm run typecheck",
        "npx vitest run",
    ):
        assert required in content


def test_dependabot_tracks_uv_and_npm() -> None:
    content = (PROJECT_ROOT / ".github/dependabot.yml").read_text(encoding="utf-8")

    assert 'package-ecosystem: "uv"' in content
    assert 'package-ecosystem: "npm"' in content
    assert 'directory: "/desktop/pet"' in content
    assert 'package-ecosystem: "docker"' not in content
