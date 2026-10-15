import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[1]
PUBLIC_DOCUMENTS = (
    Path("README.md"),
    Path("docs/architecture.md"),
    Path("docs/configuration.md"),
    Path("docs/api.md"),
    Path("docs/troubleshooting.md"),
    Path("docs/versioning.md"),
)
# Markdown links and images, plus HTML src/href attributes used by the README.
MARKDOWN_LINK = re.compile(r"!?\[[^]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HTML_LINK = re.compile(r"""(?:src|href)=["']([^"']+)["']""")


@pytest.mark.parametrize("path", PUBLIC_DOCUMENTS)
def test_public_document_exists(path: Path) -> None:
    assert (PROJECT_ROOT / path).is_file()


def test_readme_relative_links_resolve() -> None:
    readme = PROJECT_ROOT / "README.md"
    content = readme.read_text(encoding="utf-8")
    targets = MARKDOWN_LINK.findall(content) + HTML_LINK.findall(content)
    for raw_target in targets:
        target = raw_target.split("#", maxsplit=1)[0].split("?", maxsplit=1)[0]
        if not target or "://" in target or target.startswith(("mailto:", "data:")):
            continue
        assert (readme.parent / target).resolve().exists(), (
            f"README.md: unresolved link {raw_target}"
        )
