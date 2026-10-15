from __future__ import annotations

import subprocess
import sys

from fish_memory.runtime_memory import release_unused_memory


def test_desktop_runtime_does_not_eagerly_import_both_roles() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, fish_memory.desktop_runtime; "
            "print(int('fish_memory.app' in sys.modules), "
            "int('fish_memory.companion.runner' in sys.modules))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "0 0"


def test_release_unused_memory_is_safe() -> None:
    assert release_unused_memory("test") >= 0
