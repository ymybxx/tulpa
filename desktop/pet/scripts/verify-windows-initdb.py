"""Run actual initdb command variants before compiling the frozen runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fish_memory.windows_postgres_diagnostics import run_initdb_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--postgres-root", type=Path, required=True)
    args = parser.parse_args()
    report = run_initdb_matrix(args.postgres_root.resolve(), copy_installation=True)
    print(json.dumps({"windowsInitdbMatrix": report}), flush=True)
    cases = report.get("cases", [])
    if len(cases) != 6 or not all(
        case.get("returnCode") == 0 and case.get("dataInitialized") is True
        for case in cases
    ):
        raise RuntimeError("Actual Windows initdb command matrix did not pass")


if __name__ == "__main__":
    main()
