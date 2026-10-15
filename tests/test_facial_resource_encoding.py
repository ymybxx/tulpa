"""Bundled facial resources load on Windows with a legacy default code page."""

import json
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch


class FacialResourceEncodingTests(unittest.TestCase):
    def test_import_and_catalog_preserve_utf8_under_gbk_default(self):
        root = Path(__file__).resolve().parents[1]
        core = root / "src/fish_memory/facial_test/core.py"
        cases_file = core.with_name("cases.json")
        catalog_file = root / "desktop/pet/src/shared/facial-demo-parameters.json"
        expected_cases = json.loads(cases_file.read_text(encoding="utf-8"))
        expected_catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
        original_read_text = Path.read_text

        def legacy_read_text(path, encoding=None, errors=None):
            return original_read_text(path, encoding=encoding or "gbk", errors=errors)

        with patch.object(Path, "read_text", legacy_read_text):
            loaded = runpy.run_path(str(core))
            self.assertEqual(loaded["CASES"], tuple(expected_cases))
            self.assertEqual(
                loaded["catalog"](),
                {entry["id"]: entry for entry in expected_catalog},
            )


if __name__ == "__main__":
    unittest.main()
