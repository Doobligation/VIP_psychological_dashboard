import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import build_dashboard


class HistoricalRegenerationTests(unittest.TestCase):
    def test_regeneration_preserves_maintained_files_and_replaces_historical_json(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            paths = {}
            for name in ('OUTPUT_HTML', 'README', 'SERVE', 'OUTPUT_JSON'):
                paths[name] = Path(directory) / name
                paths[name].write_text('maintained content', encoding='utf-8')
                stack.enter_context(patch.object(build_dashboard, name, paths[name]))

            build_dashboard.write_files({'sessions': [], 'updated': True})

            self.assertEqual(json.loads(paths['OUTPUT_JSON'].read_text()), {'sessions': [], 'updated': True})
            for name in ('OUTPUT_HTML', 'README', 'SERVE'):
                self.assertEqual(paths[name].read_text(), 'maintained content')

    def test_missing_html_is_generated_without_creating_server_or_readme(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            paths = {}
            for name in ('OUTPUT_HTML', 'README', 'SERVE', 'OUTPUT_JSON'):
                paths[name] = Path(directory) / name
                stack.enter_context(patch.object(build_dashboard, name, paths[name]))

            build_dashboard.write_files({'sessions': []})

            self.assertTrue(paths['OUTPUT_HTML'].exists())
            self.assertFalse(paths['README'].exists())
            self.assertFalse(paths['SERVE'].exists())


if __name__ == '__main__':
    unittest.main()
