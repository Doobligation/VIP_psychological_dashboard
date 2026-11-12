import csv
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'polar_recordings.py'
HAS_OPENPYXL = importlib.util.find_spec('openpyxl') is not None


class PolarRecordingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'input.csv'
        self.output = self.root / 'converted.csv'

    def write_csv(self, headers, rows):
        with self.source.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)

    def run_cli(self, kind='ecg', unit='ns', extra=(), *, no_site=False):
        environment = os.environ.copy()
        if no_site:
            # -S omits site-packages, but explicit PYTHONPATH also needs to be
            # removed when checking the standard-library-only installation.
            environment.pop('PYTHONPATH', None)
        return subprocess.run(
            [sys.executable, *(['-S'] if no_site else []), str(SCRIPT), str(self.source),
             '--kind', kind, '--timestamp-column', 'time', '--timestamp-unit', unit,
             '--output', str(self.output), *extra],
            capture_output=True, text=True, check=False, env=environment,
        )

    def read_output(self):
        with self.output.open(newline='', encoding='utf-8') as stream:
            return list(csv.DictReader(stream))

    def test_ecg_preserves_raw_columns_and_nanoseconds_without_dependencies(self):
        self.write_csv(['time', 'ecg_uv', 'note'], [['1700000000123456789', '-00042.50', 'café, raw']])
        result = self.run_cli(no_site=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.read_output(), [{
            'time': '1700000000123456789', 'ecg_uv': '-00042.50', 'note': 'café, raw',
            'timestamp_utc': '2023-11-14T22:13:20.123456789Z',
        }])

    def test_explicit_timestamp_units_and_negative_epoch(self):
        for unit, value, expected in [
            ('s', '1700000000.123456789', '2023-11-14T22:13:20.123456789Z'),
            ('ms', '1700000000123', '2023-11-14T22:13:20.123Z'),
            ('us', '1700000000123456', '2023-11-14T22:13:20.123456Z'),
            ('ns', '-1', '1969-12-31T23:59:59.999999999Z'),
            ('s', '-1e-100', '1969-12-31T23:59:59.999999999Z'),
            ('ns', '-1e-100', '1969-12-31T23:59:59.999999999Z'),
        ]:
            with self.subTest(unit=unit, value=value):
                self.write_csv(['time', 'ecg'], [[value, '0']])
                result = self.run_cli(unit=unit)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.read_output()[0]['timestamp_utc'], expected)

    def test_acc_units_mappings_and_signed_posture_angles(self):
        for unit, scale in [('g', 1), ('mg', 1000), ('m/s2', 9.80665)]:
            with self.subTest(unit=unit):
                self.write_csv(['time', 'X', 'Y', 'Z'], [
                    [0, 0, 0, scale],
                    [1, -.5 * scale, 0, math.sqrt(3) / 2 * scale],
                    [2, 0, .5 * scale, math.sqrt(3) / 2 * scale],
                ])
                result = self.run_cli(kind='acc', unit='s', extra=[
                    '--acc-unit', unit, '--ax-column', 'X', '--ay-column', 'Y', '--az-column', 'Z',
                ])
                self.assertEqual(result.returncode, 0, result.stderr)
                rows = self.read_output()
                self.assertEqual([r['posture_level'] for r in rows], ['Good', 'Poor', 'Critical'])
                self.assertAlmostEqual(float(rows[1]['ax_g']), -.5)
                self.assertAlmostEqual(float(rows[1]['pitch_deg']), -30)
                self.assertAlmostEqual(float(rows[2]['roll_deg']), 30)
                self.assertAlmostEqual(float(rows[2]['acceleration_magnitude_g']), 1)
                self.assertNotIn('yaw_deg', rows[0])

    def test_reference_angles_and_inclusive_threshold_boundaries(self):
        # With the sensor upright its raw pitch/roll are zero. Offsets make the
        # corrected angles exactly the boundary values without trig rounding.
        for pitch, roll, expected in [
            (20, 10, 'Good'), (20.001, 0, 'Poor'), (0, -10.001, 'Poor'),
            (45, 20, 'Poor'), (-45.001, 0, 'Critical'), (0, 20.001, 'Critical'),
        ]:
            with self.subTest(pitch=pitch, roll=roll):
                self.write_csv(['time', 'ax', 'ay', 'az'], [[0, 0, 0, 1]])
                result = self.run_cli(kind='acc', unit='s', extra=[
                    '--acc-unit', 'g', '--reference-pitch-deg', str(pitch),
                    '--reference-roll-deg', str(roll),
                ])
                self.assertEqual(result.returncode, 0, result.stderr)
                row = self.read_output()[0]
                self.assertEqual(float(row['pitch_deg']), -pitch)
                self.assertEqual(float(row['roll_deg']), -roll)
                self.assertEqual(row['posture_level'], expected)

    def test_invalid_rows_report_line_and_do_not_create_or_replace_output(self):
        for invalid in ['', 'NaN', 'Infinity', 'not-a-time', '1e100', '1e9999999']:
            for existing in [False, True]:
                with self.subTest(invalid=invalid, existing=existing):
                    self.write_csv(['time', 'ecg'], [[0, 1], [invalid, 2]])
                    if self.output.exists():
                        self.output.unlink()
                    if existing:
                        self.output.write_text('previous result', encoding='utf-8')
                    result = self.run_cli()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('error: line 3:', result.stderr)
                    self.assertNotIn('Traceback', result.stderr)
                    if existing:
                        self.assertEqual(self.output.read_text(), 'previous result')
                    else:
                        self.assertFalse(self.output.exists())
                    self.assertFalse(list(self.root.glob('*.tmp')))

    def test_acc_rejects_nonfinite_and_zero_vector_rows(self):
        for axes in [(0, 0, 0), ('NaN', 0, 1), (0, '', 1), ('1e309', 0, 1)]:
            with self.subTest(axes=axes):
                self.write_csv(['time', 'ax', 'ay', 'az'], [[0, *axes]])
                result = self.run_cli(kind='acc', extra=['--acc-unit', 'g'])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('line 2', result.stderr)
                self.assertFalse(self.output.exists())

    def test_refuses_input_overwrite_and_aliases(self):
        self.write_csv(['time', 'ecg'], [[0, 1]])
        original = self.source.read_bytes()
        for target in [self.source, self.root / 'alias.csv']:
            with self.subTest(target=target.name):
                if target != self.source:
                    target.symlink_to(self.source)
                self.output = target
                result = self.run_cli()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('input', result.stderr)
                self.assertEqual(self.source.read_bytes(), original)

    def test_rejects_ambiguous_headers_and_malformed_rows(self):
        for headers, rows in [
            (['wrong', 'ecg'], [[0, 1]]),
            (['time', 'time'], [[0, 1]]),
            (['time', 'timestamp_utc'], [[0, 'do not replace me']]),
            (['time', 'ecg'], [[0, 1, 'extra']]),
            (['time', 'ecg'], [[0]]),
        ]:
            with self.subTest(headers=headers, rows=rows):
                self.write_csv(headers, rows)
                result = self.run_cli()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('line', result.stderr)
                self.assertFalse(self.output.exists())

    def test_acc_requires_explicit_units(self):
        self.write_csv(['time', 'ax', 'ay', 'az'], [[0, 0, 0, 1]])
        result = self.run_cli(kind='acc')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--acc-unit', result.stderr)

    def test_acc_axes_must_map_to_distinct_columns(self):
        self.write_csv(['time', 'ax', 'ay', 'az'], [[0, 0, 0, 1]])
        result = self.run_cli(kind='acc', extra=['--acc-unit', 'g', '--ax-column', 'az'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('distinct', result.stderr)
        self.assertFalse(self.output.exists())

    def test_xlsx_missing_dependency_gives_install_command(self):
        self.source = self.root / 'input.xlsx'
        self.source.write_bytes(b'')
        result = self.run_cli(no_site=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('pip install', result.stderr)
        self.assertIn('openpyxl', result.stderr)
        self.assertFalse(self.output.exists())

    @unittest.skipUnless(HAS_OPENPYXL, 'optional openpyxl is not installed')
    def test_xlsx_selected_sheet_preserves_epoch_text_and_columns(self):
        import openpyxl
        self.source = self.root / 'input.xlsx'
        book = openpyxl.Workbook()
        book.active.title = 'Ignore'
        sheet = book.create_sheet('Recording')
        sheet.append(['time', 'ecg'])
        sheet.append(['1700000000123456789', -42])
        book.save(self.source)
        book.close()
        result = self.run_cli(extra=['--sheet', 'Recording'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.read_output(), [{
            'time': '1700000000123456789', 'ecg': '-42',
            'timestamp_utc': '2023-11-14T22:13:20.123456789Z',
        }])


if __name__ == '__main__':
    unittest.main()
