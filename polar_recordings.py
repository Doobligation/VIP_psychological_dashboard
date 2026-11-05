"""Convert Polar CSV/XLSX recordings to UTC timestamps and ACC pitch/roll.

CSV needs only Python's standard library. XLSX input optionally uses openpyxl.
ECG samples remain raw; this tool does not derive heart rate or HRV. ACC angles
are gravity-based tilt estimates: movement acceleration affects them, and a
single accelerometer does not establish yaw, arm elevation, or fatigue.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal, DecimalException, ROUND_FLOOR, localcontext
import math
import os
from pathlib import Path
import tempfile


TIME_DIVISORS = {'s': 1, 'ms': 1_000, 'us': 1_000_000, 'ns': 1_000_000_000}
ACC_DIVISORS = {'g': 1, 'mg': 1000, 'm/s2': 9.80665}
ACC_COLUMNS = [
    'ax_g', 'ay_g', 'az_g', 'acceleration_magnitude_g',
    'pitch_deg', 'roll_deg', 'posture_level',
]


class RecordingError(ValueError):
    """An invalid recording or conversion configuration."""


def unix_to_utc(value, unit: str) -> str:
    """Keep up to nanosecond precision without first converting to float."""
    try:
        with localcontext() as context:
            context.prec = max(50, len(str(value)) + 12)
            seconds = Decimal(str(value)) / TIME_DIVISORS[unit]
            if not seconds.is_finite() or not -62135596800 <= seconds < 253402300800:
                raise RecordingError('timestamp must be finite and within UTC years 0001–9999')
            # Convert to integer nanoseconds before splitting the second. Decimal
            # subtraction can otherwise round a fraction just below 1 up to 1.
            total_nanoseconds = int((seconds * 1_000_000_000).to_integral_value(rounding=ROUND_FLOOR))
            whole_seconds, nanoseconds = divmod(total_nanoseconds, 1_000_000_000)
    except (DecimalException, ValueError, OverflowError) as error:
        raise RecordingError(f'invalid Unix timestamp {value!r}: {error}') from error
    moment = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=whole_seconds)
    fraction = f'.{nanoseconds:09d}'.rstrip('0') if nanoseconds else ''
    return moment.isoformat(timespec='seconds')[:-6] + fraction + 'Z'


def finite_number(value, name: str) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise RecordingError(f'{name} must be a finite number, received {value!r}') from error
    if not math.isfinite(number):
        raise RecordingError(f'{name} must be a finite number, received {value!r}')
    return number


def acceleration_features(values, unit: str, reference_pitch: float, reference_roll: float):
    ax, ay, az = [finite_number(value, name) / ACC_DIVISORS[unit]
                  for name, value in zip(('Ax', 'Ay', 'Az'), values)]
    magnitude = math.hypot(ax, ay, az)
    if magnitude == 0 or not math.isfinite(magnitude):
        raise RecordingError('ACC vector must have a finite, nonzero magnitude to estimate tilt')
    # Source: the supplied ML decision-rule PDF's final Polar H10 framework.
    pitch = math.degrees(math.atan2(ax, math.hypot(ay, az))) - reference_pitch
    roll = math.degrees(math.atan2(ay, math.hypot(ax, az))) - reference_roll
    if abs(pitch) > 45 or abs(roll) > 20:
        posture = 'Critical'
    elif abs(pitch) > 20 or abs(roll) > 10:
        posture = 'Poor'
    else:
        posture = 'Good'
    return [ax, ay, az, magnitude, pitch, roll, posture]


def csv_rows(reader):
    while True:
        line = reader.line_num + 1
        try:
            row = next(reader)
        except StopIteration:
            return
        except csv.Error as error:
            raise RecordingError(f'line {line}: malformed CSV: {error}') from error
        if row:
            yield line, row


@contextmanager
def recording_rows(path: Path, sheet: str | None):
    if path.suffix.lower() == '.csv':
        if sheet is not None:
            raise RecordingError('--sheet applies only to XLSX input')
        with path.open(newline='', encoding='utf-8-sig') as stream:
            yield csv_rows(csv.reader(stream, strict=True))
    elif path.suffix.lower() == '.xlsx':
        try:
            import openpyxl
        except ImportError as error:
            raise RecordingError('XLSX input requires openpyxl. Install it with: '
                                 'python3 -m pip install openpyxl') from error
        try:
            book = openpyxl.load_workbook(path, read_only=True, data_only=False)
        except Exception as error:
            raise RecordingError(f'cannot read XLSX workbook: {error}') from error
        try:
            if sheet is not None and sheet not in book.sheetnames:
                raise RecordingError(f'worksheet {sheet!r} not found; available: {", ".join(book.sheetnames)}')
            selected = book[sheet] if sheet is not None else book.worksheets[0]
            yield ((line, list(row)) for line, row in enumerate(selected.iter_rows(values_only=True), 1)
                   if any(value is not None for value in row))
        finally:
            book.close()
    else:
        raise RecordingError('input must be a .csv or .xlsx file')


def convert_recording(input_path, output_path, *, kind: str, timestamp_column: str,
                      timestamp_unit: str, sheet=None, ax_column='ax', ay_column='ay',
                      az_column='az', acc_unit=None, reference_pitch_deg=0,
                      reference_roll_deg=0) -> int:
    """Write a complete CSV atomically; leave previous output intact on error."""
    source, output = Path(input_path), Path(output_path)
    if source.resolve() == output.resolve() or (output.exists() and source.samefile(output)):
        raise RecordingError('output must differ from the input file')
    if output.suffix.lower() != '.csv':
        raise RecordingError('--output must name a .csv file')
    if kind not in ('acc', 'ecg') or timestamp_unit not in TIME_DIVISORS:
        raise RecordingError('select a valid kind and explicit Unix timestamp unit')
    if kind == 'acc' and acc_unit not in ACC_DIVISORS:
        raise RecordingError('--acc-unit is required for ACC: choose g, mg, or m/s2')
    pitch_offset = finite_number(reference_pitch_deg, 'reference pitch')
    roll_offset = finite_number(reference_roll_deg, 'reference roll')
    derived_columns = ['timestamp_utc'] + (ACC_COLUMNS if kind == 'acc' else [])
    temporary = None
    count = 0
    try:
        with recording_rows(source, sheet) as rows:
            try:
                header_line, headers = next(rows)
            except StopIteration as error:
                raise RecordingError('line 1: recording is empty') from error
            headers = ['' if value is None else str(value) for value in headers]
            if any(not header.strip() for header in headers) or len(set(headers)) != len(headers):
                raise RecordingError(f'line {header_line}: column names must be nonempty and unique')
            required = [timestamp_column] + ([ax_column, ay_column, az_column] if kind == 'acc' else [])
            if len(set(required)) != len(required):
                raise RecordingError(f'line {header_line}: timestamp and axes must map to distinct columns')
            missing = [name for name in required if name not in headers]
            if missing:
                raise RecordingError(f'line {header_line}: missing required column(s): {", ".join(missing)}')
            collisions = set(headers) & set(derived_columns)
            if collisions:
                raise RecordingError(f'line {header_line}: derived column(s) already exist: '
                                     f'{", ".join(sorted(collisions))}')
            indexes = [headers.index(name) for name in required]
            with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8',
                                             dir=output.parent, prefix=f'.{output.name}.',
                                             suffix='.tmp', delete=False) as destination:
                temporary = Path(destination.name)
                writer = csv.writer(destination)
                writer.writerow(headers + derived_columns)
                for line, values in rows:
                    if len(values) != len(headers):
                        raise RecordingError(f'line {line}: expected {len(headers)} columns, got {len(values)}')
                    try:
                        derived = [unix_to_utc(values[indexes[0]], timestamp_unit)]
                        if kind == 'acc':
                            derived.extend(acceleration_features(
                                [values[index] for index in indexes[1:]], acc_unit,
                                pitch_offset, roll_offset,
                            ))
                    except RecordingError as error:
                        raise RecordingError(f'line {line}: {error}') from error
                    writer.writerow(values + derived)
                    count += 1
                if count == 0:
                    raise RecordingError(f'line {header_line + 1}: no data rows')
            os.replace(temporary, output)
            temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Convert Polar CSV/XLSX Unix timestamps to UTC; optionally derive ACC pitch/roll.',
        epilog=('Output is CSV and preserves original columns. Epoch units are never guessed. '
                'UTC retains up to 9 fractional digits (sub-nanoseconds truncated); '
                'store long epoch values as text in Excel to avoid Excel rounding them. '
                'ACC uses 1 g = 9.80665 m/s2; offsets are subtracted from the computed angles. '
                'Good: |pitch| <= 20 and |roll| <= 10 degrees; Critical: |pitch| > 45 or '
                '|roll| > 20; otherwise Poor. These are experimental tilt rules, affected '
                'by sensor mounting and movement, not validated fatigue predictions. '
                'ECG samples are unchanged; no HR, HRV, yaw, or arm angle is derived. '
                'XLSX needs: python3 -m pip install openpyxl.'),
    )
    parser.add_argument('input', type=Path, help='input CSV or XLSX recording with a header row')
    parser.add_argument('--kind', choices=('acc', 'ecg'), required=True, help='recording type')
    parser.add_argument('--timestamp-column', required=True, help='exact column name containing Unix time')
    parser.add_argument('--timestamp-unit', choices=tuple(TIME_DIVISORS), required=True,
                        help='explicit Unix time unit: seconds, milliseconds, microseconds, or nanoseconds')
    parser.add_argument('--sheet', help='XLSX worksheet name; defaults to the first worksheet')
    parser.add_argument('--ax-column', default='ax', help='ACC X-axis column name (default: ax)')
    parser.add_argument('--ay-column', default='ay', help='ACC Y-axis column name (default: ay)')
    parser.add_argument('--az-column', default='az', help='ACC Z-axis column name (default: az)')
    parser.add_argument('--acc-unit', choices=tuple(ACC_DIVISORS), help='required for ACC; raw axis units')
    parser.add_argument('--reference-pitch-deg', type=float, default=0, help='neutral pitch to subtract, in degrees')
    parser.add_argument('--reference-roll-deg', type=float, default=0, help='neutral roll to subtract, in degrees')
    parser.add_argument('--output', type=Path, required=True, help='output CSV path; must differ from input')
    args = parser.parse_args(argv)
    options = vars(args).copy()
    source, output = options.pop('input'), options.pop('output')
    try:
        count = convert_recording(source, output, **options)
    except (RecordingError, OSError, UnicodeError) as error:
        parser.exit(2, f'error: {error}\n')
    print(f'Converted {count} rows to {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
