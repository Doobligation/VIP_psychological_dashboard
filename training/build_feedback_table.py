#!/usr/bin/env python3
"""Import a normalized feedback CSV/XLSX table; preserve existing decision defaults.

Columns: state,level,cause,message,action,goal,reference,enabled. One row per
state/level/cause. A level is enabled when its rows say `true` (XLSX booleans are also
supported); disabled levels use one row with cause `any` and empty text. Low and
medium levels must stay disabled. Every existing state/level must appear. XLSX uses
its first worksheet and rejects formulas. Timing (consecutive High predictions,
cooldown, warning cap) is kept from the base config's `decision` block.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from training.train_model import atomic_json
from pipeline.config import NO_FEEDBACK_LEVELS, load_config

COLUMNS = ['state', 'level', 'cause', 'message', 'action', 'goal', 'reference', 'enabled']
DEFAULT_CONFIG = ROOT / 'config/feedback_table.json'


def _read_rows(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        with path.open(newline='', encoding='utf-8-sig') as stream:
            return list(csv.reader(stream, strict=True))
    if path.suffix.lower() == '.xlsx':
        try:
            import openpyxl
        except ImportError as exc:
            raise ValueError('XLSX import requires openpyxl; install requirements.txt.') from exc
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=False)
        try:
            rows = []
            for row in workbook.worksheets[0].iter_rows():
                if any(cell.data_type == 'f' for cell in row):
                    raise ValueError('Formula cells are not allowed; supply literal normalized values.')
                rows.append([cell.value for cell in row])
            return rows
        finally:
            workbook.close()
    raise ValueError('Feedback input must be a normalized .csv or .xlsx file.')


def _text(value, column, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f'{column} must be a nonempty text cell.')
    if len(value) > 4000:
        raise ValueError(f'{column} exceeds 4000 characters.')
    return value.strip()


def _enabled(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ('true', 'false'):
        return value.strip().lower() == 'true'
    raise ValueError('enabled must be true or false.')


def build_feedback_table(source, output, *, base=DEFAULT_CONFIG, version=None):
    """Validate the complete sheet before atomically replacing output."""
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('Output must not overwrite the source spreadsheet.')
    original = json.loads(Path(base).read_text(encoding='utf-8'))
    rows = _read_rows(source)
    if not rows:
        raise ValueError('Missing normalized column header.')
    header = [value.strip() if isinstance(value, str) else value for value in rows[0]]
    if len(header) != len(COLUMNS) or set(header) != set(COLUMNS):
        raise ValueError('Columns must be exactly: ' + ','.join(COLUMNS))
    result = copy.deepcopy(original)
    seen, enabled_by_level = set(), {}
    for state, levels in result['states'].items():
        for row in levels.values():
            row['causes'] = {}
    for index, cells in enumerate(rows[1:], start=2):
        if not cells or all(value in (None, '') for value in cells):
            continue
        if len(cells) != len(header):
            raise ValueError(f'Row {index}: expected exactly {len(header)} cells.')
        row = dict(zip(header, cells))
        try:
            state, level = _text(row['state'], 'state'), _text(row['level'], 'level')
            cause = _text(row['cause'], 'cause')
            if state not in original['states'] or level not in original['states'][state]:
                raise ValueError('Unknown configured state/level.')
            if (state, level, cause) in seen:
                raise ValueError('Duplicate state/level/cause row.')
            seen.add((state, level, cause))
            enabled = _enabled(row['enabled'])
            if enabled_by_level.setdefault((state, level), enabled) != enabled:
                raise ValueError('All rows of one level must share the same enabled value.')
            if enabled and level in NO_FEEDBACK_LEVELS:
                raise ValueError('Low and medium levels must remain disabled.')
            result['states'][state][level]['enabled'] = enabled
            if enabled:
                result['states'][state][level]['causes'][cause] = {
                    key: _text(row[key], key) for key in ('message', 'action', 'goal')}
                result['states'][state][level]['causes'][cause]['reference'] = _text(
                    '' if row['reference'] is None else row['reference'], 'reference', empty=True)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Row {index}: {exc}') from exc
    expected = {(state, level) for state, levels in original['states'].items() for level in levels}
    if set(enabled_by_level) != expected:
        missing = ', '.join(f'{state}/{level}' for state, level in sorted(expected - set(enabled_by_level)))
        raise ValueError('Sheet must contain every configured state/level; missing: ' + missing)
    load_config(result, 'feedback_table.json')
    if version is not None:
        result['version'] = _text(version, 'version')
    atomic_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base', type=Path, default=DEFAULT_CONFIG,
                        help='Feedback config supplying unchanged decision defaults and required rows.')
    parser.add_argument('--version', help='Explicit feedback version for this import.')
    args = parser.parse_args(argv)
    try:
        result = build_feedback_table(args.source, args.output, base=args.base, version=args.version)
    except (ValueError, OSError, TypeError, KeyError, csv.Error) as exc:
        parser.error(str(exc))
    print(f"Saved {args.output}; version={result['version']}; decision defaults preserved.")


if __name__ == '__main__':
    main()
