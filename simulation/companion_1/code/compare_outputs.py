"""Numerical reference comparisons with explicit, serialized tolerances."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path


def compare_csv(produced: Path, reference: Path, rtol: float, atol: float) -> dict:
    """Compare headers/text exactly and finite numeric cells within a fixed bound."""
    result = {'file': reference.name, 'relative_tolerance': rtol, 'absolute_tolerance': atol}
    if not produced.is_file():
        return dict(result, status='FAIL', error='Missing regenerated output')
    a = list(csv.reader(produced.open(newline='', encoding='utf-8')))
    b = list(csv.reader(reference.open(newline='', encoding='utf-8')))
    shape_ok = len(a) == len(b) and all(len(x) == len(y) for x, y in zip(a, b))
    failures = []; n = 0; max_abs = 0.; max_ratio = 0.
    if not shape_ok:
        failures.append({'kind': 'shape', 'rows_produced': len(a), 'rows_reference': len(b)})
    else:
        for i, (row, refrow) in enumerate(zip(a, b)):
            for j, (x, y) in enumerate(zip(row, refrow)):
                try:
                    u, v = float(x), float(y)
                except ValueError:
                    if x != y:
                        failures.append({'row': i, 'column': j, 'kind': 'text', 'value': x, 'expected': y})
                    continue
                n += 1
                if math.isnan(u) or math.isnan(v):
                    ok = math.isnan(u) and math.isnan(v)
                elif math.isinf(u) or math.isinf(v):
                    ok = u == v
                else:
                    error = abs(u-v); bound = atol+rtol*abs(v)
                    max_abs = max(max_abs, error)
                    max_ratio = max(max_ratio, error/bound if bound else float(error != 0))
                    ok = error <= bound
                if not ok:
                    failures.append({'row': i, 'column': j, 'kind': 'numeric', 'value': x, 'expected': y})
    return dict(result, status='PASS' if not failures else 'FAIL', numeric_cells=n,
                byte_identical=produced.read_bytes() == reference.read_bytes(),
                maximum_absolute_error=max_abs, maximum_tolerance_fraction=max_ratio,
                failure_count=len(failures), failures=failures[:20])


def compare_group(produced: Path, reference: Path, output: Path, rtol=3e-10, atol=3e-14,
                  exclude=()) -> dict:
    records = [compare_csv(produced/f.name, f, rtol, atol)
               for f in sorted(reference.glob('*.csv')) if f.name not in exclude]
    result = {'status': 'PASS' if records and all(r['status']=='PASS' for r in records) else 'FAIL',
              'files': len(records), 'numeric_cells': sum(r.get('numeric_cells',0) for r in records),
              'byte_identical_files': sum(r.get('byte_identical',False) for r in records),
              'comparisons': records}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if result['status'] != 'PASS':
        raise RuntimeError(f'Numerical reference comparison failed. See {output.name}.')
    return result
