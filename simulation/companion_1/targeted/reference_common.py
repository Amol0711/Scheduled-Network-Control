"""Read-only pinned-input access and explicit numerical-check records."""
from __future__ import annotations
import csv, hashlib, io, json, math, platform, stat, sys, zipfile
from pathlib import Path, PurePosixPath
import numpy as np

ARCHIVE_SHA256 = '682537d06ac2f6fdbef9ab65bc59210df5150bfca3455306babe426b006b9d33'
PREFIX = 'simulation_code_data/'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError('Cannot serialize an empty record table')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


class Checks:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def require(self, key: str, condition: bool, details: object = None) -> None:
        if any(x['id'] == key for x in self.records):
            raise ValueError('Duplicate check id: '+key)
        row = {'id': key, 'passed': bool(condition), 'details': details}
        self.records.append(row)
        if not condition:
            raise AssertionError(row)

    def close(self, key: str, actual: float, reference: float,
              atol: float = 0.0, rtol: float = 0.0) -> None:
        limit = atol + rtol*abs(reference)
        self.require(key, math.isfinite(actual) and math.isfinite(reference)
                     and abs(actual-reference) <= limit,
                     {'actual': actual, 'reference': reference, 'atol': atol,
                      'rtol': rtol, 'absolute_difference': abs(actual-reference)})

    def report(self) -> dict:
        return {'status': 'PASS', 'checks_passed': len(self.records),
                'checks': self.records}


class Archive:
    """Read-only view of pinned original code/data; never executes those files."""
    def __init__(self, path: Path):
        self.path = path.resolve(strict=True)
        if not self.path.is_dir():
            raise ValueError('Expected the consolidated package directory')
        spec = json.loads((self.path/'config/reference_inputs.json').read_text())
        self.manifest = spec['unchanged_source_data_sha256']
        self.original_archive_sha256 = spec['original_archive_sha256']
        self.assert_unchanged()

    def raw(self, name: str) -> bytes:
        p = PurePosixPath(name)
        if (p.is_absolute() or '..' in p.parts or '\\' in name
                or name not in self.manifest):
            raise ValueError('Unregistered input request: '+name)
        source = self.path / name
        if source.is_symlink() or any(a.is_symlink() for a in source.parents):
            raise ValueError('Linked input: '+name)
        data = source.read_bytes()
        if digest(data) != self.manifest[name]:
            raise ValueError('Input checksum mismatch: '+name)
        return data

    def text(self, name: str) -> str:
        return self.raw(name).decode('utf-8')

    def json(self, name: str) -> dict:
        return json.loads(self.text(name))

    def csv(self, name: str) -> list[dict]:
        return list(csv.DictReader(io.StringIO(self.text(name))))

    def array(self, name: str) -> np.ndarray:
        return np.genfromtxt(io.StringIO(self.text(name)), delimiter=',', names=True)

    def npz(self, name: str) -> dict[str, np.ndarray]:
        with np.load(io.BytesIO(self.raw(name)), allow_pickle=False) as nz:
            return {k: nz[k].copy() for k in nz.files}

    def assert_unchanged(self) -> None:
        for name in self.manifest:
            self.raw(name)

    def metadata(self) -> dict:
        return {'input_kind':'consolidated_directory',
                'original_archive_sha256': self.original_archive_sha256,
                'unchanged_scientific_inputs_verified': len(self.manifest),
                'archive_code_executed': False}


def environment() -> dict:
    import scipy, mpmath
    return {'python': sys.version, 'platform': platform.platform(),
            'numpy': np.__version__, 'scipy': scipy.__version__,
            'mpmath': mpmath.__version__}
