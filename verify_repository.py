#!/usr/bin/env python3
"""Read-only repository-payload verification using the Python standard library.

This checks bytes and file membership, not scientific correctness or live anonymity.
A root .git directory is permitted for a checkout but is not audited or distributed.
No scientific module is imported, and no files or caches are written.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        m = re.fullmatch(r'([0-9a-f]{64})  ([^\r\n]+)', line)
        if not m:
            raise ValueError('Malformed checksum record')
        checksum, name = m.groups()
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name or name in entries or name == path.name:
            raise ValueError('Unsafe, duplicate or circular manifest entry')
        entries[name] = checksum
    return entries


def inventory(root: Path, allow_git: bool) -> tuple[set[str], bool]:
    found: set[str] = set()
    git_present = False
    def visit(directory: Path) -> None:
        nonlocal git_present
        for p in directory.iterdir():
            relative = p.relative_to(root).as_posix()
            if p.is_symlink():
                raise ValueError('Symbolic link in payload: ' + relative)
            if relative == '.git' and allow_git:
                if not p.is_dir():
                    raise ValueError('Root .git must be a real directory, not a pointer file')
                git_present = True
                continue
            if p.is_file():
                if p.stat().st_nlink != 1:
                    raise ValueError('Hard-linked payload file: ' + relative)
                found.add(relative)
            elif p.is_dir():
                visit(p)
            else:
                raise ValueError('Nonregular payload entry: ' + relative)
    visit(root)
    return found, git_present


def verify(root: Path) -> dict:
    checks = []
    def check(name: str, passed: bool) -> None:
        checks.append({'check': name, 'pass': bool(passed)})
        if not passed:
            raise ValueError(name)
    try:
        root = root.resolve(strict=True)
        manifest = read_manifest(root / 'SHA256SUMS.txt')
        actual, git = inventory(root, True)
        check('Exact repository payload membership', actual == set(manifest) | {'SHA256SUMS.txt'})
        check('Repository payload hashes', all(digest(root / n) == h for n, h in manifest.items()))
        companion = root / 'simulation/companion_1'
        internal = read_manifest(companion / 'SHA256SUMS.txt')
        members, _ = inventory(companion, False)
        check('Original 129-file companion membership', len(members) == 129 and members == set(internal) | {'SHA256SUMS.txt'})
        check('Original companion checksum entries', len(internal) == 128 and all(digest(companion / n) == h for n, h in internal.items()))
        check('Original 32 Python files and one shell launcher', len(list(companion.rglob('*.py'))) == 32 and len(list(companion.rglob('*.sh'))) == 1)
        data = root / 'verified_current/journal/source/data'
        check('Twelve publication plotting CSVs only', len(list(data.iterdir())) == 12 and all(p.suffix == '.csv' for p in data.iterdir()))
        check('Original plotting exporters present', all((root / ('tools/export_' + name + '_plot_data.py')).is_file() for name in ['fg2', 'fg3']))
        for p in root.rglob('*.py'):
            if '.git' not in p.relative_to(root).parts:
                ast.parse(p.read_text(encoding='utf-8'), filename=p.relative_to(root).as_posix())
        check('Python syntax parsed without imports or compilation caches', True)
        check('No artifact ZIPs or journal PDF copies in the payload', not any(Path(n).suffix == '.zip' for n in actual) and {n for n in actual if n.endswith('.pdf')} == {'simulation/companion_1/docs/COMPUTATIONAL_COMPANION.pdf'})
        check('Byte-preserving Git attributes', '* -text' in (root / '.gitattributes').read_text())
        check('No new license declaration', json.loads((companion / 'RELEASE.json').read_text())['license_selected'] is False and not any(Path(n).name.upper().startswith(('LICENSE','COPYING')) for n in actual))
        return {'status': 'PASS', 'checks_passed': len(checks), 'checks_total': len(checks),
                'payload_files': len(actual), 'root_git_metadata_present_but_not_audited': git,
                'scientific_routes_executed': False, 'live_anonymous_link_tested': False,
                'checks': checks}
    except (OSError, ValueError, KeyError, SyntaxError) as error:
        return {'status': 'FAIL', 'checks_passed': sum(x['pass'] for x in checks),
                'checks_total': len(checks), 'error': str(error), 'checks': checks}


if __name__ == '__main__':
    if len(sys.argv) > 2:
        raise SystemExit('Usage: python3 -B verify_repository.py [repository_root]')
    result = verify(Path(sys.argv[1]) if len(sys.argv) == 2 else Path(__file__).resolve().parent)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
