"""Output paths configured by the root command, never by numerical constants."""
from __future__ import annotations
import os
from pathlib import Path


def code_dir() -> Path:
    return Path(__file__).resolve().parent


def package_dir() -> Path:
    return code_dir().parent


def output_dir() -> Path:
    value = os.environ.get('SIMULATION_OUTPUT')
    if not value:
        raise RuntimeError('Run a numerical command through reproduce.py, not a code module directly.')
    result = Path(value).resolve()
    root = package_dir()
    if result == root or root in result.parents or result in root.parents:
        raise ValueError('The output directory must be separate from the code/data package.')
    return result


def work_dir() -> Path:
    result = output_dir() / 'work'
    if not result.is_dir():
        raise FileNotFoundError('Missing staged input directory. Run reproduce.py to prepare it.')
    return result
