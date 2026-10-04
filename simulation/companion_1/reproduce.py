#!/usr/bin/env python3
"""Portable numerical reconstruction and explicitly selected simulation routes."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

# Avoid package-local bytecode, including when an unwritable archive is used.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
CHECK_MODULES = [
    'schedule_profiles', 'dirichlet_envelope', 'functional_hierarchy',
    'actuator_realization', 'benchmark_certificate', 'schedule_asymptotics',
    'modal_limits', 'realization_gaps', 'comparison_metrics',
]
VERSIONS = {'numpy':'2.3.5', 'scipy':'1.17.0', 'sympy':'1.14.0', 'mpmath':'1.3.0',
            'numba':'0.65.1', 'llvmlite':'0.47.0'}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify() -> dict:
    """Read-only strict verification, requiring only the Python standard library."""
    manifest = ROOT/'SHA256SUMS.txt'
    expected = {}
    for line in manifest.read_text(encoding='utf-8').splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  ([^\r\n]+)', line)
        if not match:
            raise ValueError('Malformed checksum manifest')
        checksum, name = match.groups(); p = Path(name)
        if p.is_absolute() or '..' in p.parts or name in expected or name == 'SHA256SUMS.txt':
            raise ValueError('Unsafe, repeated or circular checksum entry')
        expected[name] = checksum
    actual = set(); links = []
    for path in ROOT.rglob('*'):
        name = path.relative_to(ROOT).as_posix()
        if path.is_symlink() or (path.is_file() and path.stat().st_nlink > 1):
            links.append(name)
        if path.is_file() and name != 'SHA256SUMS.txt':
            actual.add(name)
    missing = sorted(set(expected)-actual); extra = sorted(actual-set(expected))
    mismatched = [n for n,h in expected.items() if n in actual and sha(ROOT/n) != h]
    result = dict(status='PASS' if not (missing or extra or mismatched or links) else 'FAIL',
                  manifest_files=len(expected), missing=missing, unexpected=extra,
                  mismatched=mismatched, symlinks=links)
    if result['status'] != 'PASS':
        raise ValueError('Package integrity failure: '+json.dumps(result))
    return result


def dependencies(mode: str) -> dict:
    required = ['numpy','scipy','mpmath']
    if mode in ['checks','full','deterministic']:
        required.append('sympy')
    if mode in ['replay','monte-carlo','full']:
        required += ['numba','llvmlite']
    versions = {}; errors = []
    for name in required:
        try:
            module = importlib.import_module(name)
            versions[name] = importlib.metadata.version(name)
        except (ImportError, OSError, RuntimeError) as error:
            errors.append(f'{name}: {error}')
    if errors:
        raise RuntimeError('Missing or incompatible numerical dependencies. Install the supplied '
                           'requirements in the chosen Python environment. '+ '; '.join(errors))
    return {'python':platform.python_version(), 'system':platform.system(),
            'machine':platform.machine(), 'packages':versions,
            'matches_tested_package_versions':all(VERSIONS[n]==v for n,v in versions.items()),
            'cross_platform_bitwise_identity_not_assumed':True}


def run_module(name: str, job: Path, records: list, env: dict, arguments=()) -> None:
    logs = job/'logs'; logs.mkdir(exist_ok=True)
    t0 = time.monotonic()
    command = [sys.executable,'-B',str(ROOT/'code'/f'{name}.py'),*map(str,arguments)]
    with (logs/f'{name}.txt').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=job, env=env, stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic()-t0
    records.append({'module':name,'returncode':result.returncode,
                    'elapsed_seconds':round(elapsed,3),'log':f'logs/{name}.txt'})
    if result.returncode:
        print((logs/f'{name}.txt').read_text(encoding='utf-8')[-6000:], file=sys.stderr)
        raise RuntimeError(f'{name} failed with exit status {result.returncode}; see its log.')
    print(f'{name}: PASS ({elapsed:.1f} s)', flush=True)


def statistics(job: Path, records: list, env: dict) -> dict:
    dest = job/'statistics'
    run_module('stochastic_reporting',job,records,env,['--output-dir',dest])
    from compare_outputs import compare_group
    return compare_group(dest,ROOT/'data/statistics',job/'statistics_comparison.json',
                         rtol=3e-10,atol=2e-16)


def deterministic(job: Path, records: list, env: dict) -> dict:
    run_module('figure_series',job,records,env)
    run_module('local_certificate',job,records,env,['--output-dir',job/'local'])
    # The local trajectory series is the solver output itself, with no fitted interpolation.
    shutil.copyfile(job/'local/local_trajectory.csv',job/'figure_series/local_certified_trajectory.csv')
    from compare_outputs import compare_group
    return compare_group(job/'figure_series', ROOT/'data/figure_series',
                         job/'figure_comparison.json', rtol=3e-9, atol=3e-12)


def reference(job: Path, records: list, env: dict) -> dict:
    """Fresh independent diagnostics; no stochastic simulator is invoked."""
    dest=job/'reference';logdir=job/'logs';logdir.mkdir(exist_ok=True)
    if dest.exists():
        raise ValueError('Reference results already exist; choose a new output root')
    command=[sys.executable,'-B',str(ROOT/'targeted/run_reference.py'),
             '--package',str(ROOT),'--output',str(dest)]
    start=time.monotonic()
    with (logdir/'reference.txt').open('w') as log:
        result=subprocess.run(command,cwd=job,env=env,stdout=log,stderr=subprocess.STDOUT)
    records.append({'module':'reference','returncode':result.returncode,
                    'elapsed_seconds':time.monotonic()-start,'log':'logs/reference.txt'})
    if result.returncode:
        raise RuntimeError('Targeted reference calculation failed; see reference log')
    report=json.loads((dest/'RUN_STATUS.json').read_text())
    import numpy as np
    from compare_outputs import compare_csv
    comparisons=[]
    for expected in sorted((ROOT/'data/targeted_reference').rglob('*.csv')):
        produced=dest/expected.relative_to(ROOT/'data/targeted_reference')
        comparisons.append(compare_csv(produced,expected,rtol=3e-9,atol=3e-12))
    # JSON contains diagnostics and >double-precision decimal strings; never round it for comparison.
    exact=[{'file':str(f.relative_to(ROOT/'data/targeted_reference')),
            'byte_identical':(dest/f.relative_to(ROOT/'data/targeted_reference')).read_bytes()==f.read_bytes()}
           for f in sorted((ROOT/'data/targeted_reference').rglob('*')) if f.is_file()]
    comparison={'status':'PASS' if all(x['status']=='PASS' for x in comparisons) else 'FAIL',
                'csv_comparisons':comparisons,'all_reference_files':exact,
                'checks_passed':report['checks_passed'],
                'json_equality_policy':'Byte comparison is reported, not replaced by rounded numeric comparisons.'}
    (job/'targeted_reference_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    if comparison['status']!='PASS':raise RuntimeError('Targeted reference mismatch')
    print('reference: PASS',report['checks_passed'],'checks',flush=True)
    return comparison


def checkpoint_guard(job: Path, environment: dict, reset: bool) -> None:
    file = job/'checkpoint_guard.json'
    guard = {'package_manifest_sha256':sha(ROOT/'SHA256SUMS.txt'),
             'environment':environment}
    if file.exists() and not reset:
        if json.loads(file.read_text()) != guard:
            raise ValueError('Checkpoint code/data/environment differs. Use a separate output '
                             'directory, or explicitly --reset after retaining needed results.')
    elif (job/'checkpoint/checkpoint.json').exists() and not reset:
        raise ValueError('Checkpoint has no matching design guard; do not resume an unverified state.')
    file.write_text(json.dumps(guard,indent=2)+'\n')


def compare_paths(job: Path) -> dict:
    import numpy as np
    name = 'stochastic_crn_pathwise_terminal_mse.npz'
    comparisons = []
    with np.load(job/'work'/name,allow_pickle=False) as a, np.load(ROOT/'data/records'/name,allow_pickle=False) as b:
        for key in b.files:
            x,y=a[key],b[key]
            shape = x.shape==y.shape and x.dtype==y.dtype
            # Absolute tolerance is well below the precision of the reported means.
            equal = shape and np.array_equal(x,y)
            close = shape and np.allclose(x,y,rtol=3e-11,atol=3e-15)
            comparisons.append({'array':key,'shape':list(y.shape),'dtype':str(y.dtype),
                                'exact':bool(equal),'within_tolerance':bool(close),
                                'maximum_absolute_error':float(np.max(np.abs(x-y))) if shape else None})
    result = {'status':'PASS' if all(x['within_tolerance'] for x in comparisons) else 'FAIL',
              'relative_tolerance':3e-11,'absolute_tolerance':3e-15,'arrays':comparisons,
              'scope':'Same-stream regeneration, not an independent replication.'}
    (job/'pathwise_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status'] != 'PASS':
        raise RuntimeError('Regenerated terminal arrays differ from the supplied same-stream reference.')
    return result


def compare_full_records(job: Path) -> dict:
    from compare_outputs import compare_csv
    records=[]
    for ref in sorted((ROOT/'data/records').glob('*.csv')):
        produced=(job/'local' if ref.name.startswith('local_') else job/'work')/ref.name
        records.append(compare_csv(produced,ref,rtol=3e-9,atol=3e-12))
    result={'status':'PASS' if all(r['status']=='PASS' for r in records) else 'FAIL',
            'files':len(records),'numeric_cells':sum(r.get('numeric_cells',0) for r in records),
            'byte_identical_files':sum(r.get('byte_identical',False) for r in records),
            'sealed_records_staged':False,'comparisons':records}
    (job/'record_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if result['status']!='PASS':raise RuntimeError('Full-route numerical-record comparison failed.')
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',nargs='?',default='quick',
                   choices=['verify','quick','statistics','tables','deterministic','figures',
                            'checks','reference','replay','monte-carlo','full'])
    p.add_argument('--output-dir',type=Path,default=ROOT.parent/'simulation_results',
                   help='External output root; each route has its own subdirectory.')
    p.add_argument('--threads',type=int,default=min(4,os.cpu_count() or 1),help='Numba worker threads, default up to 4.')
    p.add_argument('--chunk-paths',type=int,default=20000,
                   help='For monte-carlo only; positive multiple of 200, preserving the fixed stream.')
    p.add_argument('--confirm-full',action='store_true',help='Required to start the 20000-path generator.')
    p.add_argument('--reset',action='store_true',help='Explicitly restart a full/monte-carlo checkpoint.')
    a=p.parse_args(); mode={'tables':'statistics','figures':'deterministic'}.get(a.mode,a.mode)
    if a.threads<1 or a.threads> (os.cpu_count() or 1):
        p.error('--threads must lie between 1 and the available logical CPU count.')
    if a.chunk_paths<1 or a.chunk_paths%200:
        p.error('--chunk-paths must be a positive multiple of 200.')
    if mode=='full' and a.chunk_paths!=20000:
        p.error('Use monte-carlo for partial chunks. full uses all 20000 primary paths.')
    if mode in ['full','monte-carlo'] and not a.confirm_full:
        p.error('This route generates stochastic trajectories. Add --confirm-full explicitly.')
    if a.reset and mode not in ['full','monte-carlo']:
        p.error('--reset applies only to full or monte-carlo.')
    verification=verify()
    if mode=='verify':
        print(json.dumps(verification,indent=2)); return 0
    # Set deterministic thread controls before importing numerical libraries.
    os.environ['NUMBA_NUM_THREADS']=str(a.threads)
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('OMP_NUM_THREADS','1')
    environment=dependencies(mode)
    base=a.output_dir.expanduser().resolve(); job=(base/mode).resolve()
    if job==ROOT or ROOT in job.parents or job in ROOT.parents or base==ROOT or base in ROOT.parents:
        raise ValueError('Output must be separate from the code/data package, not its ancestor.')
    job.mkdir(parents=True,exist_ok=True)
    for existing in job.rglob('*'):
        if existing.is_symlink() or (existing.is_file() and existing.stat().st_nlink>1):
            raise ValueError('Output trees with symbolic or hard links are rejected to protect external files.')
    lock=job/'run.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:
        raise RuntimeError('This output route is locked. Confirm that no process is active before removing run.lock.')
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid())+'\n')
    records=[];summary={'mode':mode,'status':'RUNNING','input_verification':verification,
                        'environment':environment,'modules':records,
                        'primary_cap_experiment_complete':False,'all_full_route_stages_complete':False,'independent_new_replication':False,
                        'sealed_input_directory_modified':False}
    start=time.monotonic();exitcode=1
    try:
        work=job/'work';work.mkdir(exist_ok=True)
        if mode!='full':
            for f in (ROOT/'data/records').iterdir():
                if f.is_file():shutil.copyfile(f,work/f.name)
        summary['sealed_records_staged']=mode!='full'
        env=os.environ.copy()
        env.update(SIMULATION_OUTPUT=str(job),PYTHONDONTWRITEBYTECODE='1',
                   NUMBA_CACHE_DIR=str(job/'cache/numba'),NUMBA_NUM_THREADS=str(a.threads))
        sys.path.insert(0,str(ROOT/'code'))
        run_module('validate_configuration',job,records,env)
        run_module('validate_release',job,records,env)
        if mode in ['quick','statistics']:
            summary['statistics']=statistics(job,records,env)
            if mode=='quick':run_module('certificate_reconstruction',job,records,env)
            summary['trajectories_simulated']=0
        elif mode=='reference':
            summary['reference']=reference(job,records,env)
            summary['stochastic_paths_simulated']=0
        elif mode=='deterministic':
            summary['figure_series']=deterministic(job,records,env)
            summary['stochastic_paths_simulated']=0
        elif mode=='checks':
            for name in CHECK_MODULES:run_module(name,job,records,env)
            summary['stochastic_paths_simulated']=0
        elif mode=='replay':
            run_module('seeded_replay',job,records,env,['--output-dir',job/'replay'])
            summary['primary_paths_replayed']=200
            summary['nested_coarse_paths_replayed']=200
        elif mode in ['monte-carlo','full']:
            checkpoint_guard(job,environment,a.reset)
            if mode=='full':
                run_module('network_experiments',job,records,env)
                run_module('schedule_noise',job,records,env)
                for name in CHECK_MODULES:run_module(name,job,records,env)
            args=['--chunk-paths',str(a.chunk_paths)] + (['--reset'] if a.reset else [])
            run_module('stochastic_network',job,records,env,args)
            cursor=int(json.loads((job/'checkpoint/checkpoint.json').read_text())['cursor'])
            summary['completed_primary_paths']=cursor
            if cursor==20000:
                summary['pathwise_comparison']=compare_paths(job)
                summary['statistics']=statistics(job,records,env)
                if mode=='full':
                    summary['figure_series']=deterministic(job,records,env)
                    summary['numerical_records']=compare_full_records(job)
                    summary['reference']=reference(job,records,env)
                summary['primary_cap_experiment_complete']=True
                summary['all_full_route_stages_complete']=mode=='full'
            else:
                summary['status']='PARTIAL'
                summary['note']='Checkpoint only; no completed-experiment result is claimed. Repeat the same command to resume.'
        # Reverify every delivered input after execution, not just selected arrays.
        summary['input_verification_after']=verify()
        if summary['status']=='RUNNING':summary['status']='PASS'
        exitcode=0
    except (Exception,SystemExit) as error:
        summary['status']='FAIL';summary['error']=str(error)
        print(f'ERROR: {error}',file=sys.stderr)
    finally:
        summary['elapsed_seconds']=round(time.monotonic()-start,3)
        (job/'run_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        with contextlib.suppress(FileNotFoundError):lock.unlink()
    print(f"{summary['status']}: {mode}. Results: {job}")
    return exitcode


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (OSError,ValueError,RuntimeError) as error:
        print(f'ERROR: {error}',file=sys.stderr)
        raise SystemExit(1)
