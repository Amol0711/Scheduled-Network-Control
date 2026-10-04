#!/usr/bin/env python3
"""Replay the first 200 recorded paths, not the 20,000-path experiment.

This small provenance audit checks the actual seeded Brownian construction at
both recorded steps. It does not supply independent validation data.
"""
from pathlib import Path
import argparse
import json
import os
import numpy as np
from numba import set_num_threads, get_num_threads
import stochastic_network as legacy

from artifact_paths import work_dir, output_dir, code_dir
ROOT=work_dir()

def main(output:Path)->int:
    output.mkdir(parents=True,exist_ok=True)
    set_num_threads(min(4,get_num_threads()))
    with np.load(ROOT/'stochastic_crn_pathwise_terminal_mse.npz',allow_pickle=False) as z:
        caps=z['caps'];refcaps=z['refinement_caps'];recorded=z['terminal_squared_disagreement'][:legacy.BATCH_SIZE].copy()
        recorded_coarse=z['refinement_dt_1e3'][:legacy.BATCH_SIZE].copy()
    steps=int(round(legacy.T/legacy.PRIMARY_DT))
    dw=np.random.default_rng(legacy.SEED).standard_normal((steps,legacy.BATCH_SIZE,legacy.N))*np.sqrt(legacy.PRIMARY_DT)
    primary=legacy._simulate_terminal_sq_batch(legacy.gain_matrix(caps,legacy.PRIMARY_DT),dw,legacy.initial_network_state(),legacy.PRIMARY_DT,legacy.SIGMA)
    coarse_dw=dw.reshape(steps//2,2,legacy.BATCH_SIZE,legacy.N).sum(axis=1)
    coarse=legacy._simulate_terminal_sq_batch(legacy.gain_matrix(refcaps,legacy.COARSE_DT),coarse_dw,legacy.initial_network_state(),legacy.COARSE_DT,legacy.SIGMA)
    records=[]
    for name,new,old in [('primary',primary,recorded),('coarse',coarse,recorded_coarse)]:
        delta=np.abs(new-old)
        records.append(dict(array=name,shape=list(old.shape),bitwise_identical=bool(np.array_equal(new,old)),
            max_abs_difference=float(delta.max()),max_relative_difference=float((delta/np.maximum(abs(old),1e-300)).max()),
            tolerance_passed=bool(np.allclose(new,old,rtol=1e-11,atol=1e-16))))
    passed=all(r['tolerance_passed'] for r in records)
    result=dict(status='PASS' if passed else 'FAIL',paths_replayed=legacy.BATCH_SIZE,seed=legacy.SEED,records=records,
        full_nonlinear_monte_carlo_rerun=False,new_independent_data=False,
        caveat='Replay checks provenance only. Platform-dependent floating-point equality is recorded separately from tolerance agreement.')
    (output/'seeded_replay_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return 0 if passed else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=ROOT/'reporting_statistics')
    raise SystemExit(main(p.parse_args().output_dir))
