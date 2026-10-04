"""Finite-amplitude coupling, drift-warp, cross and total actuation energies."""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
from scipy.integrate import solve_ivp
from reference_common import Archive, Checks, write_csv, write_json
from ode_common import N,T,K0,OMEGA,EPS,L,X0,field


def integrate(rtol,atol,nominal_deltas):
    tstar=np.array([T-float(d) for d in nominal_deltas]);delta=T-tstar
    tau=np.log(T/delta);initial_phase=np.arctan2(X0[:,1].mean(),X0[:,0].mean())
    def rhs(clock,z):
        x=z[:2*N].reshape(N,2);f=field(x);lx=L@x;dx=f-K0*lx
        gain=math.exp(clock)/T;dt=1/gain
        uc=-K0*gain*lx;uw=(gain-1)*f;total=uc+uw
        y=x.mean(axis=0);dy=dx.mean(axis=0)
        phase_rate=(y[0]*dy[1]-y[1]*dy[0])/np.dot(y,y)
        return np.r_[dx.ravel(),phase_rate,dt*np.sum(uc*uc),dt*np.sum(uw*uw),
                     dt*2*np.sum(uc*uw),dt*np.sum(total*total)]
    z0=np.r_[X0.ravel(),initial_phase,0.,0.,0.,0.]
    sol=solve_ivp(rhs,(0,float(tau[-1])),z0,t_eval=tau,method='DOP853',rtol=rtol,atol=atol)
    if not sol.success:raise RuntimeError(sol.message)
    rows=[]
    for j in range(len(tau)):
        phase,jc,jw,cross,jtot=sol.y[2*N:,j];x=sol.y[:2*N,j].reshape(N,2)
        f=field(x);g=math.exp(tau[j])/T;uc=-K0*g*(L@x);uw=(g-1)*f
        cycle=N*OMEGA**2*(1/delta[j]-1/T-2*math.log(T/delta[j])+(T-delta[j]))
        rows.append({'nominal_T_minus_t':float(nominal_deltas[j]),'actual_T_minus_t':float(delta[j]),
                     't_star':float(tstar[j]),'tau_star':float(tau[j]),'rtol':rtol,'atol':atol,
                     'coupling_energy':float(jc),'drift_warp_energy':float(jw),
                     'cross_energy':float(cross),'total_energy_direct_integral':float(jtot),
                     'total_energy_from_components':float(jc+jw+cross),
                     'component_identity_absolute_error':float(abs(jtot-jc-jw-cross)),
                     'continuous_phase_excess':float(phase-initial_phase-OMEGA*tstar[j]),
                     'native_vector_field_residual_in_regular_clock':float(np.max(abs((f+uc+uw)/g-(f-K0*(L@x))))),
                     'delta_times_drift_warp_energy':float(delta[j]*jw),
                     'cycle_exact_warp_energy':float(cycle),
                     'finite_amplitude_minus_cycle_warp_energy':float(jw-cycle),
                     'mean_radius':float(np.linalg.norm(x.mean(axis=0))),'nfev':sol.nfev})
    return rows


def run(archive:Archive,out:Path)->dict:
    ck=Checks();ref=archive.json('data/records/numerical_summary.json')['method_metrics']['full_clock']
    settings=[(1e-10,1e-12),(1e-12,1e-13),(2.3e-14,1e-15)]
    # Each exact-cutoff run terminates at the historical binary64 snapshot.
    snapshot=[integrate(r,a,[EPS])[0] for r,a in settings]
    for i,row in enumerate(snapshot):
        ck.close(f'run{i}:coupling_reference',row['coupling_energy'],ref['energy'],atol=2e-9)
        ck.close(f'run{i}:phase_reference',row['continuous_phase_excess'],ref['phase_error'],atol=2e-8)
        ck.require(f'run{i}:total_component_identity',row['component_identity_absolute_error']<2e-8)
        ck.require(f'run{i}:native_dynamics_identity',row['native_vector_field_residual_in_regular_clock']<1e-12)
        ck.require(f'run{i}:cross_is_nonzero_negative',row['cross_energy']<-.10)
    for i,row in enumerate(snapshot[:-1]):
        for name in ('coupling_energy','drift_warp_energy','cross_energy','total_energy_direct_integral','continuous_phase_excess'):
            ck.close(f'run{i}:refinement:{name}',row[name],snapshot[-1][name],atol=3e-8,rtol=2e-9)
    series=integrate(1e-12,1e-13,[1e-3,1e-4,1e-5,1e-6])
    errors=[abs(r['delta_times_drift_warp_energy']-N*OMEGA**2) for r in series]
    ck.require('cutoff_series:increasing_warp_energy',all(a['drift_warp_energy']<b['drift_warp_energy'] for a,b in zip(series,series[1:])))
    ck.require('cutoff_series:scaled_coefficient_approaches_24',all(a>b for a,b in zip(errors,errors[1:])))
    ck.require('cutoff_series:last_coefficient_within_1e-3',errors[-1]<1e-3)
    for i,row in enumerate(series):
        ck.require(f'cutoff{i}:component_identity',row['component_identity_absolute_error']<1e-7)
    write_csv(out/'full_clock_snapshot.csv',snapshot);write_csv(out/'full_clock_cutoff_series.csv',series)
    report={'scope':'Independent deterministic re-integration, no overwrite of Table S1 or archived coupling energies.',
            'normalization':'kappa1(t)=1/(T-t); dt/dtau=T exp(-tau)',
            'parameters':{'N':N,'T':T,'k0':K0,'omega':OMEGA,'alpha':1,'bSL':1},
            'initial_state':X0.tolist(),'preferred_snapshot_run':snapshot[-1],
            'reference_coupling_value_unchanged':ref['energy'],
            'time_precision':'nominal delta=1e-5, with t_star=2.0-1e-5 and delta=2.0-t_star in binary64, matching the original code.',
            'cycle_exact_integral':'N omega^2 a_star^2 [1/delta-1/T-2 log(T/delta)+(T-delta)]',
            'asymptotic_cycle_coefficient':24,'error_certificate':False,
            'snapshot_refinement':snapshot,'cutoff_series':series,**ck.report()}
    write_json(out/'effort_reference.json',report)
    print('effort',snapshot[-1],flush=True)
    return {'checks':len(ck.records),'snapshot_runs':len(snapshot),'cutoff_samples':len(series)}
