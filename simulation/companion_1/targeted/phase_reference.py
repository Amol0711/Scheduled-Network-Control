"""Transported terminal-endpoint and solver-refinement diagnostics.
No reduced-flow round-trip error is reinterpreted as terminal-state accuracy.
The archived plot is preserved; raw and transported endpoint curves are separate
successor diagnostics. Full-state and cancellation-free normal-coordinate
formulations provide a second deterministic comparison.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from reference_common import Archive, Checks, write_csv, write_json
from ode_common import N,T,K0,OMEGA,L,X0,J,field,flow


def integrate(formulation,rtol,atol,maxstep):
    s=np.geomspace(.2,5e-4,240);tau=-np.log(s)
    ends=np.array([25.,30.,35.]);times=np.unique(np.r_[0.,tau,ends]);rho=1.2
    if formulation=='full_state':
        z0=X0.ravel()
        def rhs(clock,z):
            x=z.reshape(N,2)
            return (T*np.exp(-clock)*field(x)-K0*(L@x)).ravel()
        def means(z):return z.reshape(-1,N,2).mean(axis=1)
    elif formulation=='scaled_normal':
        vals,V=np.linalg.eigh(L);vp=V[:,1:];normal_rates=K0*vals[1:]
        base=X0.mean(axis=0);z0=np.r_[base,(vp.T@(X0-base)).ravel()]
        def rhs(clock,z):
            y=z[:2];w=z[2:].reshape(N-1,2);scaled=vp@w;a=np.exp(-rho*clock)
            eta=a*scaled;dot=scaled@y;norm2=np.sum(scaled*scaled,axis=1)
            # e^(rho*tau) times the exact cubic Taylor remainder.
            rem_scaled=-(2*a*dot[:,None]*scaled+a*norm2[:,None]*y+a*a*norm2[:,None]*scaled)
            h=a*rem_scaled.mean(axis=0)
            D=(1-np.dot(y,y))*np.eye(2)-2*np.outer(y,y)+OMEGA*J
            dy=T*np.exp(-clock)*(field(y[None,:])[0]+h)
            dw=-(normal_rates-rho)[:,None]*w+T*np.exp(-clock)*(w@D.T+vp.T@rem_scaled)
            return np.r_[dy,dw.ravel()]
        def means(z):return z[:,:2]
    else:raise ValueError(formulation)
    sol=solve_ivp(rhs,(0,35),z0,t_eval=times,method='DOP853',rtol=rtol,atol=atol,max_step=maxstep)
    if not sol.success:raise RuntimeError(sol.message)
    ys=means(sol.y.T);sample_indices=np.searchsorted(times,tau);samples=ys[sample_indices]
    endpoints=ys[np.searchsorted(times,ends)];transported=np.array([flow(y,T*np.exp(-u)) for y,u in zip(endpoints,ends)])
    raw= np.linalg.norm(samples-flow(endpoints[1],-T*s),axis=1)
    aligned=np.linalg.norm(samples-flow(transported[-1],-T*s),axis=1)
    mask=(s>=1e-3)&(s<=.05)&(raw>5e-13)
    amask=(s>=1e-3)&(s<=.05)&(aligned>5e-13)
    record={'formulation':formulation,'rtol':rtol,'atol':atol,'max_tau_step':None if not np.isfinite(maxstep) else maxstep,
            'nfev':sol.nfev,'terminal_times_tau':ends.tolist(),'raw_endpoints':endpoints.tolist(),
            'transported_endpoints':transported.tolist(),
            'raw_tau25_to_tau30_difference':float(np.linalg.norm(endpoints[1]-endpoints[0])),
            'transported_tau25_to_tau30_difference':float(np.linalg.norm(transported[1]-transported[0])),
            'transported_tau30_to_tau35_difference':float(np.linalg.norm(transported[2]-transported[1])),
            'transport_correction_at_tau30':float(np.linalg.norm(transported[1]-endpoints[1])),
            'raw_tau30_slope':float(np.polyfit(np.log(s[mask]),np.log(raw[mask]),1)[0]),
            'transported_tau35_slope':float(np.polyfit(np.log(s[amask]),np.log(aligned[amask]),1)[0]),
            'raw_fit_points':int(mask.sum()),'transported_fit_points':int(amask.sum()),
            'minimum_raw_fitted_tail':float(min(raw[mask])),'minimum_transported_fitted_tail':float(min(aligned[amask])),
            'roundtrip_closure_of_transported_point':float(np.linalg.norm(flow(flow(transported[-1],-T),T)-transported[-1]))}
    sens=[]
    for direction in (np.array([1.,0.]),np.array([0.,1.]),np.ones(2)/np.sqrt(2)):
        for sign in (-1,1):
            perturb=sign*1e-12*direction;tail=np.linalg.norm(samples-flow(transported[-1]+perturb,-T*s),axis=1)
            smask=(s>=1e-3)&(s<=.05)&(tail>5e-13)
            sens.append({'direction_x':float(direction[0]),'direction_y':float(direction[1]),'sign':sign,
                         'endpoint_perturbation_norm':float(np.linalg.norm(perturb)),
                         'fitted_slope':float(np.polyfit(np.log(s[smask]),np.log(tail[smask]),1)[0]),
                         'fit_points':int(smask.sum())})
    rows=[{'s':float(q),'tau':float(u),'raw_tau30_tail':float(v),'transported_tau35_tail':float(w),
           'raw_fit_selected':int(m),'transported_fit_selected':int(n)} for q,u,v,w,m,n in zip(s,tau,raw,aligned,mask,amask)]
    return record,rows,sens


def run(archive:Archive,out:Path)->dict:
    ck=Checks();source=archive.text('code/network_experiments.py')
    original=archive.array('data/records/phase_map_validation.csv');summary=archive.json('data/records/numerical_summary.json')['phase_map_validation']
    ck.require('source:raw_terminal_difference_is_not_transport','terminal_convergence_error = float(np.linalg.norm(y_terminal - y_terminal_25))' in source)
    ck.require('source:terminal_mean_is_tau30','y_terminal = y_all[-1]' in source)
    s=np.geomspace(.2,5e-4,240);oldtail=original['stuart_landau_phase_tail'];mask=(s>=1e-3)&(s<=.05)&(oldtail>5e-13)
    oldslope=float(np.polyfit(np.log(s[mask]),np.log(oldtail[mask]),1)[0])
    ck.close('frozen_curve:registered_slope',oldslope,summary['stuart_landau_fitted_slope'],atol=1e-12)
    ck.require('frozen_curve:fit_count',int(mask.sum())==156)
    configs=[('full_state',2e-12,2e-13,np.inf),('full_state',1e-13,1e-14,.1),
             ('full_state',2.3e-14,1e-15,.05),('scaled_normal',1e-13,1e-14,.1),('scaled_normal',2.3e-14,1e-15,.05)]
    records=[];sensitivity=[]
    for i,config in enumerate(configs):
        row,curves,sens=integrate(*config);records.append(row)
        write_csv(out/f'phase_curves_run_{i+1}.csv',curves)
        sensitivity.extend({'run':i+1,**r} for r in sens)
        ck.require(f'run{i+1}:raw_endpoint_motion_above_minimum_tail',row['raw_tau25_to_tau30_difference']>row['minimum_raw_fitted_tail'])
        ck.require(f'run{i+1}:transported_25_30_difference',row['transported_tau25_to_tau30_difference']<3e-13)
        ck.require(f'run{i+1}:transported_30_35_difference',row['transported_tau30_to_tau35_difference']<3e-13)
        ck.require(f'run{i+1}:raw_and_transported_fit_points',row['raw_fit_points']==156 and row['transported_fit_points']==156)
        ck.require(f'run{i+1}:finite_window_slope_regime',3.39<row['transported_tau35_slope']<3.43)
        ck.require(f'run{i+1}:roundtrip_not_error_certificate',row['roundtrip_closure_of_transported_point']<1e-13)
        print('phase',i+1,config,'raw',row['raw_tau30_slope'],'transported',row['transported_tau35_slope'],flush=True)
    reference=np.array(records[-1]['transported_endpoints'][-1])
    for i,row in enumerate(records):
        change=float(np.linalg.norm(np.array(row['transported_endpoints'][-1])-reference))
        row['transported_tau35_difference_from_finest_scaled_run']=change
        ck.require(f'run{i+1}:terminal_solver_formulation_agreement',change<8e-13)
    # Independent RHS check at finite disagreements, not just on the invariant set.
    vals,V=np.linalg.eigh(L);vp=V[:,1:];y=X0.mean(axis=0);eta=X0-y
    for i,tau in enumerate((0.,.5,2.)):
        q=np.exp(-1.2*tau);e=q*eta;xx=y+e;ff=field(xx)
        rem=-(2*(e@y)[:,None]*e+np.sum(e*e,axis=1)[:,None]*y+np.sum(e*e,axis=1)[:,None]*e)
        D=(1-np.dot(y,y))*np.eye(2)-2*np.outer(y,y)+OMEGA*J
        expansion=field(y[None,:])[0]+e@D.T+rem
        ck.require(f'formulation{i}:exact_cubic_expansion',np.max(abs(expansion-ff))<3e-15)
        ck.require(f'formulation{i}:mean_remainder',np.max(abs(ff.mean(axis=0)-(field(y[None,:])[0]+rem.mean(axis=0))))<3e-15)
    write_csv(out/'phase_endpoint_sensitivity.csv',sensitivity)
    report={'status':'PASS','frozen_curve_slope':oldslope,'frozen_raw_endpoint_summary':summary,
            'grid':{'s_start':.2,'s_end':5e-4,'samples':240,'fit_s_low':1e-3,'fit_s_high':.05,'tail_threshold':5e-13,'fit_count':156},
            'endpoint_transport':'yhat_T(tau)=phi_{T exp(-tau)}(y(T(1-exp(-tau))))',
            'numerical_endpoint_convergence_is_not_a_rigorous_error_bound':True,
            'existing_phase_plot_unchanged':True,'runs':records,'sensitivity':sensitivity,
            'interpretation':'The raw 25-to-30 difference includes retained reduced-flow motion. Transport removes this motion; solver/formulation comparisons assess numerical consistency but do not certify the exact terminal point.',**ck.report()}
    write_json(out/'phase_reference.json',report)
    return {'checks':len(ck.records),'solver_runs':len(records),'endpoint_sensitivity_cases':len(sensitivity)}
