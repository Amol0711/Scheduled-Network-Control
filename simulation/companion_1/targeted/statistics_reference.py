"""Independent QR/delta-method reconstruction from the unchanged pathwise arrays.
Uses no archived calculation modules and generates no new trajectories.
"""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
from scipy.stats import t as student_t
from scipy.optimize import brentq
from reference_common import Archive,Checks,write_csv,write_json,digest
from ode_common import L,K0,T


def run(archive:Archive,out:Path)->dict:
    ck=Checks();path='data/records/stochastic_crn_pathwise_terminal_mse.npz';data=archive.npz(path)
    caps=data['caps'];k=float(data['Kopt'][0]);Z=data['terminal_squared_disagreement'];n=len(Z)
    ck.require('primary_shape',Z.shape==(20000,16));ck.require('finite_nonnegative_outcomes',np.isfinite(Z).all() and (Z>=0).all())
    ck.require('seed',int(data['seed'][0])==20260829)
    ratios=np.array([.975,.99,.995,1.,1.005,1.01,1.025]);indices=np.array([np.argmin(abs(caps/k-r)) for r in ratios])
    ck.require('fixed_seven_point_design',np.allclose(caps[indices]/k,ratios,rtol=0,atol=1e-14))
    x=caps[indices]/k-1;A=np.column_stack((np.ones(7),x,x*x));Q,R=np.linalg.qr(A)
    b=np.linalg.solve(R,Q.T@Z[:,indices].T).T
    bm=np.array([math.fsum(b[:,j])/n for j in range(3)])
    covariance=np.cov(b,rowvar=False,ddof=1)/n
    vertex=k*(1-bm[1]/(2*bm[2]));gradient=np.array([0.,-k/(2*bm[2]),k*bm[1]/(2*bm[2]**2)])
    se=float(np.sqrt(gradient@covariance@gradient));crit=float(student_t.ppf(.975,n-1));interval=[float(vertex-crit*se),float(vertex+crit*se)]
    old=archive.json('data/records/stochastic_summary.json')['quadratic_localization']
    for name,value,prior in [('vertex',float(vertex),old['estimated_cap']),('standard_error',se,old['standard_error']),
                              ('lower',interval[0],old['ci95_low']),('upper',interval[1],old['ci95_high'])]:
        ck.close('quadratic:'+name,value,float(prior),atol=2e-12)
    direct=np.linalg.lstsq(A,np.array([math.fsum(Z[:,j])/n for j in indices]),rcond=None)[0]
    for j in range(3):ck.close(f'quadratic:mean_fit_equivalence{j}',float(bm[j]),float(direct[j]),atol=3e-15)
    ck.require('quadratic:positive_curvature',bm[2]>0)
    ck.require('quadratic:vertex_inside_design',caps[indices].min()<vertex<caps[indices].max())
    # Nested sample relation is exact, not an independent second sample.
    ridx=np.array([np.argmin(abs(caps-r)) for r in data['refinement_caps']])
    ck.require('nested_fine_array_exact',np.array_equal(Z[:10000,ridx],data['refinement_dt_5e4']))
    ck.require('coarse_shape',data['refinement_dt_1e3'].shape==(10000,3))
    # Reconstruct all displayed cap means and uncertainty conventions without changing estimators.
    table=archive.csv('data/statistics/stochastic_cap_statistics.csv');qbonf=float(student_t.ppf(1-.05/30,n-1));ref=int(np.argmin(abs(caps/k-1)))
    ck.close('marginal_t_quantile_inverse_cdf',float(student_t.cdf(crit,n-1)),.975,atol=1e-13)
    ck.close('marginal_t_quantile_printed_rounding',crit,1.9600826,atol=5e-8)
    caprows=[]
    for j,K in enumerate(caps):
        mean=math.fsum(Z[:,j])/n;dev=Z[:,j]-mean;sem=math.sqrt(math.fsum(dev*dev)/(n*(n-1)))
        D=Z[:,j]-Z[:,ref];dm=math.fsum(D)/n;ds=D-dm;dsse=math.sqrt(math.fsum(ds*ds)/(n*(n-1)))
        values={'K':float(K),'nonlinear_mean':mean,'se':sem,'ci95_low':mean-crit*sem,'ci95_high':mean+crit*sem,
                'paired_difference':dm,'paired_se':dsse,'bonferroni15_ci95_low':dm-qbonf*dsse,'bonferroni15_ci95_high':dm+qbonf*dsse}
        for key,val in values.items():ck.close(f'cap{j}:{key}',val,float(table[j][key]),atol=2e-17,rtol=1e-12)
        caprows.append(values)
    unresolved=[float(caps[j]/k) for j,r in enumerate(caprows) if r['bonferroni15_ci95_low']<=0<=r['bonferroni15_ci95_high']]
    ck.require('four_caps_three_nonzero_contrasts',np.allclose(unresolved,[.99,.995,1,1.005],rtol=0,atol=1e-14))
    # Independent phase-offset versus Cartesian initialization check, for protocol traceability only.
    vals,V=np.linalg.eigh(L);rho=K0*vals[1:];sig=.002
    phase=2*np.array([.08,-.05,.12,-.10,.03,-.08]);phase-=phase.mean()
    C=rho*rho*np.exp(-2*rho)/(2*rho+1);B=.5*rho*(1-np.exp(-2*rho))+C;initializations={}
    for name,u in [('phase_offsets',phase),('Cartesian_tangent',np.sin(phase)-np.sin(phase).mean())]:
        xi=V[:,1:].T@u;D=xi*xi*np.exp(-2*rho)*T**(-2*rho)-C*sig**2*T**(-(2*rho+1))
        def value(cap):return np.sum(D*cap**(-2*rho)+sig*sig*B*cap)
        def derivative(cap):return np.sum(-2*rho*D*cap**(-2*rho-1)+sig*sig*B)
        opt=brentq(derivative,.5,100,xtol=1e-14)
        initializations[name]={'root_optimal_cap':float(opt),'mse_at_reference_cap':float(value(k))}
    ck.close('phase_initialization:cap_matches',initializations['phase_offsets']['root_optimal_cap'],k,atol=1e-8)
    relative=100*(initializations['Cartesian_tangent']['mse_at_reference_cap']/initializations['phase_offsets']['mse_at_reference_cap']-1)
    ck.require('Cartesian_is_distinct_not_substituted',relative<-.6)
    report={'npz_sha256':digest(archive.raw(path)),'paths':n,'caps':16,'seed':20260829,
            'reference_cap':k,'design_ratios':ratios.tolist(),'cap_indices':indices.tolist(),
            'qr_condition_number_design':float(np.linalg.cond(A)),
            'mean_coefficients':bm.tolist(),'coefficient_mean_covariance':covariance.tolist(),
            'gradient':gradient.tolist(),'vertex':float(vertex),'standard_error':se,'CI95':interval,
            'student_t_quantile':crit,'initialization_checks':initializations,
            'Cartesian_MSE_difference_pct_at_reference':float(relative),'unresolved_cap_ratios':unresolved,
            'estimator_interpretation':'Vertex of the fixed seven-point quadratic projection of the fixed-step mean curve; delta-method large-sample interval, not a continuous-domain optimizer confidence interval.',
            'same_paths_reused':True,'new_trajectories':0,**ck.report()}
    write_json(out/'statistics_reference.json',report);write_csv(out/'cap_statistics_reconstructed.csv',caprows)
    print('statistics vertex',vertex,'SE',se,'CI',interval,flush=True)
    return {'checks':len(ck.records),'pathwise_arrays':len(data),'cap_rows':len(caprows)}
