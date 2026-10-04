#!/usr/bin/env python3
"""Reconstruct uncertainty statements from the sealed pathwise network outputs.

No trajectories are simulated and no reference record is overwritten.  The
intervals quantify sampling uncertainty for fixed-step estimands, not a bound
on continuous-time model error.  Student-t calibration is asymptotic here.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.stats import t as student_t

from artifact_paths import work_dir, output_dir, code_dir
ROOT = work_dir()
ARCHIVE = 'stochastic_crn_pathwise_terminal_mse.npz'
RATIOS = np.array([.60,.75,.85,.90,.95,.975,.990,.995,1.,1.005,1.010,1.025,1.05,1.10,1.25,1.50])
REFINEMENT_RATIOS = np.array([.95,1.,1.05])
FIT_RATIOS = np.array([.975,.990,.995,1.,1.005,1.010,1.025])
N_PATHS, N_REFINEMENT = 20000,10000
T, K0, SIGMA = 2.,1.2,.002
H, H_FINE = .001,.0005
SEED, ALPHA = 20260829,.05


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def write_csv(path: Path, rows: list[dict[str,Any]]) -> None:
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def interval(values: np.ndarray, family: int = 1) -> dict[str,float]:
    """Two-sided Student-t interval using sample SD (ddof=1)."""
    if values.ndim!=1 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError('Expected at least two finite scalar path outcomes')
    mean=float(np.mean(values));se=float(np.std(values,ddof=1)/math.sqrt(len(values)))
    critical=float(student_t.ppf(1-ALPHA/(2*family),len(values)-1))
    return dict(mean=mean,se=se,critical=critical,low=mean-critical*se,high=mean+critical*se)


def independent_modal(caps: np.ndarray) -> np.ndarray:
    """Integrate transition-weighted noise; independent of archived formulas."""
    lap=2*np.eye(6)-np.roll(np.eye(6),1,axis=1)-np.roll(np.eye(6),-1,axis=1)
    lam,vec=np.linalg.eigh(lap)
    a=np.array([.08,-.05,.12,-.10,.03,-.08]);initial=2*(a-a.mean())
    xi=vec.T@initial
    result=[]
    for cap in caps:
        switch=T-1/cap; total=0.
        for j in range(1,6):
            rho=K0*lam[j]
            def integrand(t:float) -> float:
                s=T-t
                gain=1/s if t<switch else cap
                mass=1+math.log(cap*s) if t<switch else cap*s
                return (rho*gain*math.exp(-rho*mass))**2
            noise=quad(integrand,0,switch,epsabs=1e-12,epsrel=2e-12)[0]+quad(integrand,switch,T,epsabs=1e-12,epsrel=2e-12)[0]
            total+=xi[j]**2*math.exp(-2*rho*(1+math.log(T*cap)))+SIGMA**2*noise
        result.append(total)
    return np.array(result)


def main(output: Path) -> int:
    output.mkdir(parents=True,exist_ok=True)
    checks=[]
    def check(name:str,ok:Any,**evidence:Any)->None:
        checks.append(dict(name=name,passed=bool(ok),**evidence))
    def close(name:str,a:Any,b:Any,rtol:float=3e-12,atol:float=2e-18)->None:
        check(name,np.allclose(a,b,rtol=rtol,atol=atol),max_abs_error=float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
    with np.load(ROOT/ARCHIVE,allow_pickle=False) as z:
        caps=z['caps'].copy(); kopt=float(z['Kopt'][0]); x=z['terminal_squared_disagreement'].copy()
        refcaps=z['refinement_caps'].copy();fine=z['refinement_dt_5e4'].copy();coarse=z['refinement_dt_1e3'].copy();seed=int(z['seed'][0])
    check('primary shape 20000 by 16',x.shape==(N_PATHS,16))
    check('nested shapes 10000 by 3',fine.shape==coarse.shape==(N_REFINEMENT,3))
    check('finite nonnegative squared disagreements',all(np.isfinite(v).all() and (v>=0).all() for v in [x,fine,coarse]))
    check('random seed',seed==SEED)
    close('full cap grid',caps/kopt,RATIOS)
    close('refinement cap grid',refcaps/kopt,REFINEMENT_RATIOS)
    refindex=int(np.argmin(abs(caps-kopt)))
    indices=[int(np.argmin(abs(caps-c))) for c in refcaps]
    check('fine values are exactly the first 10000 primary paths',np.array_equal(fine,x[:N_REFINEMENT,indices]))
    check('reference cap included exactly',caps[refindex]==kopt)
    archive_rows=list(csv.DictReader((ROOT/'stochastic_crn_cap_validation.csv').open()))
    check('archived cap table has all 16 rows',len(archive_rows)==16)
    modal=np.array([float(row['modal_exact_mse']) for row in archive_rows])
    exact=independent_modal(caps)
    close('transition-integral reconstruction of all modal values',exact,modal,rtol=2e-12)
    primary_rows=[];unresolved=[];point_better=[];rejected_worse=[]
    for j,c in enumerate(caps):
        p=interval(x[:,j]);d=x[:,j]-x[:,refindex];paired=interval(d);sim=interval(d,15)
        delta=(p['mean']/modal[j]-1)*100
        rel_low=(p['low']/modal[j]-1)*100;rel_high=(p['high']/modal[j]-1)*100
        row=dict(K=float(c),K_over_Kopt=float(c/kopt),paths=N_PATHS,dt=H_FINE,
            modal_mse=float(modal[j]),nonlinear_mean=p['mean'],se=p['se'],ci95_low=p['low'],ci95_high=p['high'],
            relative_discrepancy_pct=delta,relative_se_modal_pct=100*p['se']/modal[j],
            relative_se_estimate_pct=100*p['se']/p['mean'],relative_ci95_low_pct=rel_low,relative_ci95_high_pct=rel_high,
            paired_difference=paired['mean'],paired_se=paired['se'],paired_ci95_low=paired['low'],paired_ci95_high=paired['high'],
            bonferroni15_ci95_low=sim['low'],bonferroni15_ci95_high=sim['high'],
            significantly_worse=sim['low']>0,significantly_better=sim['high']<0,
            unresolved_vs_reference=sim['low']<=0<=sim['high'])
        primary_rows.append(row)
        if row['unresolved_vs_reference']:unresolved.append(float(c/kopt))
        if row['significantly_better']:point_better.append(float(c/kopt))
        if row['significantly_worse']:rejected_worse.append(float(c/kopt))
        pairs=[('nonlinear_mc_mse','nonlinear_mean'),('mc_se','se'),('mc_ci95_low','ci95_low'),('mc_ci95_high','ci95_high'),
               ('paired_difference_vs_Kopt','paired_difference'),('paired_se','paired_se'),('paired_ci95_low','paired_ci95_low'),
               ('paired_ci95_high','paired_ci95_high'),('paired_simultaneous_ci95_low','bonferroni15_ci95_low'),
               ('paired_simultaneous_ci95_high','bonferroni15_ci95_high')]
        for old,new in pairs:close(f'cap {c/kopt:.3f} archived {old}',row[new],float(archive_rows[j][old]))
        # An independent high-accuracy summation path validates the estimators.
        fsum_mean=math.fsum(x[:,j])/N_PATHS
        fsum_se=math.sqrt(math.fsum((float(v)-fsum_mean)**2 for v in x[:,j])/(N_PATHS-1)/N_PATHS)
        close(f'cap {c/kopt:.3f} independent mean',p['mean'],fsum_mean)
        close(f'cap {c/kopt:.3f} independent SE',p['se'],fsum_se)
        close(f'cap {c/kopt:.3f} exact relative CI centering',(rel_low+rel_high)/2,delta,atol=2e-13)
        close(f'cap {c/kopt:.3f} covariance preserves paired SE',paired['se']**2,
              (np.var(x[:,j],ddof=1)+np.var(x[:,refindex],ddof=1)-2*np.cov(x[:,j],x[:,refindex],ddof=1)[0,1])/N_PATHS,atol=2e-27)
    close('unresolved four-point tested set',unresolved,[.99,.995,1.,1.005])
    check('no tested cap significantly improves on reference',not point_better)
    check('12 remaining tested caps significantly worse',len(rejected_worse)==12)
    maxidx=int(np.argmax([abs(r['relative_discrepancy_pct']) for r in primary_rows]))
    check('maximum point discrepancy occurs at ratio 1.5',maxidx==15)
    check('reference empirical grid minimum',int(np.argmin(x.mean(0)))==refindex)
    write_csv(output/'stochastic_cap_statistics.csv',primary_rows)

    # Independent QR/pseudoinverse fit instead of archived normal equations.
    fitindices=[int(np.argmin(abs(caps/kopt-v))) for v in FIT_RATIOS]
    coordinates=caps[fitindices]/kopt-1
    design=np.column_stack((np.ones(7),coordinates,coordinates**2))
    coefficients=x[:,fitindices]@np.linalg.pinv(design).T
    b=coefficients.mean(0);covariance=np.cov(coefficients,rowvar=False,ddof=1)/N_PATHS
    vertex=kopt*(1-b[1]/(2*b[2]))
    gradient=np.array([0.,-kopt/(2*b[2]),kopt*b[1]/(2*b[2]**2)])
    vse=float(np.sqrt(gradient@covariance@gradient));q=float(student_t.ppf(.975,N_PATHS-1))
    observed=x[:,fitindices].mean(0);residual=observed-design@b
    rsq=float(1-residual@residual/np.sum((observed-observed.mean())**2))
    fit=dict(ratios=FIT_RATIOS.tolist(),vertex=float(vertex),se=vse,ci95_low=float(vertex-q*vse),ci95_high=float(vertex+q*vse),
             r_squared=rsq,coefficients=b.tolist(),covariance_of_mean=covariance.tolist(),
             estimand='Vertex of the least-squares quadratic projection of fixed-step expected outcomes on the specified seven-point design.',
             interval_scope='Approximate delta-method sampling interval. Excludes quadratic-model error, grid error and time-discretization bias.')
    old_summary=json.loads((ROOT/'stochastic_summary.json').read_text())
    for field,old in [('vertex','estimated_cap'),('se','standard_error'),('ci95_low','ci95_low'),('ci95_high','ci95_high'),('r_squared','r_squared')]:
        close('quadratic independent '+field,fit[field],old_summary['quadratic_localization'][old],rtol=3e-10,atol=3e-12)
    check('quadratic curvature positive',b[2]>0)
    check('quadratic CI contains modal cap',fit['ci95_low']<kopt<fit['ci95_high'])
    write_json(output/'stochastic_quadratic_fit.json',fit)

    step_rows=[];rich_rows=[];split_rows=[]
    archived_steps=list(csv.DictReader((ROOT/'stochastic_step_refinement.csv').open()))
    for j,idx in enumerate(indices):
        p=interval(fine[:,j]);c=interval(coarse[:,j]);d=interval(coarse[:,j]-fine[:,j]);r=interval(2*fine[:,j]-coarse[:,j])
        covariance=float(np.cov(fine[:,j],coarse[:,j],ddof=1)[0,1]);n=N_REFINEMENT;mo=float(modal[idx])
        reconstructed_var=4*np.var(fine[:,j],ddof=1)+np.var(coarse[:,j],ddof=1)-4*covariance
        close(f'refinement {RATIOS[idx]} pathwise Richardson variance identity',r['se']**2,reconstructed_var/n,rtol=2e-11,atol=2e-25)
        close(f'refinement {RATIOS[idx]} pathwise mean identity',r['mean'],2*p['mean']-c['mean'])
        close(f'refinement {RATIOS[idx]} difference variance identity',d['se']**2,
              (np.var(fine[:,j],ddof=1)+np.var(coarse[:,j],ddof=1)-2*covariance)/n,rtol=2e-9,atol=2e-27)
        sr=dict(K=float(caps[idx]),K_over_Kopt=float(RATIOS[idx]),paths=n,coarse_dt=H,fine_dt=H_FINE,
                coarse_mean=c['mean'],fine_mean=p['mean'],coarse_se=c['se'],fine_se=p['se'],
                paired_difference=d['mean'],paired_se=d['se'],paired_ci95_low=d['low'],paired_ci95_high=d['high'],
                relative_shift_pct=100*d['mean']/p['mean'],fine_coarse_covariance=covariance,
                fine_coarse_correlation=float(np.corrcoef(fine[:,j],coarse[:,j])[0,1]))
        rr=dict(K=float(caps[idx]),K_over_Kopt=float(RATIOS[idx]),paths=n,modal_mse=mo,
                richardson_mean=r['mean'],richardson_se=r['se'],ci95_low=r['low'],ci95_high=r['high'],
                relative_discrepancy_pct=100*(r['mean']/mo-1),relative_ci95_low_pct=100*(r['low']/mo-1),
                relative_ci95_high_pct=100*(r['high']/mo-1),relative_ci95_halfwidth_pct=100*(r['high']-r['low'])/(2*mo),
                covariance_corrected_variance=float(reconstructed_var),negative_extrapolated_path_outcomes=int(np.sum(2*fine[:,j]-coarse[:,j]<0)))
        step_rows.append(sr);rich_rows.append(rr)
        for old,new in [('mean_dt_1e-3','coarse_mean'),('mean_dt_5e-4','fine_mean'),('paired_dt_1e-3_minus_5e-4','paired_difference'),
                        ('paired_se','paired_se'),('paired_ci95_low','paired_ci95_low'),('paired_ci95_high','paired_ci95_high')]:
            close(f'refinement {RATIOS[idx]} archived {old}',sr[new],float(archived_steps[j][old]))
        check(f'refinement {RATIOS[idx]} measured step shift positive',d['low']>0)
        check(f'Richardson {RATIOS[idx]} CI does not certify 0.04 percent accuracy',rr['relative_ci95_halfwidth_pct']>1)
        a=x[:n,idx];b2=x[n:,idx]
        split_rows.append(dict(K_over_Kopt=float(RATIOS[idx]),first_10000_mean=float(a.mean()),last_10000_mean=float(b2.mean()),all_20000_mean=float(x[:,idx].mean()),
             first_minus_last=float(a.mean()-b2.mean()),independent_blocks_se=float(np.sqrt(np.var(a,ddof=1)/n+np.var(b2,ddof=1)/n))))
        close(f'refinement {RATIOS[idx]} full mean is block average',x[:,idx].mean(),(a.mean()+b2.mean())/2)
    write_csv(output/'stochastic_step_statistics.csv',step_rows)
    write_csv(output/'stochastic_richardson_statistics.csv',rich_rows)
    write_csv(output/'stochastic_sample_split.csv',split_rows)
    check('same observed three-cap ordering at both steps',all(a.mean(0)[0]>a.mean(0)[1] and a.mean(0)[2]>a.mean(0)[1] for a in [fine,coarse]))

    # Guards against the sampling-uncertainty conventions.
    check('marginal and family critical values differ',student_t.ppf(1-ALPHA/30,N_PATHS-1)>student_t.ppf(.975,N_PATHS-1))
    check('reference t interval full width is twice halfwidth',np.isclose((primary_rows[8]['ci95_high']-primary_rows[8]['ci95_low'])/2,
                                                                       q*primary_rows[8]['se'],rtol=1e-12,atol=0))
    check('first-half fine mean not mislabeled full-sample mean',fine[:,1].mean()!=x[:,8].mean())
    check('reference paired difference is identically zero',np.array_equal(x[:,8]-x[:,8],np.zeros(N_PATHS)))
    # Analytic synthetic examples expose covariance dropping and wrong ddof.
    toy=np.array([1.,2.,4.,8.,16.]);co=toy+1;fi=toy
    close('synthetic extrapolation retains shared variance',interval(2*fi-co)['se'],interval(toy)['se'])
    check('synthetic independent-error formula is wrong for paired inputs',not np.isclose(4*np.var(fi,ddof=1)+np.var(co,ddof=1),np.var(2*fi-co,ddof=1)))
    close('synthetic sample SE ddof=1',interval(np.array([1.,2.,3.]))['se'],1/math.sqrt(3))

    validation=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',checks_passed=sum(c['passed'] for c in checks),checks_total=len(checks),checks=checks)
    write_json(output/'stochastic_reporting_validation.json',validation)
    (output/'stochastic_reporting_validation.txt').write_text('\n'.join([f"{validation['status']}: {validation['checks_passed']}/{len(checks)}"]+
        [f"[{'PASS' if c['passed'] else 'FAIL'}] {c['name']}" for c in checks])+'\n')
    summary=dict(status=validation['status'],checks_passed=validation['checks_passed'],checks_total=len(checks),
        sample_provenance=dict(primary_paths=N_PATHS,refinement_paths=N_REFINEMENT,independent_paths_total=N_PATHS,
            refinement_primary_row_range_zero_based=[0,N_REFINEMENT-1],fine_refinement_exact_primary_subset=True,
            primary_dt=H_FINE,coarse_dt=H,seed=SEED,generator='NumPy PCG64',batch_size=200,
            stochastic_initial_phase_offsets=(2*(np.array([.08,-.05,.12,-.10,.03,-.08])-np.mean([.08,-.05,.12,-.10,.03,-.08]))).tolist(),
            notes='Independent paths; Brownian increments shared across caps; adjacent fine increments summed for each coarse increment. No 30000-path claim.'),
        Kopt=kopt,reference=primary_rows[refindex],maximum_point_discrepancy=primary_rows[maxidx],
        unresolved_tested_ratios=unresolved,significantly_worse_tested_ratios=rejected_worse,
        significantly_better_tested_ratios=point_better,bonferroni_comparisons=15,
        marginal_critical=float(student_t.ppf(.975,N_PATHS-1)),bonferroni_critical=float(student_t.ppf(1-ALPHA/30,N_PATHS-1)),
        quadratic_fit=fit,richardson=rich_rows,step_refinement=step_rows,
        interpretation=dict(intervals='Approximate sampling intervals for fixed-step means; squared disagreements need not be normally distributed.',
            pairing='Bonferroni family is the 15 nonzero contrasts against the analytical reference, not all pairwise differences.',
            grid='The unresolved set is not a continuous-domain optimizer confidence interval.',
            quadratic='Delta-method CI for the seven-point quadratic-projection vertex; not a theorem for the nonlinear optimizer.',
            richardson='Diagnostic expectation 2 M_(h/2)-M_h. No validated weak-order expansion or certified continuous-time accuracy.',
            discrepancies='Point discrepancies combine sampling fluctuation, time-discretization bias and nonlinear-modal mismatch.'),
        input_sha256={n:sha((code_dir() if n.endswith(".py") else ROOT)/n) for n in [ARCHIVE,'stochastic_crn_cap_validation.csv','stochastic_step_refinement.csv','stochastic_summary.json','stochastic_network.py']},
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        trajectories_simulated=0)
    write_json(output/'stochastic_reporting_summary.json',summary)
    # Rounded numerical tables use the same scales and decimal places as the
    # complete statistical records. They contain data only, without typesetting.
    from table_reporting import write_display_tables
    write_display_tables(output, primary_rows, step_rows, rich_rows)
    print(f"{validation['status']}: {validation['checks_passed']}/{len(checks)} stochastic reporting checks")
    print('Maximum point discrepancy:',primary_rows[maxidx]['relative_discrepancy_pct'],'percent at',primary_rows[maxidx]['K_over_Kopt'])
    print('Unresolved tested set:',unresolved)
    print('Output:',output)
    return 0 if validation['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'reporting_statistics')
    args=parser.parse_args()
    raise SystemExit(main(args.output_dir.resolve()))
