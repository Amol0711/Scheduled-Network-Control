"""High-precision successor for the p=3 Lambert-W numerical comparison.
Independent split-integral derivative and an erfi antiderivative cross-check.
No global-minimum theorem or stronger asymptotic remainder is inferred from
finite numerical grids, root residuals, or precision refinement.
"""
from __future__ import annotations
from pathlib import Path
import mpmath as mp
from reference_common import Archive, Checks, write_csv, write_json, digest


def one(dps: int, sigma_string: str) -> dict:
    with mp.workdps(dps):
        p, rho, horizon, eta0, sig = map(mp.mpf, ('3','1.2','1','1',sigma_string))
        H = eta0**2 * mp.exp(2*rho/(p-1))
        Z = 2*rho*p*(4*horizon*H/sig**2)**(p-1)
        theta_L = (mp.lambertw(Z)/(2*rho*p))**(1/(p-1))

        def parts(theta, method='quadrature'):
            bias = eta0**2*mp.exp(-rho*(3*theta**2-1))
            if method == 'quadrature':
                pre = rho**2*sig**2/horizon*mp.quad(
                    lambda z:z**4*mp.exp(rho*(z*z-3*theta*theta)),[1,theta])
            elif method == 'antiderivative':
                def primitive(z):
                    return (mp.exp(rho*z*z)*(z**3/(2*rho)-3*z/(4*rho**2))
                            +3*mp.sqrt(mp.pi)*mp.erfi(mp.sqrt(rho)*z)/(8*rho**mp.mpf('2.5')))
                pre = rho**2*sig**2/horizon*mp.exp(-3*rho*theta**2)*(primitive(theta)-primitive(1))
            else:
                raise ValueError(method)
            e = mp.exp(-2*rho*theta**2)
            cap = rho*sig**2/(2*horizon)*theta**3*(1-e)
            return bias,pre,cap,e

        def obj(theta, method='quadrature'):
            return sum(parts(theta,method)[:3])

        def deriv(theta, method='quadrature'):
            b,v,_,e = parts(theta,method)
            return (-6*rho*theta*(b+v)
                    +3*rho*sig**2/(2*horizon)*theta**2*(1-e)
                    +3*rho**2*sig**2/horizon*theta**4*e)/sig**2

        root = mp.findroot(lambda a:deriv(a),(theta_L*mp.mpf('.9999'),theta_L*mp.mpf('1.0001')),
                           tol=mp.power(10,-dps+12),maxsteps=40)
        alt_root = mp.findroot(lambda a:deriv(a,'antiderivative'),
                              (theta_L*mp.mpf('.9999'),theta_L*mp.mpf('1.0001')),
                              tol=mp.power(10,-dps+12),maxsteps=40)
        # A narrow, explicit numerical bracket and independent derivative calculation.
        width = mp.power(10,-(dps//2))
        left,right = root-width,root+width
        probe = root*mp.mpf('1.00001')
        diff = mp.diff(obj,probe)/sig**2
        b,v,c,_ = parts(root)
        K=root**3/horizon; KL=theta_L**3/horizon
        interior_probes = [1+(root-1)*mp.mpf(i)/20 for i in range(21)]
        upper_probes = [root*(1+mp.mpf(i)/10) for i in range(1,21)]
        objective_probes = interior_probes[:-1]+upper_probes
        s=lambda x:mp.nstr(x,dps-8)
        return {
            'precision_digits':dps,'sigma':sigma_string,'theta_root':s(root),
            'K_derivative_root':s(K),'K_Lambert':s(KL),
            'signed_relative_error_Lambert_over_reference_minus_one':s(KL/K-1),
            'relative_error':s(abs(KL/K-1)), 'optimized_rms':s(mp.sqrt(b+v+c)),
            'bias_squared':s(b),'variance_pre_cap':s(v),'variance_cap':s(c),
            'scaled_derivative_residual':s(deriv(root)),
            'theta_bracket_left':s(left),'theta_bracket_right':s(right),
            'derivative_left':s(deriv(left)),'derivative_right':s(deriv(right)),
            'quadrature_vs_antiderivative_K_relative_difference':s(abs(root**3/alt_root**3-1)),
            'precap_quadrature_vs_antiderivative_relative_difference':s(abs(v/parts(root,'antiderivative')[1]-1)),
            'analytic_vs_automatic_derivative_relative_difference':s(abs(deriv(probe)/diff-1)),
            'finite_objective_probe_count':len(objective_probes),
            'finite_objective_probes_above_candidate':all(obj(a)>obj(root) for a in objective_probes),
            'derivative_negative_below_candidate':all(deriv(a)<0 for a in interior_probes[:-1]),
            'derivative_positive_above_candidate':all(deriv(a)>0 for a in upper_probes),
            'scope':'Numerical stationarity, finite probes and precision refinement; not a global-optimality proof.'}


def run(archive: Archive, out: Path) -> dict:
    checks=Checks(); rows=[]
    for dps in (60,80):
        for sig in ('1e-5','1e-8','1e-12'):
            row=one(dps,sig);rows.append(row); key=f'{dps}:{sig}'
            with mp.workdps(100):
                checks.require(key+':admissible_cap',mp.mpf(row['K_derivative_root'])>1)
                checks.require(key+':small_stationarity_residual',abs(mp.mpf(row['scaled_derivative_residual']))<mp.mpf('1e-45'))
                checks.require(key+':left_sign',mp.mpf(row['derivative_left'])<0)
                checks.require(key+':right_sign',mp.mpf(row['derivative_right'])>0)
                for field in ('quadrature_vs_antiderivative_K_relative_difference',
                              'precap_quadrature_vs_antiderivative_relative_difference',
                              'analytic_vs_automatic_derivative_relative_difference'):
                    checks.require(key+':'+field,mp.mpf(row[field])<mp.mpf('1e-45'))
                for field in ('finite_objective_probes_above_candidate','derivative_negative_below_candidate','derivative_positive_above_candidate'):
                    checks.require(key+':'+field,row[field])
            print('power',key,'relative_error',row['relative_error'],flush=True)
    refinements=[]
    with mp.workdps(100):
        for low,high in zip(rows[:3],rows[3:]):
            delta=abs(mp.mpf(low['K_derivative_root'])/mp.mpf(high['K_derivative_root'])-1)
            de=abs(mp.mpf(low['relative_error'])/mp.mpf(high['relative_error'])-1)
            checks.require(low['sigma']+':60_to_80_K_agreement',delta<mp.mpf('1e-45'))
            checks.require(low['sigma']+':60_to_80_error_agreement',de<mp.mpf('1e-35'))
            refinements.append({'sigma':low['sigma'],'K_relative_change':mp.nstr(delta,40),'error_relative_change':mp.nstr(de,40)})
    old_path='data/records/asymptotic_power_p3_accuracy.csv'
    old=archive.csv(old_path)
    (out/'historical_power_p3_accuracy.csv').write_bytes(archive.raw(old_path))
    successor=[]
    for prior,row in zip(old,rows[3:]):
        checks.close(row['sigma']+':sigma_matches_frozen_row',float(prior['sigma']),float(row['sigma']))
        successor.append({'sigma':row['sigma'],'rho':'1.2','T':'1','eta0':'1','p':3,
                          'precision_digits':80,'K_derivative_root':row['K_derivative_root'],
                          'K_Lambert':row['K_Lambert'],'relative_error':row['relative_error'],
                          'optimized_rms':row['optimized_rms'],
                          'historical_double_minimizer_K':prior['K_exact'],
                          'historical_relative_error':prior['relative_error']})
    write_csv(out/'power_p3_successor.csv',successor)
    report={'status':'PASS','parameters':{'p':3,'eta0':'1','rho':'1.2','T':'1'},
            'historical_record_sha256':digest(archive.raw(old_path)),
            'supersedes':'The maximum 9.1e-9 precision claim, not the p>1 asymptotic theorem.',
            'precision_runs':rows,'precision_refinement':refinements,**checks.report()}
    write_json(out/'power_reference.json',report)
    return {'checks':len(checks.records),'records':len(rows)}
