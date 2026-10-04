#!/usr/bin/env python3
"""Exact, asymptotic and limiting-case numerical checks.

High-precision arithmetic resolves cap remainders below double precision.
Finite-grid checks are diagnostics, not mathematical proofs."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import brentq

from artifact_paths import work_dir, output_dir, code_dir
ROOT = work_dir()


@dataclass(frozen=True)
class Modes:
    rho: np.ndarray
    xi0: np.ndarray
    noise_shape: np.ndarray
    T: float

    def __post_init__(self) -> None:
        for name in ('rho', 'xi0', 'noise_shape'):
            a = np.asarray(getattr(self, name), dtype=float)
            if a.ndim != 1 or not len(a) or not np.all(np.isfinite(a)):
                raise ValueError(f'{name} must be a finite nonempty vector')
            object.__setattr__(self, name, a)
        if not (self.rho.shape == self.xi0.shape == self.noise_shape.shape):
            raise ValueError('Modal vectors must have matching shapes')
        if np.any(self.rho <= 0) or np.any(self.noise_shape < 0):
            raise ValueError('Rates must be positive and noise shapes nonnegative')
        if not math.isfinite(self.T) or self.T <= 0:
            raise ValueError('T must be finite and positive')

    def coefficients(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r = self.rho
        b = r / 2 * (-np.expm1(-2*r))
        C = r*r*np.exp(-2*r)/(2*r+1)
        B = b+C
        a = self.xi0**2*np.exp(-2*r)*self.T**(-2*r)
        c = self.noise_shape**2*C*self.T**(-(2*r+1))
        return a, c, B, b, C

    def mse(self, K: float, sigma: float) -> float:
        if not math.isfinite(K) or K < 1/self.T:
            raise ValueError('K must be finite and at least 1/T')
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError('sigma must be finite and nonnegative')
        a, c, B, _, _ = self.coefficients()
        # Separate nonnegative bias and total variance rather than forming
        # nearly cancelling effective deterministic coefficients.
        powers = K**(-2*self.rho)
        return float(np.dot(a, powers) + sigma**2*(np.dot(self.noise_shape**2, B)*K - np.dot(c, powers)))

    def derivative(self, K: float, sigma: float) -> float:
        a, c, B, _, _ = self.coefficients()
        p = 2*self.rho
        return float(sigma**2*np.dot(self.noise_shape**2, B) - np.dot(p*(a-sigma**2*c), K**(-p-1)))

    def leading(self, sigma: float) -> tuple[float, float, float]:
        if sigma <= 0 or not np.any(self.xi0 != 0) or not np.any(self.noise_shape > 0):
            raise ValueError('Leading equivalents require nonzero bias and effective noise')
        a, _, B, _, _ = self.coefficients()
        rs = float(np.min(self.rho[self.xi0 != 0]))
        csq = float(np.sum(a[self.rho == rs]))
        SB = float(np.dot(self.noise_shape**2, B))
        L = (2*rs*csq/(SB*sigma*sigma))**(1/(2*rs+1))
        minimum = (2*rs+1)*csq*L**(-2*rs)
        return L, minimum, rs

    def optimum(self, sigma: float) -> tuple[float, float]:
        """Constrained finite-cap regression solver on a certified finite bracket.

        The derivative is positive above `upper` by a termwise bound on all
        negative derivative contributions. Within the bracket, every sign
        change on a dense log grid is refined. The analytic global asymptotic
        argument does not rely on this numerical root isolation.
        """
        if sigma < 0 or not math.isfinite(sigma):
            raise ValueError('sigma must be finite and nonnegative')
        a, _, B, _, _ = self.coefficients()
        SB = float(np.dot(self.noise_shape**2, B))
        lower = 1/self.T
        if sigma*sigma*SB == 0:
            return (math.inf, 0.0) if np.any(self.xi0 != 0) else (lower, 0.0)
        p = 2*self.rho
        upper = max(lower*(1+1e-8), float(np.max((2*len(a)*p*a/(SB*sigma*sigma))**(1/(p+1)))))
        grid = np.geomspace(lower, upper, 1500)
        values = [self.derivative(float(k), sigma) for k in grid]
        candidates = [lower, upper]
        for k1, k2, v1, v2 in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
            if v1*v2 < 0:
                root = brentq(lambda u: self.derivative(math.exp(u), sigma), math.log(k1), math.log(k2), xtol=1e-13, rtol=1e-14)
                candidates.append(max(lower, math.exp(root)))
        kopt = min(candidates, key=lambda k: self.mse(k, sigma))
        return kopt, self.mse(kopt, sigma)


CHECKS: list[dict[str, object]] = []


def check(name: str, passed: bool, group: str, **evidence: object) -> None:
    CHECKS.append(dict(name=name, passed=bool(passed), group=group, **evidence))


def close(name: str, actual: float, expected: float, group: str, rtol: float = 2e-10, atol: float = 1e-14) -> None:
    err = abs(actual-expected)
    check(name, err <= atol+rtol*abs(expected), group,
          actual=float(actual), expected=float(expected), absolute_error=float(err), rtol=rtol, atol=atol)


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    with (ROOT/name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def symbolic_checks() -> None:
    r, th, T, e, s, x = sp.symbols('rho theta T eta sigma x', positive=True)
    D = (2*r*r-2*r+1)/(4*r)
    P = th**2/(2*r)-2*th/(2*r)**2+2/(2*r)**3
    M = s*s/T*(r*th*th/2+(1/(4*r)-th/2)*sp.exp(-2*r*th))+(e*e-D*s*s/T)*sp.exp(-4*r*th+2*r)
    dM = s*s/T*(r*th+(r*th-1)*sp.exp(-2*r*th))-4*r*(e*e-D*s*s/T)*sp.exp(-4*r*th+2*r)
    H = x*sp.exp(x)+(x-4)*sp.exp(x/2)
    Z = 16*r*T*e*e*sp.exp(2*r)/(s*s)
    rel = s*s/(4*T)*sp.exp(-x)*(H-Z*(1-D*s*s/(T*e*e)))
    stationary = s*s/T*(r*th*th/2+(1/(4*r)-th/2)*sp.exp(-2*r*th)) + s*s/(4*r*T)*(r*th+(r*th-1)*sp.exp(-2*r*th))
    expected = s*s/(32*r*T)*(x*x+2*x*(1-sp.exp(-x/2)))
    identities = {
        'polynomial_antiderivative': sp.diff(sp.exp(2*r*th)*P, th)-th*th*sp.exp(2*r*th),
        'compact_MSE_derivative': sp.diff(M, th)-dM,
        'stationarity_H_transformation': dM.subs(th, x/(4*r))-rel,
        'stationary_optimum_value': stationary.subs(th, x/(4*r))-expected,
        'strict_increase_H_second_derivative': sp.diff(H,x,2)-((x+2)*sp.exp(x)+x/4*sp.exp(x/2)),
        'envelope_log_slope': s*sp.diff(sp.log(e*s/sp.sqrt(T*e*e+s*s)),s)-T*e*e/(T*e*e+s*s),
        'stationarity_RHS_log_derivative': s*sp.diff(Z*(1-D*s*s/(T*e*e)),s)+2*Z,
    }
    for name, residual in identities.items():
        check(name, sp.simplify(residual) == 0, 'symbolic', residual=str(sp.simplify(residual)))
    check('H_prime_at_zero', sp.diff(H,x).subs(x,0) == 0, 'symbolic')
    p, z = sp.symbols('p z', positive=True)
    f = z**(-p)+p*z
    check('modal_limit_stationary_at_one', sp.diff(f,z).subs(z,1) == 0, 'symbolic')
    check('modal_limit_strict_convexity', sp.simplify(sp.diff(f,z,2)-p*(p+1)*z**(-p-2)) == 0, 'symbolic')


def modal_checks() -> None:
    cases = {
        'single': ([1.2], [1.0], [0.8]),
        'repeated_slowest': ([0.8, 0.8, 1.7], [0.7, -0.4, 1.1], [0.2, 0.6, 0.9]),
        'unexcited_slower_noisy': ([0.3, 1.1, 2.4], [0.0, 1.0, -0.4], [0.9, 0.2, 0.5]),
        'disjoint_bias_and_noise': ([0.7, 1.6], [1.0, 0.0], [0.0, 1.0]),
    }
    rows = []
    for name, arrays in cases.items():
        for T in [0.5, 1.0, 2.3]:
            modes = Modes(*arrays, T)
            for ratio in [1.0, 1.5, 5.0, 20.0]:
                K = max(1/T, ratio/T)
                tc = T-1/K
                for sigma in [0.0, 1e-3, 0.2, 2.0]:
                    numerical = 0.0
                    for r, eta, shape in zip(modes.rho, modes.xi0, modes.noise_shape):
                        mean = eta*math.exp(-r*(math.log(K*T)+1))
                        # Independent quadrature of the terminal-kernel energy.
                        q1 = quad(lambda u: (math.exp(-r)*(K*(T-u))**(-r)/(T-u))**2, 0, tc, epsabs=1e-12, epsrel=1e-12)[0] if tc > 0 else 0.0
                        q2 = quad(lambda u: (K*math.exp(-r*K*(T-u)))**2, tc, T, epsabs=1e-12, epsrel=1e-12)[0]
                        numerical += mean*mean+(r*sigma*shape)**2*(q1+q2)
                    close(f'{name}_T{T}_K{ratio}_s{sigma}_exact_integral', modes.mse(K,sigma), numerical, 'modal_exact', rtol=1e-10, atol=1e-16)
                    a, _, B, b, _ = modes.coefficients()
                    bias = float(np.dot(a, K**(-2*modes.rho)))
                    variance = modes.mse(K,sigma)-bias
                    vlo = float(np.dot(modes.noise_shape**2,b))*sigma*sigma*K
                    vhi = float(np.dot(modes.noise_shape**2,B))*sigma*sigma*K
                    check(f'{name}_T{T}_K{ratio}_s{sigma}_variance_bracket', vlo-1e-12*max(1,vhi) <= variance <= vhi+1e-12*max(1,vhi), 'modal_bounds')
            sigmas = [1e-3,1e-6,1e-9,1e-12]
            trajectory = []
            for sigma in sigmas:
                L, Mlead, rs = modes.leading(sigma)
                K, Mmin = modes.optimum(sigma)
                row = dict(case=name, T=T, sigma=sigma, rho_star=rs, K_opt=K, K_leading=L, K_ratio=K/L, MSE_opt=Mmin, MSE_leading=Mlead, MSE_ratio=Mmin/Mlead)
                rows.append(row);trajectory.append(row)
                check(f'{name}_T{T}_s{sigma}_cap_constraint', K>=1/T, 'modal_optimization')
                close(f'{name}_T{T}_s{sigma}_stationarity', modes.derivative(K,sigma)/(sigma*sigma), 0.0, 'modal_optimization', atol=2e-10)
                check(f'{name}_T{T}_s{sigma}_nearby_MSE', all(modes.mse(K*q,sigma)>=Mmin*(1-1e-12) for q in [0.9,0.99,1.01,1.1] if K*q>=1/T), 'modal_optimization')
            last, previous = trajectory[-1],trajectory[-2]
            close(f'{name}_T{T}_K_equivalent', last['K_ratio'], 1.0, 'modal_asymptotic', rtol=2e-6)
            close(f'{name}_T{T}_MSE_equivalent', last['MSE_ratio'], 1.0, 'modal_asymptotic', rtol=2e-6)
            slopeK = math.log(last['K_opt']/previous['K_opt'])/math.log(last['sigma']/previous['sigma'])
            slopeR = 0.5*math.log(last['MSE_opt']/previous['MSE_opt'])/math.log(last['sigma']/previous['sigma'])
            close(f'{name}_T{T}_K_exponent', slopeK, -2/(2*last['rho_star']+1), 'modal_asymptotic', atol=2e-6)
            close(f'{name}_T{T}_RMS_exponent', slopeR, 2*last['rho_star']/(2*last['rho_star']+1), 'modal_asymptotic', atol=2e-6)
    write_csv('modal_asymptotics.csv',rows)
    # Grouping repeated rates must leave the whole finite-cap objective intact.
    repeated = Modes([0.8,0.8],[0.7,-0.4],[0.2,0.6],2.0)
    merged = Modes([0.8],[math.hypot(0.7,0.4)],[math.hypot(0.2,0.6)],2.0)
    for K in [0.5,1,10,100]:
        close(f'repeated_mode_grouping_K{K}',repeated.mse(K,0.2),merged.mse(K,0.2),'modal_degeneracy')
    for T in [0.5,2.0]:
        zero = Modes([0.4,1.2],[0,0],[0.3,1.0],T)
        for sigma in [1e-6,0.2,2.0]:
            K,_ = zero.optimum(sigma)
            close(f'zero_initial_endpoint_T{T}_s{sigma}',K,1/T,'modal_degeneracy',rtol=0,atol=0)
            check(f'zero_initial_increasing_T{T}_s{sigma}',all(zero.derivative(q/T,sigma)>0 for q in [1,2,20,100]),'modal_degeneracy')
        noiseless = Modes([0.4,1.2],[1,-1],[0,0],T)
        check(f'noiseless_no_finite_minimizer_T{T}',math.isinf(noiseless.optimum(1)[0]),'modal_degeneracy')
        check(f'noiseless_decreasing_T{T}',all(noiseless.derivative(q/T,1)<0 for q in [1,2,20,100]),'modal_degeneracy')
        both = Modes([0.4,1.2],[0,0],[0,0],T)
        check(f'both_zero_T{T}',all(both.mse(q/T,1)==0 for q in [1,2,20]),'modal_degeneracy')
        tiny = Modes([1.2],[1e-4],[1],T)
        close(f'active_cap_large_noise_T{T}',tiny.optimum(2)[0],1/T,'modal_degeneracy',rtol=0,atol=0)
    # Orthogonality, not vanishing covariance, is what makes the MSE additive.
    rng=np.random.default_rng(210906)
    for index in range(10):
        A=rng.normal(size=(4,4));cov=A@A.T;mu=rng.normal(size=4)
        Q,_=np.linalg.qr(rng.normal(size=(4,4)))
        close(f'correlated_orthogonal_norm_{index}',float(np.trace(Q@cov@Q.T)+np.dot(Q@mu,Q@mu)),float(np.trace(cov)+mu@mu),'modal_covariance')
    invalid=[lambda:Modes([0],[1],[1],1),lambda:Modes([1,2],[1],[1],1),lambda:Modes([1],[1],[-1],1),lambda:Modes([1],[1],[1],0),lambda:Modes([1],[1],[1],1).mse(0.1,1),lambda:Modes([1],[1],[1],1).mse(1,-1)]
    for i,call in enumerate(invalid):
        try: call()
        except ValueError: check(f'invalid_input_{i}',True,'input_validation')
        else: check(f'invalid_input_{i}',False,'input_validation')


def hp_optimum(rho: mp.mpf, T: mp.mpf, eta: mp.mpf, sigma: mp.mpf) -> tuple[mp.mpf,mp.mpf,mp.mpf,mp.mpf]:
    D=(2*rho*rho-2*rho+1)/(4*rho)
    Z=16*rho*T*eta*eta*mp.exp(2*rho)/(sigma*sigma)
    W=mp.lambertw(Z)
    target=mp.log(Z*(1-D*sigma*sigma/(T*eta*eta)))
    f=lambda x: x+mp.log(x)+mp.log1p((1-4/x)*mp.exp(-x/2))-target
    x=mp.findroot(f,(W,W-mp.mpf('0.01')),tol=mp.mpf('1e-85'))
    K=x*x/(16*rho*rho*T)
    M=sigma*sigma/(32*rho*T)*(x*x+2*x*(1-mp.exp(-x/2)))
    return x,W,K,M


def lambert_checks() -> None:
    mp.mp.dps=100
    rows=[]
    for rf,Tf,ef in [('0.35','0.4','0.2'),('0.8','2.3','-1.5'),('1.2','1','1'),('2.8','2','0.7')]:
        rho,T,eta=map(mp.mpf,[rf,Tf,ef])
        scaled_errors=[]
        for sf in ['1e-3','1e-6','1e-12','1e-24']:
            sigma=mp.mpf(sf)
            x,W,K,M=hp_optimum(rho,T,eta,sigma)
            C=16*rho*T*eta*eta*mp.exp(2*rho);Z=C/sigma**2
            D=(2*rho*rho-2*rho+1)/(4*rho)
            H=x*mp.exp(x)+(x-4)*mp.exp(x/2)
            Hp=(x+1)*mp.exp(x)+(x/2-1)*mp.exp(x/2)
            F=x*x+2*x*(1-mp.exp(-x/2));Fp=2*x+2+(x-2)*mp.exp(-x/2)
            slopeR=1-Fp/F*Z/Hp
            slopeEnv=T*eta*eta/(T*eta*eta+sigma*sigma)
            slopeG=slopeR-slopeEnv
            direct=sigma**2/T*(rho*(x/(4*rho))**2/2+(1/(4*rho)-x/(8*rho))*mp.exp(-x/2))+(eta*eta-D*sigma*sigma/T)*mp.exp(-x+2*rho)
            relativeCap=K/(W*W/(16*rho*rho*T))-1
            scaledCap=abs(relativeCap)*mp.sqrt(W)/sigma*mp.sqrt(C)
            tag=f'r{rf}_T{Tf}_e{ef}_s{sf}'
            check(tag+'_stationarity',abs(H/(Z*(1-D*sigma*sigma/(T*eta*eta)))-1)<mp.mpf('1e-75'),'lambert_exact')
            check(tag+'_stationary_MSE',abs(M/direct-1)<mp.mpf('1e-75'),'lambert_exact')
            check(tag+'_Lambert_conversion',abs(mp.exp(-W/2)/(sigma*mp.sqrt(W)/mp.sqrt(C))-1)<mp.mpf('1e-75'),'lambert_exact')
            check(tag+'_relative_cap_remainder',scaledCap<mp.mpf('2.1'),'lambert_remainder',normalized_remainder=float(scaledCap))
            check(tag+'_large_W_signed_cap_error',relativeCap<0 and W>4,'lambert_remainder')
            check(tag+'_RMS_slope_remainder',abs((slopeR-(1-2/(1+W)))*W**2)<3,'slope')
            check(tag+'_ratio_slope_remainder',abs((slopeG+2/(1+W))*W**2)<3,'slope')
            check(tag+'_RMS_positive_ratio_negative',slopeR>0 and slopeG<0,'slope')
            h=mp.mpf('1e-5')
            plus=hp_optimum(rho,T,eta,sigma*mp.exp(h))[3]
            minus=hp_optimum(rho,T,eta,sigma*mp.exp(-h))[3]
            finiteSlope=(mp.log(plus)-mp.log(minus))/(4*h)
            check(tag+'_exact_slope_vs_finite_difference',abs(finiteSlope-slopeR)<mp.mpf('1e-10'),'slope')
            scaled_errors.append(float(mp.exp(-W/2)/sigma))
            rows.append(dict(rho=float(rho),T=float(T),eta0=float(eta),sigma=sf,W=mp.nstr(W,25),x=mp.nstr(x,25),K_opt=mp.nstr(K,25),relative_cap_error=mp.nstr(relativeCap,25),normalized_cap_remainder=mp.nstr(scaledCap,25),RMS_log_slope=mp.nstr(slopeR,25),ratio_log_slope=mp.nstr(slopeG,25)))
        check(f'r{rf}_exp_over_sigma_increases',all(b>a for a,b in zip(scaled_errors,scaled_errors[1:])),'lambert_remainder')
    write_csv('modal_lambert_remainder_and_slopes.csv',rows)


def bounded_noise_and_gate_checks() -> None:
    rho,T,nbar=1.2,2.0,0.3
    rows=[]
    for power in [1,2]:
        for eta in [-1.0,0.0,0.1,0.3,1.0]:
            errors=[]
            for ratio in [1,2,10,100]:
                K=ratio/T
                theta=(K*T)**(1/power);tc=T-T/theta
                J=math.log(K*T)+1 if power==1 else 2*theta-1
                alpha=math.exp(-rho*J)
                if power==1:
                    transition=lambda u: math.exp(-rho)*(K*(T-u))**(-rho)
                    gain=lambda u: 1/(T-u)
                else:
                    transition=lambda u: math.exp(-rho*(2*theta-T/(T-u)))
                    gain=lambda u: T/(T-u)**2
                mass=rho*(quad(lambda u:transition(u)*gain(u),0,tc,epsabs=1e-12,epsrel=1e-12)[0] if tc>0 else 0)
                mass+=rho*quad(lambda u:math.exp(-rho*K*(T-u))*K,tc,T,epsabs=1e-12,epsrel=1e-12)[0]
                aligned=-nbar*(1 if eta>=0 else -1)
                actual=abs(alpha*eta-aligned*mass)
                expected=alpha*abs(eta)+(1-alpha)*nbar
                close(f'p{power}_eta{eta}_K{ratio}_aligned_supremum',actual,expected,'bounded_noise')
                close(f'p{power}_eta{eta}_K{ratio}_kernel_mass',mass,1-alpha,'bounded_noise')
                close(f'p{power}_eta{eta}_K{ratio}_offset_identity',expected-nbar,alpha*(abs(eta)-nbar),'bounded_noise')
                # Equality is subject only to floating-point rounding in the convex sum.
                correct_side = abs(expected-nbar)<=1e-15 if abs(eta)==nbar else (expected>=nbar-1e-15 if abs(eta)>nbar else expected<=nbar+1e-15)
                check(f'p{power}_eta{eta}_K{ratio}_approach_side',correct_side,'bounded_noise')
                errors.append(expected)
                rows.append(dict(profile=power,eta0=eta,K=K,alpha=alpha,worst_case=expected,nbar=nbar))
            check(f'p{power}_eta{eta}_monotonicity',all((b<=a+1e-14 if abs(eta)>=nbar else b>=a-1e-14) for a,b in zip(errors,errors[1:])),'bounded_noise')
    alpha_zero_gain = math.exp(-rho*0.0)
    error_zero_gain = alpha_zero_gain*abs(0.0)+(1-alpha_zero_gain)*nbar
    check('zero_initial_zero_gain_counterexample',error_zero_gain==0.0 and error_zero_gain<nbar,'bounded_noise')
    # Ordering uses fixed tube constants. It saturates at q_* exp(-b_* T),
    # not at q_*/2, and includes the L_h=0 infinite-branch convention.
    qstar,b=0.07,0.49
    normal=qstar*math.exp(-b*T)
    tangential=0.01
    gates=[min(normal,tangential/I) for I in [2,1,0.5,0.1,0.01]]
    check('gate_enlargement_under_decreasing_I',all(v>=u for u,v in zip(gates,gates[1:])),'gate')
    close('gate_saturation_normal_branch',gates[-1],normal,'gate',rtol=0,atol=0)
    check('normal_branch_is_not_half_radius',abs(normal-qstar/2)>1e-3,'gate')
    close('zero_Lh_gate',min(normal,math.inf),normal,'gate',rtol=0,atol=0)
    write_csv('modal_bounded_noise_limits.csv',rows)


def main() -> int:
    symbolic_checks(); modal_checks(); lambert_checks(); bounded_noise_and_gate_checks()
    failed=[c for c in CHECKS if not c['passed']]
    groups={k:dict(total=v,passed=sum(c['passed'] for c in CHECKS if c['group']==k)) for k,v in Counter(c['group'] for c in CHECKS).items()}
    result=dict(experiment='Modal',status='FAIL' if failed else 'PASS',checks_passed=len(CHECKS)-len(failed),checks_total=len(CHECKS),groups=groups,checks=CHECKS,scope='Deterministic exact/asymptotic regressions. No nonlinear Monte Carlo rerun. Finite tests do not replace the analytical proofs.')
    (ROOT/'modal_mathematical_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    text=['Modal MATHEMATICAL VALIDATION',f"Status: {result['status']}",f"Checks: {result['checks_passed']}/{result['checks_total']}",*[f"{k}: {v['passed']}/{v['total']}" for k,v in groups.items()],'',result['scope']]
    text.extend('FAIL: '+str(c) for c in failed)
    (ROOT/'modal_mathematical_validation.txt').write_text('\n'.join(text)+'\n')
    print('\n'.join(text))
    return 1 if failed else 0


if __name__=='__main__':
    raise SystemExit(main())
