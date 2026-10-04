#!/usr/bin/env python3
"""Algebraic and high-precision checks of actuator realizations and profile gaps.
No stochastic paths are generated."""
from __future__ import annotations
import csv
import json
from collections import Counter
from pathlib import Path
import mpmath as mp
import numpy as np
import sympy as sp

from artifact_paths import work_dir, output_dir, code_dir
ROOT = work_dir()
CHECKS: list[dict] = []

def check(name: str, passed: bool, group: str, **evidence) -> None:
    CHECKS.append(dict(name=name, passed=bool(passed), group=group, **evidence))

def near(name: str, actual, expected, group: str, tolerance: float=2e-11) -> None:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    error = float(np.linalg.norm(a-b)/max(1.0, float(np.linalg.norm(b))))
    check(name, np.isfinite(error) and error <= tolerance, group,
          normalized_error=error, tolerance=tolerance)

def write_csv(name: str, rows: list[dict]) -> None:
    with (ROOT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def weighted_realization() -> None:
    rng = np.random.default_rng(20260906)
    for case in range(80):
        tangent = case % 3
        normal = 1 + case % 3
        m = tangent + normal + 2
        Q, _ = np.linalg.qr(rng.normal(size=(m,m)))
        V, U = Q[:, :tangent], Q[:, tangent:]
        Bpar = np.diag(np.linspace(1,2,tangent)) @ V.T
        C = rng.normal(size=(normal,U.shape[1]))
        while np.linalg.cond(C@C.T) > 100:
            C = rng.normal(size=C.shape)
        # Allow normal actuation also along the complementary tangential inputs.
        Bperp = C @ U.T + rng.normal(size=(normal,tangent)) @ V.T
        Av, An = rng.normal(size=(U.shape[1],U.shape[1])), rng.normal(size=(normal,normal))
        Mv, Mn = Av.T@Av+np.eye(U.shape[1]), An.T@An+np.eye(normal)
        Cstar = np.linalg.solve(Mv,C.T@Mn)
        H = U @ Cstar @ np.linalg.solve(C@Cstar,np.eye(normal))
        name=f'weighted_case_{case}'
        near(name+'_tangent_identity', Bpar@H, np.zeros((tangent,normal)), 'weighted_realization')
        near(name+'_normal_identity', Bperp@H, np.eye(normal), 'weighted_realization')
        near(name+'_range_in_kernel', (np.eye(m)-U@U.T)@H, np.zeros((m,normal)), 'weighted_realization')
        eta = rng.normal(size=normal); A = rng.normal(size=(normal,normal)); R0=A.T@A+np.eye(normal)
        baseline=rng.normal(size=m); targetpar=rng.normal(size=tangent); targetnormal=rng.normal(size=normal)
        fpar=targetpar-Bpar@baseline; fperp=targetnormal-Bperp@baseline
        kappa=float(np.exp(rng.uniform(-2,3)))
        u=baseline-kappa*H@R0@eta
        near(name+'_baseline_tangent_preserved', fpar+Bpar@u, targetpar, 'weighted_realization')
        near(name+'_scheduled_normal_exact', fperp+Bperp@u, targetnormal-kappa*R0@eta, 'weighted_realization')
        if tangent:
            check(name+'_not_all_input_directions_vertical', np.linalg.norm(Bpar)>0, 'logical_scope')
    # Rank/surjectivity alone do not solve a prescribed baseline problem.
    Bpar=np.zeros((1,1)); Bperp=np.ones((1,1)); fpar=1.; desiredpar=0.
    check('surjective_vertical_block_can_have_incompatible_baseline',
          np.linalg.matrix_rank(Bperp)==1 and fpar!=desiredpar and np.all(Bpar==0),'logical_scope')
    check('transparent_input_need_not_be_normally_surjective',
          np.all(np.zeros((1,2))==0) and np.linalg.matrix_rank(np.array([[1.,0.],[0.,0.]]))<2,'logical_scope')
    for n in range(1,6):
        G=rng.normal(size=(n,n))+3*np.eye(n)
        H=G.T@np.linalg.inv(G@G.T)
        near(f'square_inverse_{n}',H,np.linalg.inv(G),'weighted_realization')

def graph_bridge() -> None:
    rng=np.random.default_rng(610926)
    rows=[]
    for N in [2,3,6,9]:
        for q in [1,2,3]:
            for graph in ['path','ring','complete']:
                A=np.zeros((N,N))
                edges={(i,i+1) for i in range(N-1)}
                if graph=='ring':edges.add((0,N-1))
                if graph=='complete':edges={(i,j) for i in range(N) for j in range(i+1,N)}
                for i,j in edges:A[i,j]=A[j,i]=float(rng.uniform(.5,1.5))
                L=np.diag(A.sum(axis=1))-A
                Pi=np.eye(N)-np.ones((N,N))/N
                Bpar=np.kron(np.ones((1,N))/N,np.eye(q)); Bperp=np.kron(Pi,np.eye(q))
                lam,V=np.linalg.eigh(L); U=np.kron(V[:,1:],np.eye(q))
                k0=.7; R0=k0*U.T@np.kron(L,np.eye(q))@U
                name=f'{graph}_N{N}_q{q}'
                near(name+'_zero_mean_kernel',Bpar@U,np.zeros((q,(N-1)*q)),'network_bridge')
                near(name+'_normal_inclusion',Bperp@U,U,'network_bridge')
                near(name+'_identity_in_normal_coordinates',U.T@Bperp@U,np.eye((N-1)*q),'network_bridge')
                near(name+'_spectral_transverse_rate',np.linalg.eigvalsh(R0)[0],k0*lam[1],'network_bridge')
                y=rng.normal(size=q); xi=rng.normal(size=(N-1)*q); eta=U@xi
                X=np.tile(y,N)+eta; kappa=2.3
                u=-kappa*U@R0@xi
                near(name+'_feedback_matches_diffusion',u,-k0*kappa*np.kron(L,np.eye(q))@X,'network_bridge')
                near(name+'_feedback_tangent_zero',Bpar@u,np.zeros(q),'network_bridge')
                check(name+'_full_identity_has_tangent_channels',np.linalg.norm(Bpar)>0,'network_bridge')
                D=rng.normal(size=(q,q)); b=.8
                def f(x):return D@x-b*(x@x)*x
                def df(x):return D-b*((x@x)*np.eye(q)+2*np.outer(x,x))
                F=np.concatenate([f(x) for x in X.reshape(N,q)])
                h=np.mean([f(y+e)-f(y) for e in eta.reshape(N,q)],axis=0)
                Aeta=np.kron(np.eye(N),df(y))@eta
                remainder=Bperp@(F-np.tile(f(y),N)-Aeta)
                near(name+'_zero_baseline_base',Bpar@F,f(y)+h,'network_bridge')
                near(name+'_zero_baseline_normal',Bperp@F,Aeta+remainder,'network_bridge')
                # Cubic Taylor remainder: -b(2<y,e>e+||e||^2 y+||e||^2 e).
                exact=np.concatenate([-b*(2*(y@e)*e+(e@e)*y+(e@e)*e) for e in eta.reshape(N,q)])
                near(name+'_Taylor_remainder',remainder,Bperp@exact,'network_bridge')
                unit=eta/np.linalg.norm(eta)
                ratios=[]
                for eps in [1e-1,1e-2,1e-3]:
                    er=unit.reshape(N,q)*eps
                    rem=np.concatenate([-b*(2*(y@e)*e+(e@e)*y+(e@e)*e) for e in er])
                    ratio=float(np.linalg.norm(Bperp@rem)/eps**2)
                    bound=b*(3*np.linalg.norm(y)+eps)
                    check(name+f'_quadratic_remainder_{eps}',ratio<=bound*(1+1e-12),'network_Taylor',ratio=ratio,bound=float(bound))
                    ratios.append(ratio)
                rows.append(dict(graph=graph,N=N,q=q,rho=k0*lam[1],normal_dimension=U.shape[1],remainder_over_eps2_at_1e3=ratios[-1]))
    write_csv('realization_network_bridge.csv',rows)

def symbolic_gaps() -> None:
    r,T,e,s,alpha,t = sp.symbols('rho T eta sigma alpha t',positive=True)
    cE=sp.symbols('cE',positive=True)
    a=e*sp.exp(-r)*T**(-r)
    derived=cE*a**(1/(2*r+1))*s**(2*r/(2*r+1))*sp.sqrt(T)/s
    stated=cE*(e*sp.exp(-r)*sp.sqrt(T)/s)**(1/(2*r+1))
    # Compare exponents and log constants with positive symbols.
    check('inverse_time_ratio_constant',sp.simplify(sp.expand_log(sp.log(derived/stated),force=True))==0,'symbolic_gaps')
    phi=alpha+(1-alpha)*t/T
    gain=(1-alpha)/(r*(alpha*T+(1-alpha)*t))
    check('affine_transition_differential_identity',sp.simplify(sp.diff(phi,t)-r*gain*phi)==0,'symbolic_gaps')
    check('affine_gain_derivative',sp.simplify(sp.diff(gain,t)+(1-alpha)**2/(r*(alpha*T+(1-alpha)*t)**2))==0,'symbolic_gaps')
    objective=e**2*alpha**2+s**2/T*(1-alpha)**2
    optimum=s**2/(T*e**2+s**2)
    check('bounded_envelope_stationarity',sp.simplify(sp.diff(objective,alpha).subs(alpha,optimum))==0,'symbolic_gaps')
    check('bounded_envelope_value',sp.simplify(objective.subs(alpha,optimum)-(e*s)**2/(T*e**2+s**2))==0,'symbolic_gaps')
    primitive=sp.log(alpha*T+(1-alpha)*t)/r
    check('finite_gain_mass_primitive',sp.simplify(sp.diff(primitive,t)-gain)==0,'symbolic_gaps')

def exact_optimized(rho: mp.mpf,T: mp.mpf,eta: mp.mpf,sigma: mp.mpf):
    if rho<=0 or T<=0 or eta==0 or sigma<=0:
        raise ValueError('The nondegenerate corollary requires rho,T,sigma>0 and eta!=0.')
    C=rho*rho*mp.exp(-2*rho)/(2*rho+1)
    B=rho/2*(1-mp.exp(-2*rho))+C
    delta=eta*eta*mp.exp(-2*rho)*T**(-2*rho)-sigma*sigma*C*T**(-(2*rho+1))
    K1=max(1/T,(2*rho*max(delta,0)/(B*sigma*sigma))**(1/(2*rho+1)))
    M1=delta*K1**(-2*rho)+B*sigma*sigma*K1
    D=(2*rho*rho-2*rho+1)/(4*rho)
    Z=16*rho*T*eta*eta*mp.exp(2*rho)/(sigma*sigma)
    W=mp.lambertw(Z)
    target=Z*(1-D*sigma*sigma/(T*eta*eta))
    H=lambda x:x*mp.exp(x)+(x-4)*mp.exp(x/2)
    lo=4*rho
    if H(lo)>=target:x=lo
    else:
        hi=max(2*lo,2*W)
        while H(hi)<target:hi*=2
        for _ in range(300):
            mid=(lo+hi)/2
            if H(mid)<target:lo=mid
            else:hi=mid
        x=(lo+hi)/2
    th=x/(4*rho);K2=x*x/(16*rho*rho*T)
    M2=sigma*sigma/T*(rho*th*th/2+(1/(4*rho)-th/2)*mp.exp(-2*rho*th))+(eta*eta-D*sigma*sigma/T)*mp.exp(-4*rho*th+2*rho)
    env=abs(eta)*sigma/mp.sqrt(T*eta*eta+sigma*sigma)
    cE=mp.sqrt(2*rho+1)*(B/(2*rho))**(rho/(2*rho+1))
    lead1=cE*(abs(eta)*mp.exp(-rho)*mp.sqrt(T)/sigma)**(1/(2*rho+1))
    return K1,K2,mp.sqrt(M1)/env,mp.sqrt(M2)/env,lead1,W,x

def high_precision_gaps() -> None:
    mp.mp.dps=90
    cases=[('0.3','0.7','-1.5'),('0.8','2','0.6'),('1.2','1','1'),('2.4','3','2')]
    rows=[]
    for j,p in enumerate(cases):
        rho,T,eta=map(mp.mpf,p)
        for ss in ['1e-3','1e-6','1e-12','1e-24']:
            sigma=mp.mpf(ss)
            K1,K2,gap1,gap2,lead1,W,x=exact_optimized(rho,T,eta,sigma)
            scaled2=W*W*(gap2/(W/(4*mp.sqrt(2*rho)))-1-1/W)
            name=f'case{j}_sigma{ss}'
            check(name+'_feasible_caps',K1>=1/T and K2>=1/T,'high_precision_gaps')
            check(name+'_both_above_envelope',gap1>=1 and gap2>=1,'high_precision_gaps')
            check(name+'_inverse_time_relative_expansion',abs(gap1/lead1-1)<mp.mpf('3e-5'),'high_precision_gaps',relative_error=str(gap1/lead1-1))
            check(name+'_inverse_square_second_order_remainder',abs(scaled2)<2,'high_precision_gaps',scaled_remainder=str(scaled2))
            if sigma==mp.mpf('1e-24'):
                check(name+'_inverse_time_leading_constant_limit',abs(gap1/lead1-1)<mp.mpf('1e-44'),'high_precision_gaps')
            eps=mp.mpf('1e-5')
            gp=exact_optimized(rho,T,eta,sigma*mp.exp(eps))[3]
            gm=exact_optimized(rho,T,eta,sigma*mp.exp(-eps))[3]
            slope=(mp.log(gp)-mp.log(gm))/(2*eps)
            check(name+'_ratio_not_RMS_slope',abs((slope+2/(1+W))*W*W)<3,'high_precision_gaps',slope=str(slope))
            alpha=sigma*sigma/(T*eta*eta+sigma*sigma)
            peak=eta*eta/(rho*sigma*sigma)
            mass=-mp.log(alpha)/rho
            check(name+'_envelope_finite_mass_nonzero_multiplier',0<alpha<1 and mp.isfinite(mass) and mass>0,'gain_placement')
            check(name+'_peak_not_equal_to_named_optima',peak>K1 and peak>K2,'gain_placement')
            rows.append(dict(case=j,rho=str(rho),T=str(T),eta0=str(eta),sigma=ss,K1=str(K1),K2=str(K2),gap1=str(gap1),gap1_relative_error=str(gap1/lead1-1),gap2=str(gap2),W=str(W),gap2_scaled_remainder=str(scaled2),gap2_log_slope=str(slope),envelope_alpha=str(alpha),envelope_mass=str(mass),envelope_peak=str(peak)))
        # Explicit large-noise endpoint cases are not put through small-noise asymptotics.
        K1,K2,*_=exact_optimized(rho,T,eta,mp.mpf('1e4'))
        check(f'case{j}_large_noise_cap_endpoints',K1==1/T and abs(K2-1/T)<mp.mpf('1e-80'),'limiting_scope')
    for values in [(0,1,1,1),(1,0,1,1),(1,1,0,1),(1,1,1,0)]:
        try:exact_optimized(*map(mp.mpf,values))
        except ValueError:check('reject_degenerate_'+str(values),True,'limiting_scope')
        else:check('reject_degenerate_'+str(values),False,'limiting_scope')
    write_csv('realization_named_profile_gap_checks.csv',rows)

def main() -> int:
    weighted_realization();graph_bridge();symbolic_gaps();high_precision_gaps()
    failed=[x for x in CHECKS if not x['passed']]
    groups={g:dict(passed=sum(x['passed'] for x in CHECKS if x['group']==g),total=sum(x['group']==g for x in CHECKS)) for g in sorted({x['group'] for x in CHECKS})}
    out=dict(status='FAIL' if failed else 'PASS',checks_total=len(CHECKS),checks_passed=len(CHECKS)-len(failed),precision_digits=90,groups=groups,checks=CHECKS)
    (ROOT/'realization_realization_gap_validation.json').write_text(json.dumps(out,indent=2)+'\n')
    text=f"Status: {out['status']}\nChecks: {out['checks_passed']}/{out['checks_total']}\n"+'\n'.join(f'{g}: {v["passed"]}/{v["total"]}' for g,v in groups.items())+'\n'
    if failed:text+='\nFailures:\n'+'\n'.join(str(x) for x in failed)+'\n'
    (ROOT/'realization_realization_gap_validation.txt').write_text(text);print(text)
    return bool(failed)
if __name__=='__main__':raise SystemExit(main())
