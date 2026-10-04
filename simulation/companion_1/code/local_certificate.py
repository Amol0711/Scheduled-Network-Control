#!/usr/bin/env python3
"""Local annular certificate and deterministic trajectory consistency checks.

Exact decimal parameters are evaluated at 80 digits. Sampled trajectory
inequalities are diagnostics and do not establish between-sample guarantees."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad, solve_ivp
import sympy as sy

import network_experiments as sl

from artifact_paths import work_dir, output_dir, code_dir
ROOT = work_dir()
FloatArray = NDArray[np.float64]
N, DIM = 6, 2
T, RHO, GAIN = 2.0, 1.2, 1.2
R0_LO, R0_HI, R1_LO, R1_HI, R2 = 0.87, 0.90, 0.85, 1.002, 1.15
Q_STAR, Q0 = 0.07, 0.01
SEED = 20260906
REGULARITY, CLOCK_ORDER = 3, 4
STOP_GAP = 1.0e-5
TAU_END = math.log(T / STOP_GAP)
V = sl.V_RING[:, 1:]
LAMBDA = np.array([1.0, 1.0, 3.0, 3.0, 4.0])


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def radial(t: float, radius: float) -> float:
    den = 1.0 + radius * radius * math.expm1(2.0*t)
    if den <= 0.0:
        raise ValueError('The backward reduced flow leaves its existence domain.')
    return radius*math.exp(t)/math.sqrt(den)


def jacobian(y: FloatArray) -> FloatArray:
    return (1.0-float(y@y))*np.eye(2) + 2.0*sl.J - 2.0*np.outer(y, y)


def reduced_flow(y: FloatArray, t: float) -> FloatArray:
    angle = 2.0*t
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    den = 1.0+float(y@y)*math.expm1(2.0*t)
    if den <= 0.0:
        raise ValueError('Backward-flow denominator is nonpositive.')
    return math.exp(t)/math.sqrt(den) * (rotation@y)


def flow_jacobian(y: FloatArray, t: float) -> FloatArray:
    angle = 2.0*t
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    e = math.expm1(2.0*t)
    den = 1.0+float(y@y)*e
    if den <= 0.0:
        raise ValueError('Backward-flow denominator is nonpositive.')
    return math.exp(t)/math.sqrt(den)*rotation@(np.eye(2)-e/den*np.outer(y,y))


def nonlinear_terms(y: FloatArray, eta: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Cancellation-free exact quadratic/cubic Taylor remainder."""
    sq = np.sum(eta*eta, axis=1)
    rem = -(2.0*(eta@y)[:,None]*eta + sq[:,None]*y + sq[:,None]*eta)
    return rem.mean(axis=0), rem - rem.mean(axis=0)


def scalar_certificate() -> dict[str, Any]:
    with mp.workdps(80):
        t, rho = mp.mpf('2'), mp.mpf('1.2')
        a,b,c,d,e,q = [mp.mpf(x) for x in ['0.87','0.90','0.85','1.002','1.15','0.07']]
        flow=lambda tt,rr:rr*mp.exp(tt)/mp.sqrt(1+rr**2*mp.expm1(2*tt))
        beta=1-c*c; cr=3*d+q; l2=cr/6; lh=l2*q; bs=beta+cr*q
        margin_inner=a-c; margin_outer=d-flow(t,b); margin=min(margin_inner,margin_outer)
        ig=t/(rho+1); i2=t/(2*rho+1)
        gate_n=q*mp.exp(-bs*t)
        gate_b=margin/(lh*ig)*mp.exp(-(bs+beta)*t)
        vals={
            'r0_inner':a,'r0_outer':b,'r1_inner':c,'r1_outer':d,'r2':e,
            'q_star':q,'rho':rho,'beta':beta,'mu_g':beta,'C_g':mp.mpf(1),
            'C_g_minus':mp.mpf(1),'mu_g_minus':3*e*e-1,
            'c_r':cr,'L_2':l2,'L_h':lh,'gamma':cr,'b_star':bs,
            'reduced_outer_terminal_radius':flow(t,b),
            'inner_base_margin':margin_inner,'outer_base_margin':margin_outer,
            'd_star':margin,'I_1':ig,'I_2':i2,
            'gate_normal':gate_n,'gate_base':gate_b,'epsilon_star':min(gate_n,gate_b),
            'backward_radius_max':flow(-t,d),'backward_radius_min':flow(-t,c),
            'phase_collar_margin':e-flow(-t,d),'backward_critical_radius':1/mp.sqrt(1-mp.exp(-2*t)),
            'order_r_gap':rho-mp.mpf(REGULARITY)/CLOCK_ORDER,
            'normal_exit_upper_bound':mp.mpf('0.01')*mp.exp(bs*t),
            'base_exit_upper_bound':lh*mp.mpf('0.01')*ig*mp.exp((bs+beta)*t),
            'quadratic_shadow_uniform_bound':l2*mp.mpf('0.01')**2*i2*mp.exp((2*bs+beta)*t),
            'terminal_q_bound':mp.mpf('0.01')*mp.exp(bs*(t-mp.mpf('0.00001')))*(mp.mpf('0.00001')/t)**rho,
        }
        return {'parameters_exact_decimals':{'T':'2','rho':'1.2','q0':'0.01','q_star':'0.07'},
                'metric_convention':'Local ambient product metric ||dX||_*^2=||dy||_2^2+||deta||_2^2. Normal norm equals Euclidean disagreement; base norm equals Euclidean mean norm.',
                'working_precision_digits':80,
                'values':{k:float(v) for k,v in vals.items()},
                'values_70_digits':{k:mp.nstr(v,70) for k,v in vals.items()},
                'note':'Exact formulas certify the hypotheses; floating-point sample checks do not replace those proofs.'}


def scaled_rhs(tau: float, state: FloatArray) -> FloatArray:
    """Mean plus e^(rho*tau)-scaled orthogonal disagreement coordinates."""
    y=state[:2]; zm=state[2:].reshape(N-1,DIM); z=V@zm
    epsilon=math.exp(-RHO*tau); dt=T*math.exp(-tau)
    zsq=np.sum(z*z,axis=1)
    # This is exp(rho*tau) times the node Taylor remainder, without subtraction.
    scaled_rem=-(epsilon*(2.0*(z@y)[:,None]*z+zsq[:,None]*y)
                 +epsilon**2*zsq[:,None]*z)
    h=epsilon*scaled_rem.mean(axis=0)
    dy=dt*(sl.node_field(y[None,:])[0]+h)
    dz=-(GAIN*LAMBDA-RHO)[:,None]*zm+dt*(zm@jacobian(y).T+V.T@scaled_rem)
    return np.concatenate([dy,dz.ravel()])


def integrate_scaled(y0: FloatArray, eta0: FloatArray, tau_grid: FloatArray,
                     rtol: float, atol: float, max_step: float) -> dict[str,Any]:
    start=np.concatenate([y0,(V.T@eta0).ravel()])
    sol=solve_ivp(scaled_rhs,(0.0,TAU_END),start,method='DOP853',
                  rtol=rtol,atol=atol,max_step=max_step,t_eval=tau_grid)
    if not sol.success:
        raise RuntimeError(sol.message)
    y=sol.y[:2].T; zm=sol.y[2:].T.reshape(-1,N-1,DIM)
    eta=np.einsum('ij,tjk->tik',V,zm)*np.exp(-RHO*tau_grid)[:,None,None]
    q=np.linalg.norm(zm.reshape(len(tau_grid),-1),axis=1)*np.exp(-RHO*tau_grid)
    return {'y':y,'eta':eta,'q':q,'rtol':rtol,'atol':atol,'max_step_tau':max_step,
            'nfev':sol.nfev,'method':'DOP853','coordinates':'scaled orthogonal disagreement'}


def write_csv(path: Path, rows: list[dict[str,Any]]) -> None:
    if not rows:
        raise ValueError('Cannot write an empty CSV.')
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(output: Path) -> int:
    output.mkdir(parents=True,exist_ok=True)
    checks: list[dict[str,Any]]=[]
    def check(name: str, condition: Any, **evidence: Any) -> None:
        checks.append({'name':name,'passed':bool(condition),**evidence})
    cert=scalar_certificate();c=cert['values']
    check('reference application parameters', (sl.N,sl.Q,sl.ALPHA,sl.B_SL,sl.OMEGA,sl.T,sl.K0)==(6,2,1.0,1.0,2.0,2.0,1.2))
    check('ring orthogonal eigenbasis',np.linalg.norm(sl.L_RING@V-V*LAMBDA)<1e-13)
    check('orthogonal disagreement coordinates',np.linalg.norm(V.T@V-np.eye(N-1))<1e-13 and np.linalg.norm(V.sum(axis=0))<1e-13)
    check('strict compact set nesting',R1_LO<R0_LO<R0_HI<1<R1_HI<R2)
    check('inward reduced radial fields',R1_LO*(1-R1_LO**2)>0 and R1_HI*(1-R1_HI**2)<0)
    check('positive reduced orbit margin',c['d_star']>0.00414)
    check('backward flow exists on entire K1',R1_HI<c['backward_critical_radius'])
    check('positive phase collar margin',c['phase_collar_margin']>0.01936)
    check('normal contraction and compactification gap',c['rho']>0 and CLOCK_ORDER%2==0 and c['order_r_gap']>0.449999)
    check('both strict first exit gates',Q0<c['gate_normal'] and Q0<c['gate_base'])
    check('reported rounded epsilon lower bound',0.02612<c['epsilon_star']<0.02613)
    check('normal no-exit margin',c['normal_exit_upper_bound']<Q_STAR)
    check('base no-exit margin',c['base_exit_upper_bound']<c['d_star'])
    check('phase collar has positive inner radius',c['backward_radius_min']>0)

    # A constant metric on the ambient network induces exactly the selected
    # Euclidean mean metric while keeping the normal metric unchanged.
    parallel=np.ones((N,N))/N; perp=np.eye(N)-parallel
    ambient=np.kron(perp+parallel/N,np.eye(2)); lift=np.kron(np.ones((N,1)),np.eye(2))
    check('ambient product metric positive definite',np.linalg.eigvalsh(ambient).min()>0)
    check('induced mean metric is identity',np.linalg.norm(lift.T@ambient@lift-np.eye(2))<1e-13)
    normal=np.kron(V,np.eye(2))
    check('normal metric remains identity',np.linalg.norm(normal.T@ambient@normal-np.eye(10))<1e-13)
    check('tangent normal remain orthogonal',np.linalg.norm(lift.T@ambient@normal)<1e-13)
    check('standard Euclidean normalization gives same gate',abs((math.sqrt(N)*c['d_star'])/(math.sqrt(N)*c['L_h'])-c['d_star']/c['L_h'])<1e-13)

    # Exact symbolic derivative, Taylor identity and radial ODE checks.
    a,b,u,v,w=sy.symbols('a b u v w',real=True)
    Y=sy.Matrix([a,b]); Z=sy.Matrix([u,v]); J=sy.Matrix([[0,-1],[1,0]])
    field=lambda x:(1-(x.T*x)[0])*x+2*J*x
    derivative=field(Y).jacobian(Y)
    expected=(1-a*a-b*b)*sy.eye(2)+2*J-2*Y*Y.T
    for i in range(2):
        for j in range(2):check(f'symbolic node Jacobian {i},{j}',sy.expand(derivative[i,j]-expected[i,j])==0)
    remainder=-(2*(Y.T*Z)[0]*Z+(Z.T*Z)[0]*Y+(Z.T*Z)[0]*Z)
    for i in range(2):check(f'symbolic cubic remainder {i}',sy.expand((field(Y+Z)-field(Y)-derivative*Z-remainder)[i])==0)
    z,t=sy.symbols('z t',positive=True)
    radial_sq=z*sy.exp(2*t)/(1+z*(sy.exp(2*t)-1))
    check('symbolic radial logistic equation',sy.simplify(sy.diff(radial_sq,t)-2*radial_sq*(1-radial_sq))==0)
    check('symbolic radial initial condition',sy.simplify(radial_sq.subs(t,0)-z)==0)
    for key,value in [('beta',1-R1_LO**2),('c_r',3*R1_HI+Q_STAR),('L_2',(3*R1_HI+Q_STAR)/N),('L_h',(3*R1_HI+Q_STAR)*Q_STAR/N),('b_star',1-R1_LO**2+(3*R1_HI+Q_STAR)*Q_STAR)]:
        check('independent scalar '+key,abs(value-c[key])<1e-14)
    integral,_=quad(lambda tt:((T-tt)/T)**RHO,0,T,epsabs=1e-13,epsrel=1e-13)
    integral2,_=quad(lambda tt:((T-tt)/T)**(2*RHO),0,T,epsabs=1e-13,epsrel=1e-13)
    check('schedule I1 quadrature',abs(integral-c['I_1'])<1e-12)
    check('schedule I2 quadrature',abs(integral2-c['I_2'])<1e-12)

    rng=np.random.default_rng(SEED)
    extrema={'h_quadratic_ratio_max':0.0,'normal_quadratic_ratio_max':0.0,
             'forward_map_ratio_max':0.0,'backward_map_ratio_max':0.0}
    for index in range(400):
        angle=rng.uniform(-math.pi,math.pi)
        radius=R1_LO if index==0 else R1_HI if index==1 else rng.uniform(R1_LO,R1_HI)
        y=radius*np.array([math.cos(angle),math.sin(angle)])
        z0=rng.normal(size=(N-1,2)); z0*=Q_STAR*rng.uniform(0.01,0.9999)/np.linalg.norm(z0)
        eta=V@z0;qn=float(np.linalg.norm(eta));h,r=nonlinear_terms(y,eta)
        hratio=float(np.linalg.norm(h))/qn**2; rratio=float(np.linalg.norm(r))/qn**2
        extrema['h_quadratic_ratio_max']=max(extrema['h_quadratic_ratio_max'],hratio)
        extrema['normal_quadratic_ratio_max']=max(extrema['normal_quadratic_ratio_max'],rratio)
        check(f'quadratic tangential sample {index}',hratio<=c['L_2']+1e-12)
        check(f'generic tangential sample {index}',np.linalg.norm(h)<=c['L_h']*qn+1e-14)
        check(f'normal remainder sample {index}',rratio<=c['c_r']+1e-12)
        check(f'ordinary symmetric Jacobian sample {index}',np.linalg.eigvalsh((jacobian(y)+jacobian(y).T)/2).max()<=c['beta']+1e-12)
        direct=sl.node_field(y[None,:]+eta)-sl.node_field(y[None,:])-eta@jacobian(y).T
        check(f'Taylor remainder direct cross check {index}',np.linalg.norm(direct-direct.mean(axis=0)-r)<2e-14)
        tt=float(rng.uniform(0,T))
        forward_ratio=np.linalg.norm(flow_jacobian(y,tt),2)/math.exp(c['mu_g']*tt)
        backward_ratio=np.linalg.norm(flow_jacobian(y,-tt),2)/math.exp(c['mu_g_minus']*tt)
        extrema['forward_map_ratio_max']=max(extrema['forward_map_ratio_max'],float(forward_ratio))
        extrema['backward_map_ratio_max']=max(extrema['backward_map_ratio_max'],float(backward_ratio))
        check(f'forward tangent propagator sample {index}',forward_ratio<=1+1e-12)
        check(f'backward tangent propagator sample {index}',backward_ratio<=1+1e-12)
        check(f'backward forward flow reconstruction {index}',np.linalg.norm(reduced_flow(reduced_flow(y,-tt),tt)-y)<1e-12)

    # Independently compare the scaled-coordinate derivative to the full
    # network field. This guards the cancellation-free integrator itself.
    for index in range(50):
        tau=float(rng.uniform(0,TAU_END))
        yy=np.array([0.9,0.1])+0.02*rng.normal(size=2)
        zz=0.01*rng.normal(size=(N-1,DIM))
        state=np.r_[yy,zz.ravel()]
        derivative=scaled_rhs(tau,state)
        epsilon=math.exp(-RHO*tau)
        xx=yy+epsilon*(V@zz)
        reconstructed=derivative[:2]+epsilon*(V@(derivative[2:].reshape(N-1,DIM)-RHO*zz))
        direct=T*math.exp(-tau)*sl.node_field(xx)-GAIN*(sl.L_RING@xx)
        check(f'scaled to full field identity {index}',np.linalg.norm(reconstructed-direct)<1e-13)

    x_old=sl.initial_condition();y0=x_old.mean(axis=0);eta_old=x_old-y0
    eta0=Q0*eta_old/np.linalg.norm(eta_old); x0=y0+eta0
    initial_rows=[{'node':i+1,'old_x':float(x_old[i,0]),'old_y':float(x_old[i,1]),
                   'local_x':float(x0[i,0]),'local_y':float(x0[i,1]),
                   'eta_x':float(eta0[i,0]),'eta_y':float(eta0[i,1])} for i in range(N)]
    write_csv(output/'local_initial_states.csv',initial_rows)
    check('same initial mean',np.linalg.norm(x0.mean(axis=0)-y0)<1e-15)
    check('same initial disagreement direction',np.linalg.norm(eta0/Q0-eta_old/np.linalg.norm(eta_old))<1e-14)
    check('initial disagreement is prescribed',abs(np.linalg.norm(eta0)-Q0)<1e-15)
    check('initial mean in K0 interior',R0_LO<np.linalg.norm(y0)<R0_HI)
    check('initial data satisfy gate with strict slack',np.linalg.norm(eta0)<c['epsilon_star'])
    tau_grid=np.unique(np.r_[np.linspace(0,TAU_END,601),-np.log1p(-np.linspace(0,T-STOP_GAP,401)/T)])
    # Endpoint is defined analytically, avoiding a last-bit log1p mismatch.
    tau_grid=tau_grid[tau_grid<TAU_END-1e-10];tau_grid=np.r_[tau_grid,TAU_END]
    physical=-T*np.expm1(-tau_grid)
    check('strictly increasing dual-clock output grid',np.all(np.diff(tau_grid)>0) and np.all(np.diff(physical)>0))
    runs=[integrate_scaled(y0,eta0,tau_grid,*spec) for spec in
          [(1e-11,1e-13,0.2),(1e-12,1e-14,0.1),(1e-13,1e-15,0.05)]]
    selected=runs[1];refined=runs[2]
    convergence=[]
    for idx,item in enumerate(runs):
        difference=np.abs(item['q']-refined['q'])
        rel=float(np.max(difference/np.maximum(refined['q'],1e-100)))
        yd=float(np.max(np.linalg.norm(item['y']-refined['y'],axis=1)))
        convergence.append({'run':f'scaled_{idx+1}','method':'DOP853','clock':'tau',
                            'rtol':item['rtol'],'atol':item['atol'],'max_step':item['max_step_tau'],
                            'nfev':item['nfev'],'terminal_q':float(item['q'][-1]),
                            'terminal_mean_radius':float(np.linalg.norm(item['y'][-1])),
                            'max_relative_q_difference_to_refined':rel,'max_mean_difference_to_refined':yd})
        check(f'scaled trajectory convergence {idx}',rel<1e-7 and yd<1e-10,
              max_relative_q_error=rel,max_mean_error=yd)
    # Independent full-state regular-clock formulation.
    def full_tau(tau: float, flat: FloatArray) -> FloatArray:
        x=flat.reshape(N,DIM)
        return (T*math.exp(-tau)*sl.node_field(x)-GAIN*(sl.L_RING@x)).ravel()
    full=solve_ivp(full_tau,(0,TAU_END),x0.ravel(),method='DOP853',
                   rtol=1e-12,atol=1e-14,max_step=0.1,t_eval=tau_grid)
    if not full.success:raise RuntimeError(full.message)
    fx=full.y.T.reshape(-1,N,DIM);fy=fx.mean(axis=1);fe=fx-fy[:,None,:]
    fq=np.linalg.norm(fe.reshape(len(tau_grid),-1),axis=1)
    full_rel=float(np.max(np.abs(fq-refined['q'])/refined['q']))
    full_y=float(np.max(np.linalg.norm(fy-refined['y'],axis=1)))
    check('independent full state tau convergence',full_rel<2e-5 and full_y<2e-10,
          max_relative_q_error=full_rel,max_mean_error=full_y)
    convergence.append({'run':'full_state_tau','method':'DOP853','clock':'tau','rtol':1e-12,'atol':1e-14,'max_step':0.1,
                        'nfev':full.nfev,'terminal_q':float(fq[-1]),'terminal_mean_radius':float(np.linalg.norm(fy[-1])),
                        'max_relative_q_difference_to_refined':full_rel,'max_mean_difference_to_refined':full_y})
    # Physical-time vector field is independently evaluated, including its gain.
    def full_time(tt: float, flat: FloatArray) -> FloatArray:
        x=flat.reshape(N,DIM)
        return (sl.node_field(x)-GAIN/(T-tt)*(sl.L_RING@x)).ravel()
    real=solve_ivp(full_time,(0,float(physical[-1])),x0.ravel(),method='DOP853',
                   rtol=1e-12,atol=1e-14,max_step=0.025,t_eval=physical)
    if not real.success:raise RuntimeError(real.message)
    rx=real.y.T.reshape(-1,N,DIM);ry=rx.mean(axis=1);re=rx-ry[:,None,:]
    rq=np.linalg.norm(re.reshape(len(tau_grid),-1),axis=1)
    real_rel=float(np.max(np.abs(rq-refined['q'])/refined['q']))
    real_y=float(np.max(np.linalg.norm(ry-refined['y'],axis=1)))
    check('independent physical time convergence',real_rel<2e-4 and real_y<2e-10,
          max_relative_q_error=real_rel,max_mean_error=real_y)
    convergence.append({'run':'full_state_physical','method':'DOP853','clock':'t','rtol':1e-12,'atol':1e-14,'max_step':0.025,
                        'nfev':real.nfev,'terminal_q':float(rq[-1]),'terminal_mean_radius':float(np.linalg.norm(ry[-1])),
                        'max_relative_q_difference_to_refined':real_rel,'max_mean_difference_to_refined':real_y})
    write_csv(output/'local_solver_comparison.csv',convergence)

    bound=Q0*np.exp(c['b_star']*physical-RHO*tau_grid)
    ybar=np.array([reduced_flow(y0,float(tt)) for tt in physical])
    radii=np.linalg.norm(selected['y'],axis=1)
    reduced_radii=np.linalg.norm(ybar,axis=1)
    yerror=np.linalg.norm(selected['y']-ybar,axis=1)
    ratio=selected['q']/bound
    check('sampled trajectory normal first exit excluded',np.max(selected['q'])<Q_STAR)
    check('sampled trajectory mean in annulus',np.min(radii)>R1_LO and np.max(radii)<R1_HI)
    check('sampled theorem normal bound',float(np.max(ratio))<=1+1e-11)
    check('sampled generic shadow bound',float(np.max(yerror))<c['base_exit_upper_bound'])
    check('sampled quadratic shadow bound',float(np.max(yerror))<c['quadratic_shadow_uniform_bound'])
    check('terminal bound agrees with high precision formula',abs(bound[-1]/c['terminal_q_bound']-1)<1e-12)
    check('terminal q agrees with assessment exploratory value',abs(float(selected['q'][-1])/3.84573e-9-1)<2e-5)
    rows=[]
    for i,tau in enumerate(tau_grid):
        row={'tau':float(tau),'t':float(physical[i]),'s':float(math.exp(-float(tau))),
             'q':float(selected['q'][i]),'q_bound':float(bound[i]),'q_over_bound':float(ratio[i]),
             'mean_radius':float(radii[i]),'reduced_radius':float(reduced_radii[i]),
             'mean_error':float(yerror[i]),'y1':float(selected['y'][i,0]),'y2':float(selected['y'][i,1])}
        for j in range(N):
            row[f'eta{j+1}_x']=float(selected['eta'][i,j,0]);row[f'eta{j+1}_y']=float(selected['eta'][i,j,1])
        rows.append(row)
    write_csv(output/'local_trajectory.csv',rows)

    summary={'sample_count':len(tau_grid),'terminal_time':float(physical[-1]),'terminal_gap':STOP_GAP,
             'tau_terminal':TAU_END,'initial_mean':y0.tolist(),'initial_mean_radius':float(np.linalg.norm(y0)),
             'old_initial_disagreement':float(np.linalg.norm(eta_old)),
             'local_initial_disagreement':float(np.linalg.norm(eta0)),
             'terminal_disagreement':float(selected['q'][-1]),'terminal_bound':float(bound[-1]),
             'terminal_q_over_bound':float(ratio[-1]),'max_sampled_q_over_bound':float(ratio.max()),
             'mean_radius_min':float(radii.min()),'mean_radius_max':float(radii.max()),
             'max_sampled_mean_error':float(yerror.max()),'terminal_mean_error':float(yerror[-1]),
             'minimum_sampled_mean_boundary_margin':float(min(radii.min()-R1_LO,R1_HI-radii.max())),
             'selected_solver':convergence[1], 'solver_comparison':convergence,
             'integration_note':'Numerical integration stops at T-1e-5. Exact contact at T and continuous-time tube invariance follow from the analytical gate and the main theorem, not from the sampled trajectory.'}
    failures=[item for item in checks if not item['passed']]
    result={'status':'PASS' if not failures else 'FAIL','checks_total':len(checks),'checks_passed':len(checks)-len(failures),
            'seed':SEED,'random_bound_samples':400,'certificate':cert,'trajectory':summary,
            'sampled_diagnostic_extrema':extrema,'checks':checks,'failures':failures,
            'reference_network_source_sha256':sha(code_dir()/'network_experiments.py'),
            'full_nonlinear_monte_carlo_rerun':False}
    (output/'local_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'local_certificate.json').write_text(json.dumps(cert,indent=2)+'\n')
    (output/'local_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    text=(f"Local annular certificate and trajectory validation\nStatus: {result['status']}\nChecks: {result['checks_passed']}/{result['checks_total']}\n"
          f"epsilon_star = {c['epsilon_star']:.14g}\nq0 = {Q0}\nq(T-1e-5) = {summary['terminal_disagreement']:.14g}\n"
          f"theorem bound = {summary['terminal_bound']:.14g}\nmean-radius range = [{summary['mean_radius_min']:.14g}, {summary['mean_radius_max']:.14g}]\n"
          "Sampled diagnostics are not a substitute for the analytical proof.\nFull nonlinear stochastic study was not rerun.\n")
    (output/'local_validation.txt').write_text(text)
    print(text)
    if failures:print(json.dumps(failures,indent=2))
    return 0 if not failures else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT)
    args=parser.parse_args()
    raise SystemExit(run(args.output_dir.resolve()))
