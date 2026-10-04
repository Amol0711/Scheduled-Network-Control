#!/usr/bin/env python3
"""Reconstruct sealed certificate and trajectory diagnostics without random paths."""
from pathlib import Path
import json
import numpy as np
import mpmath as mp
from artifact_paths import work_dir, output_dir
S=work_dir()
checks=[]
def check(name,ok,detail=None):checks.append({'check':name,'pass':bool(ok),'detail':detail})
def close(name,value,target,tol='1e-65'):
 check(name,abs(value-mp.mpf(str(target)))<=mp.mpf(tol)*max(1,abs(value)),{'recomputed':mp.nstr(value,72),'sealed':str(target),'tolerance':tol})
mp.mp.dps=80
T=mp.mpf(2);rho=mp.mpf('1.2');q0=mp.mpf('.01');qstar=mp.mpf('.07')
r0i,r0o,r1i,r1o,r2=map(mp.mpf,['.87','.90','.85','1.002','1.15'])
def radius(t,r):
 e=mp.exp(2*t);return mp.sqrt(r*r*e/(1+r*r*(e-1)))
beta=1-r1i*r1i;cr=3*r1o+qstar;L2=cr/6;Lh=L2*qstar;b=beta+cr*qstar
I1=T/(rho+1);I2=T/(2*rho+1);d=min(r0i-r1i,r1o-radius(T,r0o));back=radius(-T,r1o)
v={'r0_inner':r0i,'r0_outer':r0o,'r1_inner':r1i,'r1_outer':r1o,'r2':r2,'q_star':qstar,'rho':rho,'beta':beta,'mu_g':beta,'C_g':mp.mpf(1),'C_g_minus':mp.mpf(1),'mu_g_minus':3*r2*r2-1,'c_r':cr,'L_2':L2,'L_h':Lh,'gamma':cr,'b_star':b,'reduced_outer_terminal_radius':radius(T,r0o),'inner_base_margin':r0i-r1i,'outer_base_margin':r1o-radius(T,r0o),'d_star':d,'I_1':I1,'I_2':I2,'gate_normal':qstar*mp.exp(-b*T),'gate_base':d/(Lh*I1)*mp.exp(-(b+beta)*T),'backward_radius_max':back,'backward_radius_min':radius(-T,r1i),'phase_collar_margin':r2-back,'backward_critical_radius':1/mp.sqrt(1-mp.exp(-2*T)),'order_r_gap':rho-mp.mpf(3)/4,'normal_exit_upper_bound':q0*mp.exp(b*T),'base_exit_upper_bound':Lh*q0*I1*mp.exp((b+beta)*T),'terminal_q_bound':q0*mp.exp(b*(T-mp.mpf('1e-5')))*(mp.mpf('1e-5')/T)**rho}
v['epsilon_star']=min(v['gate_normal'],v['gate_base'])
sealed=json.loads((S/'local_certificate.json').read_text())
for key,value in v.items():close('80_digit_certificate:'+key,value,sealed['values_70_digits'][key])
check('strict_initial_gate',q0<v['epsilon_star'])
check('strict_normal_noexit',v['normal_exit_upper_bound']<qstar)
check('strict_base_noexit',v['base_exit_upper_bound']<d)
check('strict_backward_collar',back<r2)
check('positive_order_r_gap',v['order_r_gap']>0)
check('positive_scheduled_contraction_parameter',rho>0) # scheduled graph eigenvalue is unchanged, not inferred from samples.
a=np.genfromtxt(S/'local_initial_states.csv',delimiter=',',names=True)
old=np.column_stack([a['old_x'],a['old_y']]);loc=np.column_stack([a['local_x'],a['local_y']]);y=old.mean(axis=0)
expect=y+.01*(old-y)/np.linalg.norm(old-y)
check('all_six_initial_states_reconstruct',np.allclose(loc,expect,rtol=0,atol=3e-16))
check('mean_preserved',np.allclose(loc.mean(axis=0),y,rtol=0,atol=3e-16))
check('q0_from_stored_initial_state',abs(np.linalg.norm(loc-loc.mean(axis=0))-.01)<1e-15)
check('initial_mean_in_K0',.87<np.linalg.norm(y)<.90)
t=np.genfromtxt(S/'local_trajectory.csv',delimiter=',',names=True)
check('sealed_trajectory_has_1000_rows',len(t)==1000)
expected_bound=.01*np.exp(float(b)*t['t'])*t['s']**1.2
check('all_stored_analytic_bounds',np.allclose(t['q_bound'],expected_bound,rtol=3e-14,atol=1e-22))
check('all_sampled_ratios_within_bound',np.max(t['q']/t['q_bound'])<=1+1e-12)
check('sampled_radius_stays_in_annulus',np.min(t['mean_radius'])>.85 and np.max(t['mean_radius'])<1.002)
check('terminal_sample_matches_published_digits',abs(t['q'][-1]-3.845727779e-9)<6e-19)
check('terminal_bound_matches_published_digits',abs(t['q_bound'][-1]-1.166325788e-8)<5e-18)

report={'status':'PASS' if all(x['pass'] for x in checks) else 'FAIL',
        'checks_passed':sum(x['pass'] for x in checks),'checks_total':len(checks),
        'precision_digits':80,'new_trajectories':0,'new_random_draws':0,
        'ode_solver_calls':0,'checks':checks,
        'limits':['Sampled inequalities are diagnostics, not continuous-time proofs.',
                  'Solver agreement alone is not an integration-error bound.']}
(output_dir()/'certificate_reconstruction.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['status'],f"{report['checks_passed']}/{len(checks)} certificate checks")
raise SystemExit(0 if report['status']=='PASS' else 1)
