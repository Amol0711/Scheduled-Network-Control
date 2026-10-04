"""Recover actual deterministic designs and rerun the corresponding selected ODEs.
Source binding is to the sealed archive, not guessed reviewer conventions.
The nine graph-rate runs start at t=1.6 (s=0.2) in the original implementation.
"""
from __future__ import annotations
import ast,math
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from reference_common import Archive, Checks, write_csv, write_json, digest
from ode_common import N,T,K0,OMEGA,EPS,X0,L,PHASES,RADII,field,laplacian,physical_inverse


def run(archive:Archive,out:Path)->dict:
    ck=Checks();text=archive.text('code/network_experiments.py');tree=ast.parse(text)
    funcs={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    names=['initial_condition','simulate_deterministic','simulate_full_clock_tau',
           'solve_reduced','phase_map_validation','main','graph_laplacian']
    excerpts=[]
    for name in names:
        n=funcs[name];src=ast.get_source_segment(text,n)
        excerpts.append({'function':name,'start_line':n.lineno,'end_line':n.end_lineno,
                         'sha256':digest(src.encode()),'source':src})
    contracts={
        'physical_solver':'method="Radau",',
        'physical_span_uses_first_output_time':'(float(t_eval[0]), float(t_eval[-1]))',
        'base_time_grid':'t_eval = np.linspace(0.0, t_end, 4001)',
        'main_fit_mask':'main_mask = (T - t_eval) / T < 0.03',
        'graph_grid':'s_grid = np.geomspace(2e-5, 0.2, 500)[::-1]',
        'graph_rates':'target_rhos = [0.6, 1.0, 1.4]',
        'graph_mask':'mask = (s_grid < 0.03) & (s_grid > 2e-4) & (dis > 1e-12)',
        'graph_gain':'k0 = target_rho / lambda2',
        'graph_simulation_receives_t_slope':'"fiber_ptc", x0, t_slope, lap=lap, k0=k0, compute_energy=False',
        'displacement_base':'base = np.array([0.8 * math.cos(0.2), 0.8 * math.sin(0.2)])',
        'displacement_centering':'direction -= direction.mean(axis=0)',
        'displacement_normalization':'direction /= np.linalg.norm(direction)',
        'displacement_scales':'eps_values = np.geomspace(0.015, 0.6, 12)',
        'displacement_fit_count':'fit_count = 8',
        'phase_grid':'s_values = np.geomspace(0.2, 5.0e-4, 240)',
        'phase_solver_rtol':'rtol=2e-12,','phase_solver_atol':'atol=2e-13,',
        'phase_endpoint_25':'tau_terminal_1 = 25.0','phase_endpoint_30':'tau_terminal_2 = 30.0'}
    for name,s in contracts.items():ck.require('source:'+name,s in text)
    write_json(out/'SOURCE_BINDINGS.json',{'file':'code/network_experiments.py','sha256':digest(text.encode()),
                                        'source_contracts':contracts,'functions':excerpts})
    summary=archive.json('data/records/numerical_summary.json')
    # Figure 2(a): independent re-integration under the original physical-time protocol.
    t=np.linspace(0,T-EPS,4001);s=(T-t)/T;mask=s<.03
    x,nfev=physical_inverse(X0,t);y=x.mean(axis=1);dis=np.linalg.norm((x-y[:,None,:]).reshape(len(t),-1),axis=1)
    slope=float(np.polyfit(np.log(s[mask]),np.log(dis[mask]),1)[0])
    ck.close('main:fitted_exponent',slope,summary['fitted_main_exponent'],atol=1e-6)
    base_rows=[{'t':float(tt),'s':float(ss),'disagreement':float(d),'fit_selected':int(m)} for tt,ss,d,m in zip(t,s,dis,mask)]
    write_csv(out/'panel_a_reintegration.csv',base_rows)
    figure_source=archive.text('code/figure_series.py')
    plot_indices=sorted(set(range(0,len(t),4))|set(range(len(t)-81,len(t))))
    plot_records=archive.array('data/figure_series/deterministic_timeseries.csv')
    ck.require('plot_decimation:source_every_fourth','set(range(0, len(t_eval), 4))' in figure_source)
    ck.require('plot_decimation:source_last_81','len(t_eval) - 81' in figure_source)
    ck.require('plot_decimation:1061_rows',len(plot_records)==len(plot_indices)==1061)
    ck.require('plot_decimation:time_values',np.allclose(plot_records['normalized_time'],t[plot_indices]/T,rtol=0,atol=1e-15))
    main={'t_start':float(t[0]),'t_end':float(t[-1]),'samples':len(t),'fit_points':int(mask.sum()),
          'fit_s_min':float(s[mask].min()),'fit_s_max':float(s[mask].max()),'mask':'s<0.03',
          'fitted_slope':slope,'frozen_fitted_slope':summary['fitted_main_exponent'],'nfev':nfev,
          'plot_sample_count':1061,'plot_selection':'every fourth sample plus all final 81 samples',
          'initial_state':X0.tolist(),'rtol':1e-10,'atol':1e-12,'method':'Radau'}
    # Figure 2(c): exact original late-window starting-time convention, no silent reset to zero.
    sg=np.geomspace(2e-5,.2,500)[::-1];tg=T*(1-sg);graphrows=[];graphseries=[]
    old=archive.csv('data/records/exponent_validation.csv')
    ck.close('graph:actual_initial_time',float(tg[0]),1.6)
    for kind in ('path','ring','complete'):
        lap=laplacian(kind);lam=float(np.linalg.eigvalsh(lap)[1])
        for rate in (.6,1.,1.4):
            gain=rate/lam;xx,calls=physical_inverse(X0,tg,lap,gain)
            dd=np.linalg.norm((xx-xx.mean(axis=1)[:,None,:]).reshape(len(tg),-1),axis=1)
            mm=(sg<.03)&(sg>2e-4)&(dd>1e-12)
            fit=float(np.polyfit(np.log(sg[mm]),np.log(dd[mm]),1)[0]);prior=old[len(graphrows)]
            ck.close(f'{kind}:{rate}:lambda2',lam,float(prior['lambda2']),atol=1e-14)
            ck.close(f'{kind}:{rate}:k0',gain,float(prior['k0']),atol=1e-13)
            ck.close(f'{kind}:{rate}:slope',fit,float(prior['fitted_rho']),atol=2e-6)
            ck.require(f'{kind}:{rate}:initial_state_at_t1p6',np.array_equal(xx[0],X0))
            row={'graph':kind,'N':N,'lambda2':lam,'k0':gain,'predicted_rho':rate,
                 'fitted_rho':fit,'frozen_fitted_rho':float(prior['fitted_rho']),
                 'error':fit-rate,'t_initial':float(tg[0]),'t_final':float(tg[-1]),
                 'fit_points':int(mm.sum()),'fit_s_min':float(sg[mm].min()),'fit_s_max':float(sg[mm].max()),'nfev':calls}
            graphrows.append(row)
            graphseries.extend({'graph':kind,'predicted_rho':rate,'t':float(tt),'s':float(ss),
                                'disagreement':float(d),'fit_selected':int(m)} for tt,ss,d,m in zip(tg,sg,dd,mm))
    write_csv(out/'graph_protocol_and_reintegration.csv',graphrows);write_csv(out/'graph_rate_samples.csv',graphseries)
    # Figure 2(d): not the separate local-certified initial condition.
    base=np.array([.8*math.cos(.2),.8*math.sin(.2)])
    D=np.array([[1.,.2],[-.7,.5],[.3,-1.1],[-.5,-.4],[.8,.9],[-.9,-.1]])
    D0=D.copy();D-=D.mean(axis=0);D/=np.linalg.norm(D)
    eps=np.geomspace(.015,.6,12);ts=np.linspace(0,T-EPS,1601)
    sol=solve_ivp(lambda _,v:field(v[None,:])[0],(0,float(ts[-1])),base,t_eval=ts,
                  method='DOP853',rtol=1e-12,atol=1e-13)
    if not sol.success:raise RuntimeError(sol.message)
    base_end=sol.y[:,-1];disp=[]
    old_disp=archive.csv('data/records/tangential_scaling.csv')
    for j,e in enumerate(eps):
        xx,calls=physical_inverse(base+e*D,ts)
        error=float(np.linalg.norm(xx[-1].mean(axis=0)-base_end))
        prior=float(old_disp[j]['terminal_projection_orbit_error'])
        ck.close(f'displacement:{j}:error',error,prior,atol=3e-11,rtol=2e-7)
        disp.append({'epsilon':float(e),'terminal_projection_orbit_error':error,
                     'frozen_error':prior,'fit_selected':int(j<8),'nfev':calls})
    displacement_slope=float(np.polyfit(np.log(eps[:8]),np.log([r['terminal_projection_orbit_error'] for r in disp[:8]]),1)[0])
    ck.close('displacement:slope',displacement_slope,summary['quadratic_tangential_offset_slope'],atol=2e-6)
    ck.close('displacement:direction_norm',float(np.linalg.norm(D)),1.,atol=2e-15)
    ck.require('displacement:direction_centered',np.max(abs(D.sum(axis=0)))<1e-15)
    write_csv(out/'displacement_protocol_and_reintegration.csv',disp)
    # Continuous versus sampled phase on the actual 4001-point grid.
    tau=np.log(T/(T-t));theta0=np.arctan2(X0[:,1].mean(),X0[:,0].mean())
    def full_rhs(clock,z):
        xx=z[:2*N].reshape(N,2);dx=field(xx)-K0*(L@xx)
        yy=xx.mean(axis=0);dy=dx.mean(axis=0)
        return np.r_[dx.ravel(),(yy[0]*dy[1]-yy[1]*dy[0])/np.dot(yy,yy)]
    sol=solve_ivp(full_rhs,(0,float(tau[-1])),np.r_[X0.ravel(),theta0],t_eval=tau,
                  method='DOP853',rtol=1e-12,atol=1e-13)
    if not sol.success:raise RuntimeError(sol.message)
    yy=sol.y[:2*N].T.reshape(-1,N,2).mean(axis=1);continuous=sol.y[2*N]
    unwrap=np.unwrap(np.arctan2(yy[:,1],yy[:,0]));native=theta0+OMEGA*t
    phase_excess=float(continuous[-1]-native[-1]);alias=float(unwrap[-1]-native[-1]);gap=phase_excess-alias
    phase_ref=summary['full_clock_phase_regression']
    ck.close('unwrap:continuous_excess',phase_excess,phase_ref['continuous_phase_error'],atol=2e-8)
    ck.close('unwrap:sampled_excess',alias,phase_ref['legacy_uniform_t_unwrap_error'],atol=2e-8)
    ck.close('unwrap:gap_one_revolution',gap,2*np.pi,atol=1e-9)
    ck.require('unwrap:final_continuous_increment_exceeds_two_pi',continuous[-1]-continuous[-2]>2*np.pi)
    write_csv(out/'full_clock_phase_grid.csv',[{'t':float(tt),'tau':float(u),'continuous_phase':float(c),'sampled_unwrap_phase':float(d)} for tt,u,c,d in zip(t,tau,continuous,unwrap)])
    unwrap_record={'samples':len(t),'t_start':0,'t_end':float(t[-1]),'continuous_excess':phase_excess,
                   'sampled_unwrap_excess':alias,'gap':gap,
                   'last_continuous_increment':float(continuous[-1]-continuous[-2]),
                   'scope':'This specific sampled grid only; no universal sample-count threshold.'}
    report={'panel_a':main,'panel_c':{'grid':'s=geomspace(2e-5,0.2,500)[::-1]',
            'mask':'2e-4<s<0.03 and disagreement>1e-12','initial_state':X0.tolist(),
            'initial_time':1.6,'original_protocol_late_initialization':True,
            'clarification':'The original graph comparison initializes X0 at t=1.6, not at t=0. Its exponent prediction is unchanged, but its full trajectory is not the panel-a trajectory.',
            'runs':graphrows},'panel_d':{'base':base.tolist(),'raw_direction':D0.tolist(),'centered_unit_direction':D.tolist(),
            'epsilon_definition':'X_i(0)=base+epsilon*D_i, mean(D)=0 and ||D||_F=1; epsilon=||eta(0)||_2.',
            'epsilon_grid':eps.tolist(),'fit_count':8,'fit_max_epsilon':float(eps[7]),
            'fitted_slope':displacement_slope,'frozen_slope':summary['quadratic_tangential_offset_slope'],
            'time_samples':1601,'local_certified_initialization':False},'panel_b_unwrap':unwrap_record,
            'code_executed_from_archive':False,'original_figure_data_changed':False,
            'checks_interpretation':'Numerical agreement at declared tolerances, not proof of theoretical exponents.',**ck.report()}
    write_json(out/'protocol_reference.json',report)
    print('protocols main',slope,'displacement',displacement_slope,'graph t_initial',tg[0],flush=True)
    return {'checks':len(ck.records),'main_runs':1,'graph_runs':len(graphrows),'displacement_runs':len(disp),'phase_grid_runs':1}
