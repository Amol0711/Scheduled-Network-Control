"""Check the descriptive experiment specification against executable constants."""
from pathlib import Path
import ast
import json
import numpy as np
from artifact_paths import package_dir, code_dir, work_dir, output_dir
import network_experiments as network


def literal_assignments(name):
    values={}
    tree=ast.parse((code_dir()/name).read_text())
    for node in tree.body:
        if not isinstance(node,ast.Assign):continue
        try:
            val=ast.literal_eval(node.value)
        except (ValueError,TypeError):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='array':
                try:val=ast.literal_eval(node.value.args[0])
                except (ValueError,TypeError):continue
            else:continue
        for target in node.targets:
            if isinstance(target,ast.Name):values[target.id]=val
            elif isinstance(target,(ast.Tuple,ast.List)) and isinstance(val,(tuple,list)):
                values.update({t.id:x for t,x in zip(target.elts,val) if isinstance(t,ast.Name)})
    return values


def main():
    cfg=json.loads((package_dir()/'config/experiments.json').read_text());checks=[]
    def check(n,a,b):
        checks.append({'name':n,'passed':bool(np.array_equal(a,b)), 'actual':a,'configured':b})
    n=cfg['network'];s=cfg['stochastic_cap'];l=cfg['local_certificate']
    for name,value in [('N','nodes'),('T','horizon'),('OMEGA','node_rotation_rate'),('K0','coupling_multiplier'),('EPS_T','deterministic_terminal_gap')]:
        check('network '+name,getattr(network,name),n[value])
    phases=np.array(n['deterministic_initial_phases']);radii=np.array(n['deterministic_initial_radii'])
    check('deterministic initial states',network.initial_condition().tolist(),np.column_stack((radii*np.cos(phases),radii*np.sin(phases))).tolist())
    gen=literal_assignments('stochastic_network.py')
    for name,value in [('PATHS','paths'),('REFINEMENT_PATHS','nested_refinement_paths'),('BATCH_SIZE','batch_size'),('SEED','seed'),('SIGMA','sigma'),('PRIMARY_DT','primary_dt'),('COARSE_DT','coarse_dt'),('CAP_RATIOS','cap_ratios'),('QUADRATIC_RATIOS','quadratic_ratios'),('REFINEMENT_RATIOS','refinement_ratios'),('ALPHA_CI','alpha')]:
        check('generator '+name,gen[name],s[value])
    rep=literal_assignments('stochastic_reporting.py')
    for name,value in [('N_PATHS','paths'),('N_REFINEMENT','nested_refinement_paths'),('SEED','seed'),('SIGMA','sigma'),('H_FINE','primary_dt'),('H','coarse_dt'),('RATIOS','cap_ratios'),('FIT_RATIOS','quadratic_ratios'),('REFINEMENT_RATIOS','refinement_ratios'),('ALPHA','alpha')]:
        check('reporting '+name,rep[name],s[value])
    loc=literal_assignments('local_certificate.py')
    for name,value in [('T','horizon'),('RHO','rho'),('Q0','initial_disagreement'),('Q_STAR','tube_radius'),('SEED','diagnostic_seed'),('STOP_GAP','terminal_gap')]:check('local '+name,loc[name],l[value])
    check('local annular radii',[loc[x] for x in ['R0_LO','R0_HI','R1_LO','R1_HI','R2']],l['annular_radii'])
    summary=json.loads((package_dir()/'data/records/local_summary.json').read_text())
    for field,key in [('method','selected_solver'),('clock','selected_clock'),('rtol','selected_rtol'),('atol','selected_atol'),('max_step','selected_max_step')]:check('selected solver '+field,summary['selected_solver'][field],l[key])
    check('number of contrasts',len(s['cap_ratios'])-1,s['bonferroni_contrasts'])
    # The lower-level modules fix ddof=1; no runtime setting can change it.
    calls=[]
    for node in ast.walk(ast.parse((code_dir()/'stochastic_reporting.py').read_text())):
        if isinstance(node,ast.Call):
            calls.extend(ast.literal_eval(k.value) for k in node.keywords if k.arg=='ddof')
    check('sample-variance ddof constants',calls,[s['sample_variance_ddof']]*len(calls))
    out={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','checks_passed':sum(c['passed'] for c in checks),'checks_total':len(checks),'checks':checks}
    (output_dir()/'configuration_validation.json').write_text(json.dumps(out,indent=2)+'\n')
    print(f"{out['status']}: {out['checks_passed']}/{len(checks)} configuration checks")
    return out['status']!='PASS'
if __name__=='__main__':raise SystemExit(main())
