"""Bind consolidated model/protocol metadata to the unchanged scientific inputs."""
from pathlib import Path
import ast,hashlib,json,sys
import numpy as np
from artifact_paths import package_dir, output_dir
ROOT=package_dir()
def main():
    checks=[]
    def ck(name,value):
        checks.append({'id':name,'passed':bool(value)})
        if not value:raise AssertionError(name)
    cfg=json.loads((ROOT/'config/experiments.json').read_text())
    original=json.loads((ROOT/'config/reference_inputs.json').read_text())
    for name,h in original['unchanged_source_data_sha256'].items():
        ck('pin:'+name,hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h)
    text=(ROOT/'code/stochastic_network.py').read_text()
    m=cfg['noise_model'];d=cfg['deterministic_protocol'];s=cfg['stochastic_cap']
    ck('deterministic_common_tangent',m['direction']==['-sin(omega*t)','cos(omega*t)'] and 'tangent_x = -math.sin(OMEGA * time)' in text and 'tangent_y = math.cos(OMEGA * time)' in text)
    ck('state_independent_diffusion',m['diffusion_state_dependent'] is False and 'noise_scalar = -scheduled_gain * sigma * lap_dw[i]' in text)
    ck('ring_node_noise',all(x in text for x in ['2.0 * brownian_increments[step, path, i]','brownian_increments[step, path, (i - 1) % n_nodes]','brownian_increments[step, path, (i + 1) % n_nodes]']))
    ck('phase_offsets_not_sines',m['initial_modal_coordinates']=='V.T @ centered_phase_offsets; not V.T @ sin(phases)')
    ck('phase_base',s['phase_offset_base']==[.08,-.05,.12,-.10,.03,-.08])
    ck('late_graph_initialization',d['graph_start']==1.6 and d['graph_s_grid']==[.2,.00002,500])
    ck('time_history_grid',d['time_history_points']==4001 and d['time_history_start']==0.0)
    ck('graph_modes',d['graph_nodes']==6 and d['graph_kinds']==['path','ring','complete'] and d['graph_target_rates']==[.6,1.,1.4])
    ck('displacement_base',d['displacement_base_radius']==.8 and d['displacement_base_phase']==.2)
    ck('separate_displacement',d['displacement_raw_direction']==[[1,.2],[-.7,.5],[.3,-1.1],[-.5,-.4],[.8,.9],[-.9,-.1]])
    ck('displacement_grid',d['displacement_scales']==[.015,.6,12] and d['displacement_fit_first']==8)
    ck('phase_endpoints',d['phase_raw_endpoint_tau']==30 and d['phase_transported_comparison_tau']==[25,30,35] and d['phase_grid']==[.2,.0005,240])
    lineage=json.loads((ROOT/'config/supersession.json').read_text())
    for item in lineage['supersessions']:
        for k in ['old','new']:
            if k in item:ck('supersession:'+item['topic']+':'+k,hashlib.sha256((ROOT/item[k]).read_bytes()).hexdigest()==item[k+'_sha256'])
    out={'status':'PASS','checks_total':len(checks),'checks_passed':len(checks),'checks':checks,'metadata_does_not_override_constants':True}
    (output_dir()/'release_validation.json').write_text(json.dumps(out,indent=2)+'\n')
    print('PASS',len(checks),'release bindings')
if __name__=='__main__':main()
