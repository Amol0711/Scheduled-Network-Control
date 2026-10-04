#!/usr/bin/env python3
"""Read-only journal plotting transforms. Never runs scientific-method code."""
from __future__ import annotations
import argparse, csv, hashlib, json
from decimal import Decimal, localcontext
from pathlib import Path

D=Decimal

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path: Path):
    with path.open(newline='') as f:
        rd=csv.DictReader(f); rows=list(rd); cols=rd.fieldnames
    if not rows or not cols: raise ValueError(f'Empty CSV {path}')
    return cols,rows

def write(path: Path, cols: list[str], rows: list[dict[str,str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=cols,lineterminator='\n'); wr.writeheader(); wr.writerows(rows)

def export(root: Path, out: Path) -> dict:
    scalar=root/'verified_current/journal/source/data/optimized_noise_scaling.csv'
    stats=root/'simulation/companion_1/data/statistics/stochastic_cap_statistics.csv'
    sc,sr=load(scalar); nc,nr=load(stats)
    if len(sr)!=31 or len(nr)!=16: raise ValueError('Registered row counts changed')
    out.mkdir(parents=True,exist_ok=True)
    with localcontext() as ctx:
        ctx.prec=60
        snew=[]
        for i,r in enumerate(sr):
            e=D(r['bounded_profile_envelope_rms'])
            if e<=0: raise ValueError('Nonpositive RMS denominator')
            snew.append({'source_row':str(i),'sigma':r['sigma'],
                         'inverse_time_rms_ratio':str(D(r['inverse_time_rms'])/e),
                         'inverse_square_rms_ratio':str(D(r['inverse_square_rms'])/e)})
        write(out/'fg3_scalar_rms_ratios.csv',list(snew[0]),snew)
        nnew=[]; refs=[]
        for i,r in enumerate(nr):
            if D(r['K_over_Kopt'])==1: refs.append(i)
            if D(r['paths'])!=20000 or D(r['dt'])!=D('0.0005'): raise ValueError('Protocol changed')
            rr={'source_row':str(i),**r}
            for prefix,mid,lo,hi in [('marginal','nonlinear_mean','ci95_low','ci95_high'),
                                     ('simultaneous','paired_difference','bonferroni15_ci95_low','bonferroni15_ci95_high')]:
                m,l,h=map(D,(r[mid],r[lo],r[hi]))
                if not l<=m<=h: raise ValueError('Interval ordering invalid')
                rr[prefix+'_minus']=str(m-l); rr[prefix+'_plus']=str(h-m)
            # Exact decimal unit conversions for readable axes, not new estimates.
            for name,base,scale in [('modal_1e4','modal_mse','1e4'),('mean_1e4','nonlinear_mean','1e4'),
                ('marginal_minus_1e4','marginal_minus','1e4'),('marginal_plus_1e4','marginal_plus','1e4'),
                ('contrast_1e5','paired_difference','1e5'),('simultaneous_minus_1e5','simultaneous_minus','1e5'),
                ('simultaneous_plus_1e5','simultaneous_plus','1e5'),('contrast_1e8','paired_difference','1e8'),
                ('simultaneous_minus_1e8','simultaneous_minus','1e8'),('simultaneous_plus_1e8','simultaneous_plus','1e8')]:
                rr[name]=str(D(rr[base])*D(scale))
            nnew.append(rr)
        if refs!=[8] or any(D(nr[8][k])!=0 for k in ['paired_difference','bonferroni15_ci95_low','bonferroni15_ci95_high']):
            raise ValueError('Reference row differs from registered table')
        write(out/'fg3_network_cap_plot.csv',list(nnew[0]),nnew)
    return {'stage':'FG3','precision':'60 decimal digits for divisions; decimal subtraction and unit conversion exact at this precision',
        'source_hashes':{str(scalar.relative_to(root)):sha(scalar),str(stats.relative_to(root)):sha(stats)},
        'exports':[{'path':f.name,'sha256':sha(f),'rows':31 if 'scalar' in f.name else 16} for f in sorted(out.glob('fg3_*.csv'))],
        'scalar_transform':'inverse_time_rms / bounded_profile_envelope_rms; inverse_square_rms / bounded_profile_envelope_rms',
        'network_transform':'All 23 source columns copied exactly. Source row added. Error lengths = center-low and high-center. Display units multiply by powers of ten.',
        'reference_source_row':8,'reference_K':nr[8]['K'],'near_reference_zoom_rows':[6,7,8,9,10],
        'new_simulations':0,'new_optimizations':0,'new_fits':0,'new_interval_estimation':0}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--report',type=Path)
    a=p.parse_args();r=export(a.root.resolve(),a.out.resolve())
    if a.report:a.report.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
