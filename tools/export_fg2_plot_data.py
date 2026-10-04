#!/usr/bin/env python3
"""Create the two FG2 plotting-only exports from the original journal CSVs.

No simulation, optimization, fitting, endpoint reconstruction or new statistics.
Run with --source pointing to journal/source and --output to an empty directory.
Input files are read only. Existing output files are never overwritten.
"""
from pathlib import Path
from decimal import Decimal,localcontext
import argparse,csv,hashlib,json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();src=a.source.resolve()/'data';out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    names=['fg2_timeseries_views.csv','fg2_graph_rate_residuals.csv']
    if any((out/n).exists() for n in names):raise SystemExit('Output exists; use a new output directory.')
    rows=list(csv.DictReader((src/'deterministic_timeseries.csv').open()))
    if len(rows)!=1061:raise ValueError('Expected 1,061 original history rows.')
    fields=['source_row','normalized_time','s','inverse_time_minus_native','inverse_square_minus_native','constant_disagreement','capped_disagreement','fiber_disagreement','xu_liu_disagreement']
    with (out/names[0]).open('w',newline='') as f,localcontext() as ctx:
        ctx.prec=60;w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for i,x in enumerate(rows):
            r={'source_row':str(i),'normalized_time':x['normalized_time'],'s':str(Decimal(1)-Decimal(x['normalized_time'])),
               'inverse_time_minus_native':str(Decimal(x['fiber_phase'])-Decimal(x['native_phase'])),
               'inverse_square_minus_native':str(Decimal(x['xu_liu_phase'])-Decimal(x['native_phase']))}
            for k in fields[5:]:r[k]=x[k]
            w.writerow(r)
    with (out/names[1]).open('w',newline='') as f,localcontext() as ctx:
        ctx.prec=60;w=csv.DictWriter(f,fieldnames=['graph','source_row','predicted_rho','fitted_rho','residual'],lineterminator='\n');w.writeheader()
        for g in ['path','ring','complete']:
            r=list(csv.DictReader((src/f'exponent_{g}.csv').open()))
            if len(r)!=3:raise ValueError('Expected three stored fits per graph.')
            for i,x in enumerate(r):w.writerow({'graph':g,'source_row':str(i),**x,'residual':str(Decimal(x['fitted_rho'])-Decimal(x['predicted_rho']))})
    print(json.dumps({n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in names},indent=2))
if __name__=='__main__':main()
