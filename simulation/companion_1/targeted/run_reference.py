"""Independent precision, actuation, protocol, phase and statistical references."""
from __future__ import annotations
import argparse, importlib, json, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
from reference_common import Archive, environment, write_json, digest
STAGES = {'power':'power_reference','effort':'effort_reference',
          'protocols':'protocol_reference','phase':'phase_reference','statistics':'statistics_reference'}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();src=a.package.resolve(strict=True);out=a.output.resolve()
    if a.output.exists() or a.output.is_symlink() or src==out or src in out.parents or out in src.parents:
        p.error('Output must be new and external to the package')
    data=Archive(src);out.mkdir(parents=True)
    report={'status':'RUNNING','environment':environment(),'input':data.metadata(),
            'full_stochastic_campaign_executed':False,'stages':{}}
    write_json(out/'RUN_STATUS.json',report)
    try:
        for stage,module in STAGES.items():
            target=out/stage;target.mkdir();start=time.monotonic()
            val=importlib.import_module(module).run(data,target)
            report['stages'][stage]={'status':'PASS',**val,'elapsed_seconds':time.monotonic()-start}
            write_json(out/'RUN_STATUS.json',report)
        data.assert_unchanged();report['status']='PASS'
        report['checks_passed']=sum(x.get('checks',0) for x in report['stages'].values())
        write_json(out/'RUN_STATUS.json',report)
        (out/'SHA256SUMS.txt').write_text(''.join(f'{digest(f.read_bytes())}  {f.relative_to(out).as_posix()}\n' for f in sorted(out.rglob('*')) if f.is_file() and f.name!='SHA256SUMS.txt'))
        print('PASS',report['checks_passed'],'targeted checks')
    except Exception as exc:
        report['status']='FAIL';report['error']=repr(exc);write_json(out/'RUN_STATUS.json',report);raise
if __name__=='__main__':main()
