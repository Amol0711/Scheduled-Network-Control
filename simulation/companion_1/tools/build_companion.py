#!/usr/bin/env python3
"""Render the supplied computational documentation to a fresh external directory.
Requires pandoc, pdflatex and the template's TeX packages. Numerical routes do
not use this tool or require any typesetting installation.
"""
from pathlib import Path
import argparse,os,shutil,subprocess,sys,tempfile,re
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();dest=a.output.expanduser().resolve()
    if dest==ROOT or ROOT in dest.parents or dest in ROOT.parents:
        p.error('Use a fresh external output directory, not the package or its ancestor.')
    if dest.exists():p.error('Output already exists; preserve it and choose a new directory.')
    for exe in ['pandoc','pdflatex']:
        if shutil.which(exe) is None:p.error('Required document builder is missing: '+exe)
    dest.mkdir(parents=True)
    env=os.environ.copy();env.update(SOURCE_DATE_EPOCH='1790726400',FORCE_SOURCE_DATE='1',TZ='UTC')
    md=ROOT/'docs/COMPUTATIONAL_COMPANION.md';template=ROOT/'tools/companion_template.tex'
    # No executable source code is embedded in or fetched by the document build.
    def run(args,name):
        with (dest/name).open('w') as f:subprocess.run(args,cwd=dest,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    tex=dest/'COMPUTATIONAL_COMPANION.tex'
    run(['pandoc',str(md),'--from=markdown+tex_math_dollars','--to=latex','--standalone','--toc','--toc-depth=1','--template='+str(template),'-o',str(tex)],'pandoc_latex.log')
    # Explicit fixed-width allocations keep the unchanged table cells readable.
    text=tex.read_text()
    def table_widths(match):
        block=match.group(0)
        widths=None
        if 'Nonlinear' in block and 'Simultaneous' in block:
            widths=[.10,.11,.26,.09,.135,.305]
        elif 'Profile' in block and 'I_\\kappa/T' in block:
            widths=[.20,.125,.125,.15,.15,.125,.125]
        if widths is not None:
            it=iter(widths)
            block=re.sub(r'\\real\{[0-9.]+\}',lambda _: '\\real{'+format(next(it),'.5f')+'}',block)
        return block
    text=re.sub(r'\\begin\{longtable\}.*?\\end\{longtable\}',table_widths,text,flags=re.S)
    text=text.replace(r"\subsection{Table C2.",r"\Needspace{24\baselineskip}"+"\n"+r"\subsection{Table C2.")
    tex.write_text(text)
    for k in range(3):run(['pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error',tex.name],f'latex_{k+1}.log')
    run(['pandoc',str(md),'--from=markdown+tex_math_dollars','--to=html5','--standalone','--toc','--toc-depth=1','--mathml','--embed-resources','--include-in-header='+str(ROOT/'tools/companion_style.html'),'-o',str(dest/'COMPUTATIONAL_COMPANION.html')],'pandoc_html.log')
    for name in ['EVIDENCE_INDEX','RECORD_LOCATIONS']:
        run(['pandoc',str(ROOT/'docs'/f'{name}.md'),'--to=html5','--standalone','--metadata','title='+name.replace('_',' ').title(),'--include-in-header='+str(ROOT/'tools/companion_style.html'),'-o',str(dest/f'{name}.html')],f'pandoc_{name}.log')
    print('Documentation rendered in',dest)
if __name__=='__main__':main()
