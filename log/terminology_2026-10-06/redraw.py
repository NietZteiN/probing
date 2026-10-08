from pathlib import Path
import os,re,subprocess,json,shutil
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader
root=Path('/work/jvl210002/migration/probing');fig=root/'paper/figures';log=root/'log/terminology_2026-10-06'
python='/work/jvl210002/migration/envs/probe-cu129/bin/python'
initial={p.name for p in fig.iterdir()}
jobs=[]
for p in sorted(fig.glob('*.pdf')):
    if p.stem.startswith(('kudo_fig3_','fig2_margin')):continue
    if not re.search(r'\blure\b',' '.join(page.extract_text() for page in PdfReader(p).pages),re.I):continue
    m=re.fullmatch(r'(kudo_fig2|kudo_fig5|fig2_tokens)_(.+)_L(\d+)_(cot|direct)_(v[12])',p.stem)
    if m:
        kind,model,level,regime,role=m.groups()
        script={'kudo_fig2':'53_kudo_figs.py','kudo_fig5':'54_grid_figs.py','fig2_tokens':'58_token_figure.py'}[kind]
        args=['--model',model,'--level',level,'--regime',regime,'--target' if kind=='kudo_fig5' else '--role',role]
        if kind=='kudo_fig2':args+=['--figures-only']
        if model=='gemma3-4b-it':args+=['--cache-suffix','__r5_fp32']
        jobs.append((p.stem,script,args));continue
    if p.stem=='fig1_master_L3':jobs.append((p.stem,'60_master_figure.py',[]));continue
# 52 redraws both roles together, so run once per model.
for model in ['llama32-3b','llama31-8b']:
    jobs.append(('patch_'+model,'52_figs.py',['--model',model,'--level','3','--regime','cot','--patch-regime','direct']))
def run(job):
    name,script,args=job
    with (log/(name+'.out')).open('w') as f:
        p=subprocess.run([python,str(root/'scripts'/script),*args],cwd=root,stdout=f,stderr=subprocess.STDOUT)
    return {'name':name,'script':script,'args':args,'exit_code':p.returncode}
with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,jobs))
# Keep the figure set fixed: 52 also emits unused role panels.
for p in fig.iterdir():
    if p.name not in initial:
        dest=log/'unused_redrawn';dest.mkdir(exist_ok=True);shutil.move(p,dest/p.name)
(log/'redraw_results.json').write_text(json.dumps(results,indent=2)+'\n')
for r in results:print(r['name'],r['exit_code'])
assert all(r['exit_code']==0 for r in results)
