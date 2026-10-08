from pathlib import Path
import subprocess,sys,json,os
root=Path(__file__).resolve().parents[2].parent
log=root/'probing/log/round6_2026-10-03'
stages=[('code_tests',[sys.executable,'-m','pytest','-q','tests'],root/'codecue'),
        ('fresh_programs',[sys.executable,'scripts/79_screened_replication.py','--stage','prepare'],root/'codecue'),
        ('crossfit',[sys.executable,'scripts/78_crossfit_probes.py'],root/'codecue')]
report=[]
for name,argv,cwd in stages:
 with (log/(name+'.out')).open('w') as stream:
  r=subprocess.run(argv,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':str(cwd/'src')+':'+str(root/'probing/src')})
 report.append({'name':name,'exit_code':r.returncode})
 (log/'preflight_code.json').write_text(json.dumps(report,indent=2)+'\n')
 print(name,r.returncode,flush=True)
 if r.returncode: sys.exit(r.returncode)
(log/'code_ready.ok').write_text('Code tests, independent fresh-program audit and cross-fitting passed.\n')
