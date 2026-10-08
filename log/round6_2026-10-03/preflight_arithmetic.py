from pathlib import Path
import subprocess,sys,json,os
root=Path(__file__).resolve().parents[2].parent
log=root/'probing/log/round6_2026-10-03'
stages=[('arithmetic_tests',[sys.executable,'-m','pytest','-q','tests/test_round6.py','tests/test_followups.py','tests/test_cache_precision.py'],root/'probing'),
        ('code_round6_final_tests',[sys.executable,'-m','pytest','-q','tests/test_round6.py'],root/'codecue'),
        ('generated_cohorts',[sys.executable,'scripts/80_generated_trained_probes.py','--prepare'],root/'probing')]
report=[]
for name,argv,cwd in stages:
 with (log/(name+'.out')).open('w') as stream:
  run=subprocess.run(argv,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':str(cwd/'src')})
 report.append({'name':name,'exit_code':run.returncode})
 (log/'preflight_arithmetic.json').write_text(json.dumps(report,indent=2)+'\n')
 print(name,run.returncode,flush=True)
 if run.returncode:sys.exit(run.returncode)
(log/'arithmetic_ready.ok').write_text('Generated-cohort disjointness and boundary/precision tests passed.\n')
