import subprocess
for role, flag, n in [('v1','--lure-errors',3), ('v1','--errors',2), ('v2','--errors',2)]:
    subprocess.run(['python','scripts/64_kudo_fig3.py','--model','llama32-3b','--level','3',
                    '--regime','cot','--role',role,flag,'--n',str(n)],check=True)
