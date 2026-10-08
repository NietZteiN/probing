import json
from pathlib import Path
import numpy as np
for m in ('gemma3-12b-it','gemma3-4b-it'):
 suffix='__r4' if m=='gemma3-12b-it' else '__alltok__r4'
 root=Path('/scratch/juno/jvl210002/probing/runs')/m/'L3/cot'/('train_neutral'+suffix)
 if not root.exists(): continue
 a=np.load(root/'hidden.npy',mmap_mode='r');d=json.loads((root/'meta.json').read_text());print(m,a.shape,a.dtype,flush=True)
 for i in (0,10,20,min(29,len(d['layers'])-1),len(d['layers'])-2,len(d['layers'])-1):
  x=np.asarray(a[:64,:,i],dtype=np.float32)
  print(m,d['layers'][i],'nonfinite',int((~np.isfinite(x)).sum()),'finite_max',float(np.abs(x[np.isfinite(x)]).max()),flush=True)
