import json
from pathlib import Path
import numpy as np
for m in ('gemma3-12b-it','gemma3-4b-it'):
 suffix='__r4' if m=='gemma3-12b-it' else '__alltok_r4'
 root=Path('/scratch/juno/jvl210002/probing/runs')/m/'L3/cot'/('train_neutral'+suffix)
 if not root.exists(): continue
 a=np.load(root/'hidden.npy',mmap_mode='r');d=json.loads((root/'meta.json').read_text());print(m,a.shape,a.dtype,d['layers'],flush=True)
 for i,L in enumerate(d['layers']):
  x=np.asarray(a[:,:,i],dtype=np.float32)
  bad=int((~np.isfinite(x)).sum())
  if bad: print('BAD',m,L,bad,'finite_max',float(np.abs(x[np.isfinite(x)]).max()),flush=True)
