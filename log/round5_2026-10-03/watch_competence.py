"""Publish the nine-model follow-up as soon as its validated summary arrives."""
from pathlib import Path
import json
import subprocess
import sys
import time
root=Path(__file__).resolve().parents[2]
source=root/'results/summary/round5_competence.json'
end=time.monotonic()+35*60
while time.monotonic()<end:
 try:
  data=json.loads(source.read_text())
  if data.get('validated') and data.get('n_models')==9:
   sys.exit(subprocess.run([sys.executable,'scripts/78_round5_completion.py','--allow-partial'],cwd=root).returncode)
 except (OSError,ValueError):
  pass
 time.sleep(20)
raise TimeoutError('Gemma competence repair still incomplete; final collector 439987 remains scheduled')
