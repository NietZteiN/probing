from pathlib import Path
import re,json,hashlib,shutil
ROOT=Path('/work/jvl210002/migration')
TITLE='Chain of Thought Reduces Errors from Misleading Names in Arithmetic'
OLD='Chain of Thought Limits Lure Answers in Arithmetic'
phrases=[
 ('numeric lure answers','answers suggested by misleading names'),
 ('written-lure excess','excess name errors'),('written-lure rates','name-error rates'),('written-lure rate','name-error rate'),
 ('lure-writing augmentation','additional name-error cases'),('lure-error augmentation','additional name-error cases'),
 ('lure-writing cases','name-error cases'),('lure-writing errors','name errors'),('lure-writing chains','chains with name errors'),
 ('lure-writing case','name-error case'),('lure writing','name errors'),
 ('alternative-lure','alternative-name'),('lure-equivalence','name-error equivalence'),
 ('lure-removal','error-removal'),('lure-rate','name-error-rate'),
 ('lure excess','excess name errors'),('lure errors','name errors'),('lure answers','name errors'),('lure writes','name errors'),('lure write','name error'),
 ('lure predictions','name-suggested predictions'),('lure contrasts','name-error contrasts'),('lure effects','name-error effects'),('lure effect','name-error effect'),
 ('lure values','name-suggested values'),('lure value','name-suggested value'),('lure digit','name-suggested digit'),
 ('lure operations','name-suggested operations'),('lure operation','name-suggested operation'),
 ('lure readout','readout of the name-suggested value'),('lure rate','name-suggested prediction rate'),('lure mass','probability of the name-suggested value'),
 ('readout--lure','readout--error'),('readout–lure','readout–error'),
 ('new lure: removes old','alternative name: removes error'),('new lure: followed','alternative value: followed'),
 ('neutral: removes lure','neutral: removes error'),('log p(lure)','log p(name value)'),('red = lure','red = name-suggested'),('lure err ','name error '),
 ('Sample lure probe','Sample error probe'),('Extra lure probe','Extra error probe'),('Extra lure $n$','Extra errors $n$'),
 ('Sample lure change','Sample error change'),('Extra lure change','Extra error change'),
 ('CoT lure','CoT name errors'),('Direct lure','Direct name errors'),('Direct: lure','Direct: errors'),
 ('helps & lures','helps & errors'),
]
def multi(s):
 s=s.replace(OLD,TITLE)
 for a,b in phrases:
  s=re.sub(re.escape(a),lambda m:b[0].upper()+b[1:] if m[0][0].isupper() else b,s,flags=re.I)
 return s
protect=re.compile(r'\\(?:NUM|ref|label|input|tabinput|IfFileExists|cite\w*|includegraphics(?:\[[^\]]*\])?)\{[^{}]*\}|(?:[\w./-]*lure[\w./-]*\.(?:pdf|png|py|json))')
def prose(s):
 saved=[]
 def stash(m):saved.append(m[0]);return f'ZZPROTECTED{len(saved)-1}ZZ'
 s=protect.sub(stash,s)
 s=multi(s)
 s=s.replace(r'\texttt{answered\_lure \textasciitilde{} incongruent + C(seed)}',r'\texttt{name\_error \textasciitilde{} incongruent + C(seed)}')
 s=s.replace(r'\mathrm{lure}',r'\mathrm{name}')
 s=re.sub(r'\blures\b','name-suggested values',s,flags=re.I)
 s=re.sub(r'\blure\b',lambda m:'Name-suggested value' if m[0][0].isupper() else 'name-suggested value',s)
 for i,v in enumerate(saved):s=s.replace(f'ZZPROTECTED{i}ZZ',v)
 return s
scripts={
 'probing':['51_tables','52_figs','53_kudo_figs','54_grid_figs','58_token_figure','60_master_figure','66_figure_dump','79_round5_tables','82_round6_tables'],
 'codecue':['51_numbers','77_followup_tables','80_round6_tables']}
record={'before_sha256':{},'modified':[],'summary_before_sha256':{}}
for repo in scripts:
 log=ROOT/repo/'log/terminology_2026-10-06'
 for p in (ROOT/repo/'results/summary').rglob('*.json'):
  record['summary_before_sha256'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 paths=[ROOT/repo/'paper/main.tex',ROOT/repo/'paper/ARGUMENT.md',ROOT/repo/'paper/README.md',ROOT/repo/'README.md',ROOT/repo/'paper/refs.bib',ROOT/repo/'paper/numbers.tex']
 paths+=list((ROOT/repo/'paper/tables').glob('*.tex'))
 paths+=[ROOT/repo/'scripts'/f'{stem}.py' for stem in scripts[repo]]
 for p in paths:
  old=p.read_text();new=multi(old) if p.suffix=='.py' else prose(old) if p.name!='numbers.tex' else old
  record['before_sha256'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
  dest=log/'before'/p.relative_to(ROOT/repo);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
  if new!=old:p.write_text(new);record['modified'].append(str(p.relative_to(ROOT)))
(ROOT/'probing/log/terminology_2026-10-06/edit_record.json').write_text(json.dumps(record,indent=2)+'\n')
print('Updated',len(record['modified']),'presentation sources. Backups and numeric-input hashes saved.')
