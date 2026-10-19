"""Finish the winning <=500k model and publish actual seven-stage artifacts."""
import json,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WEB=ROOT/'web';OUT=WEB/'build/outcome-v6';RUN=WEB/'build/dialogue-v6-run'
while not (OUT/'comparison-final.json').exists():time.sleep(10)
comparison=json.loads((OUT/'comparison-final.json').read_text())
spec=json.loads((OUT/'spec.json').read_text());candidate=next(c for c in spec['candidates'] if c['name']==comparison['winner'])
if RUN.exists():raise RuntimeError('Do not overwrite an existing v6 training run')
shutil.copytree(OUT/(candidate['name']+'-run'),RUN)
recipe=dict(spec);recipe.update(winner=candidate,budgets=[300,1200,2000,8000,200,600,400],cot=True,protocol='chat-v2',curriculum_resume='4000 completed attempts, selected pilot policy and optimizer retained; new deterministic RNG stream; remaining mixed-length updates continue to 8000',phase7='Outcome-only on-policy RLOO: full-vocabulary sampling, no internal scratchpad checking or teacher targets. Frozen reference; skip equal-reward groups. Greedy final-answer selection.')
(RUN/'run-spec.json').write_text(json.dumps(recipe,indent=2))
subprocess.run(['cmake','--build',str(ROOT/'build'),'--target','chat_run','-j4'],check=True)
cmd=[str(ROOT/'build/chat_run.exe'),str(RUN),'1','0','0','0','8000','200','600','4',str(RUN/'stage-4.cn'),'4','3','4',str(candidate['dim']),str(candidate['ff']),str(candidate['layers']),'400','4000']
with (OUT/'finish.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
for script,args in [('publish-dialogue.py',[]),('test-dialogue-native.py',[]),('test-dialogue-native.py',['--stage','6']),('test-dialogue-native.py',['--variant','last'])]:
 suffix='-last' if '--variant' in args else '-before' if args else ''
 with (OUT/(script+suffix+'.log')).open('w') as log:subprocess.run(['python',str(WEB/script),'--version','v6',*args],stdout=log,stderr=subprocess.STDOUT,check=True)
for script,args in [('test-parity.cjs',['--outcome','--fields']),('test-dialogue-wasm.cjs',[])]:
 with (OUT/(script+'.log')).open('w') as log:
  rc=subprocess.run(['node',str(WEB/script),*args],stdout=log,stderr=subprocess.STDOUT).returncode
 if rc and not (WEB/('build/parity-v6/report.json' if 'parity' in script else 'build/validation-outcome-wasm.json')).exists():raise RuntimeError(script+' failed without report')
(OUT/'training-complete.json').write_text(json.dumps(dict(completed=True,winner=candidate),indent=2))
print('Seven stages, native tests, WASM generation and numeric checks finished.')
