from pathlib import Path
import json,subprocess,time
root=Path.cwd();out=root/'ai-add/web/build/outcome-v6';c={'name':'wide64','layers':6,'dim':64,'ff':104,'parameters':467612}
spec=json.loads((out/'spec.json').read_text());spec['candidates'].append(c);(out/'spec.json').write_text(json.dumps(spec,indent=2))
d=out/'wide64-run';d.mkdir()
start=time.time()
cmd=[str(root/'ai-add/build/chat_run.exe'),str(d),'1','300','1200','2000','4000','0','0','4','new','1','3','4','64','104','6','0']
with (out/'wide64.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run([str(root/'ai-add/build/chat_eval.exe'),str(d/'stage-4.cn'),'1',str(out),'wide64','--outcome'],check=True)
r=json.loads((out/'wide64-metrics.json').read_text());r.update(name='wide64',seconds=time.time()-start);(out/'wide64-result.json').write_text(json.dumps(r,indent=2))
while not all((out/(n+'-result.json')).exists() for n in ['wide6','deep10']):time.sleep(10)
results=[json.loads((out/(n+'-result.json')).read_text()) for n in ['wide6','deep10','wide64']]
winner=max(results,key=lambda r:(r['correct'],-r['seconds']))
comparison=dict(candidates=results,winner=winner['name'],selection=spec['selection'])
(out/'comparison-final.json').write_text(json.dumps(comparison,indent=2));(out/'comparison.json').write_text(json.dumps(comparison,indent=2))
print('Final comparison ready: '+winner['name'])
