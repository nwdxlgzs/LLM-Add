from pathlib import Path
import subprocess,json,time
from concurrent.futures import ThreadPoolExecutor
root=Path.cwd(); out=root/'ai-add/web/build/outcome-v6';out.mkdir(exist_ok=True)
if (out/'spec.json').exists():raise RuntimeError('Existing pilots retained; use a fresh output directory/workspace')
budgets=[300,1200,2000,4000,0,0]
spec={'seed':20261018,'batch':4,'budgets':budgets,'selection':'greedy final-answer accuracy on 200 fixed validation problems, speed as tie-breaker','candidates':[{'name':'wide6','layers':6,'dim':56,'ff':128,'parameters':462412},{'name':'deep10','layers':10,'dim':40,'ff':104,'parameters':448788},{'name':'wide64','layers':6,'dim':64,'ff':104,'parameters':467612}]}
(out/'spec.json').write_text(json.dumps(spec,indent=2))
def run(c):
 d=out/(c['name']+'-run');d.mkdir(exist_ok=True)
 cmd=[str(root/'ai-add/build/chat_run.exe'),str(d),'1',*map(str,budgets),'4','new','1','3','4',str(c['dim']),str(c['ff']),str(c['layers']),'0']
 start=time.time()
 with (out/(c['name']+'.log')).open('w') as f: subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 subprocess.run([str(root/'ai-add/build/chat_eval.exe'),str(d/'stage-4.cn'),'1',str(out),c['name'],'--outcome'],check=True)
 r=json.loads((out/(c['name']+'-metrics.json')).read_text());r.update(name=c['name'],seconds=time.time()-start)
 (out/(c['name']+'-result.json')).write_text(json.dumps(r,indent=2));return r
with ThreadPoolExecutor(3) as pool: results=list(pool.map(run,spec['candidates']))
winner=max(results,key=lambda r:(r['correct'],-r['seconds']))
(out/'comparison.json').write_text(json.dumps({'candidates':results,'winner':winner['name'],'selection':spec['selection']},indent=2))
(out/'comparison-final.json').write_text((out/'comparison.json').read_text())
print(json.dumps({'completed':True,'winner':winner['name']}))
