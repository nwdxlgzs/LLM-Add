"""Check the actual offline payload and browser-visible dialogue evidence."""
import base64
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess

WEB=Path(__file__).resolve().parent


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.blocks={};self.key=None;self.external=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='script':
            self.key=a.get('id','app');self.blocks[self.key]=''
        if tag in ('script','link','img'):
            u=a.get('src') or a.get('href')
            if u and not u.startswith(('data:','#','blob:')):self.external.append(u)
    def handle_data(self,text):
        if self.key:self.blocks[self.key]+=text
    def handle_endtag(self,tag):
        if tag=='script':self.key=None


def fingerprint(s):
    b=s.encode('utf-16-le');h=2166136261
    for i in range(0,len(b),2):h=((h^(b[i]|b[i+1]<<8))*16777619)&0xffffffff
    return dict(length=len(b)//2,fnv1a_utf16=f'{h:08x}')


def main():
    page=Page();page.feed((WEB/'dist/ai-add.html').read_text(encoding='utf-8'))
    assert set(page.blocks)=={'payload','engines','worker-source','app'} and not page.external
    payload=json.loads(page.blocks['payload'])
    prefix=payload['model_prefix']
    capacity=prefix=='dialogue-v5'
    expected=[f'{prefix}-stage-{i}.cn' for i in range(1,7)]
    assert list(payload['models'])==expected
    for name,m in payload['models'].items():
        data=base64.b64decode(m['base64'],validate=True)
        assert data==(WEB/('build/'+prefix+'-export')/name).read_bytes()
        assert len(data)==m['bytes'] and hashlib.sha256(data).hexdigest()==m['sha256']
        assert m['protocol']=='chat-v2' and m['cot'] is True
    assert payload['dialogue_training']==(WEB/('build/'+prefix+'-run/progress.csv')).read_text()
    assert payload['dialogue_stages']==json.loads((WEB/('build/'+prefix+'-export/stages.json')).read_text())
    engines=json.loads(page.blocks['engines']);assert set(engines)=={'scalar','simd'}
    for v,source in engines.items():assert source==(WEB/f'build/engine-{v}.js').read_text()
    assert json.loads(page.blocks['worker-source'])==(WEB/'worker.js').read_text(encoding='utf-8')
    assert page.blocks['app']=='\n'.join((WEB/f).read_text(encoding='utf-8') for f in ('app.js','datasets.js'))
    for source in [page.blocks['app'],json.loads(page.blocks['worker-source']),*engines.values()]:
        subprocess.run(['node','--check','-'],input=source,text=True,encoding='utf-8',check=True)
    fingerprints={k:fingerprint(v) for k,v in page.blocks.items()}
    browser=json.loads((WEB/('build/browser-capacity.json' if capacity else 'build/browser-fields.json')).read_text(encoding='utf-8'))
    assert browser['script_fingerprints']==fingerprints
    assert browser['before']['models']==browser['after']['models']
    assert browser['after']['snapshots']==browser['before']['snapshots']
    assert browser['after']['builtin']==expected
    assert '完整对话自回归' in browser['complaint']['note']
    raw=browser['complaint']['raw']
    assert raw.startswith('<|im_start|>user\n7140345462+1095133250<|im_end|>\n<|im_start|>assistant\n')
    assert not any(s in raw for s in ('# pos=','# problem','scheduled','[stop='))
    assert browser['short']['raw'].startswith('<|im_start|>user\n99+1<|im_end|>')
    assert browser['data']['input'].count('<|im_start|>')==2
    assert '<think>' not in browser['data']['input']
    assert 'pos=0 a=' in browser['data']['output'] and ' cin=' in browser['data']['output'] and ' -> ' in browser['data']['output']
    assert 'digit=' not in browser['data']['output'] and 'cout=' not in browser['data']['output']
    assert browser['data']['output'].count('<think>')==browser['data']['output'].count('</think>')==1
    assert '<think>' not in browser['direct_data']['input']+browser['direct_data']['output']
    assert len(browser['random']['a'])==len(browser['random']['b'])==20
    assert browser['early']['check'] and browser['early']['raw']
    assert '输入 20 / 20 位' in browser['long']['note']
    assert all(r['overflow']=='auto' for r in browser['scroll'])
    assert not browser['errors']
    for n in (('validation-capacity-wasm.json',) if capacity else ('validation-fields-wasm.json','validation-fields-regression.json')):
        checks=json.loads((WEB/'build'/n).read_text())
        assert checks.get('protocol_passed',checks['passed'])
    parity=json.loads((WEB/('build/parity-v5/report.json' if capacity else 'build/parity-v4/report.json')).read_text())
    assert parity['complete']
    if not capacity: assert parity['backend_aligned_passed']
    # Keep the cross-backend QAT counterexample visible; do not call it a pass.
    assert parity['passed'] is False
    assert {r['name'] for r in parity['results'] if r.get('passed') is False}==({'chat_fp4_qat','trained_dialogue'} if capacity else {'chat_fp4_qat'})
    if capacity: assert parity['fp32_backend_aligned_passed']
    for n in (('validation-capacity-native.log','validation-capacity-asan.log') if capacity else ('validation-fields-native.log','validation-fields-asan.log')):
        assert '100% tests passed' in (WEB/'build'/n).read_text(encoding='utf-8')
    tests=json.loads((WEB/('build/'+prefix+'-tests/report.json')).read_text())
    report=dict(passed=True,scope='Delivery, raw-token fidelity and engine checks; task accuracy below is not a pass claim.',
                script_fingerprints=fingerprints,models=expected,external_resources=page.external,
                stage_metrics=payload['dialogue_stages'],independent_test_scores=tests['random_groups'],preserved_user_checkpoints=browser['after']['snapshots'],
                all_backend_numeric_parity=parity['passed'],backend_aligned_numeric_parity=parity['backend_aligned_passed'],
                fp32_backend_aligned_numeric_parity=parity.get('fp32_backend_aligned_passed'))
    (WEB/('build/validation-capacity-delivery.json' if capacity else 'build/validation-fields-delivery.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Dialogue delivery integrity passed; measured task scores retained.')


if __name__=='__main__':main()
