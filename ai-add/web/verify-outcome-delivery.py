"""Verify offline v6 bytes, recorded tests and browser-visible controls, independently of task accuracy."""
import base64,hashlib,json,subprocess,importlib.util
from pathlib import Path
WEB=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('delivery_common',WEB/'verify-dialogue-delivery.py');common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
page=common.Page();page.feed((WEB/'dist/ai-add.html').read_text(encoding='utf-8'))
assert not page.external
payload=json.loads(page.blocks['payload']);assert payload['model_prefix']=='dialogue-v6'
expected=[f'dialogue-v6-stage-{i}.cn' for i in range(1,8)]
assert list(payload['models'])==expected and all(s['parameters']<=500000 for s in payload['dialogue_stages'])
assert payload['dialogue_stages'][6]['validation_kind']=='greedy_final_answer'
for name,model in payload['models'].items():
 b=base64.b64decode(model['base64'],validate=True)
 assert b==(WEB/'build/dialogue-v6-export'/name).read_bytes()
 assert len(b)==model['bytes'] and hashlib.sha256(b).hexdigest()==model['sha256']
assert len(payload['variants'])==2
for name,model in payload['variants'].items():
 b=base64.b64decode(model['base64'],validate=True)
 assert b==(WEB/'build/dialogue-v6-export'/name).read_bytes() and hashlib.sha256(b).hexdigest()==model['sha256']
assert payload['dialogue_training']==(WEB/'build/dialogue-v6-run/progress.csv').read_text()
assert payload['dialogue_stages']==json.loads((WEB/'build/dialogue-v6-export/stages.json').read_text())
assert all(c['parameters']<=500000 for c in payload['outcome_comparison']['candidates'])
for variant,engine in json.loads(page.blocks['engines']).items():assert engine==(WEB/f'build/engine-{variant}.js').read_text()
assert json.loads(page.blocks['worker-source'])==(WEB/'worker.js').read_text(encoding='utf-8')
assert page.blocks['app']=='\n'.join((WEB/n).read_text(encoding='utf-8') for n in ('app.js','datasets.js'))
for code in [page.blocks['app'],json.loads(page.blocks['worker-source']),*json.loads(page.blocks['engines']).values()]:subprocess.run(['node','--check','-'],input=code,text=True,encoding='utf-8',check=True)
browser=json.loads((WEB/'build/validation-outcome-browser.json').read_text(encoding='utf-8'))
for state in (browser['before'],browser['after']):
 assert state['identifier_format']=='sha256'
 assert all(len(identifier)==64 and all(c in '0123456789abcdef' for c in identifier) for identifier in state['models'])
assert browser['script_fingerprints']=={k:common.fingerprint(v) for k,v in page.blocks.items()}
assert browser['before']['models']==browser['after']['models'] and browser['before']['snapshots']==browser['after']['snapshots']
assert browser['after']['builtin']==expected
assert '无固定' in browser['data_rl']['output_label'] and '只检查' in browser['data_rl']['objective']
assert '<think>' not in browser['data_rl']['input']
assert browser['manual_rl']['samples']==4 and browser['manual_rl']['stage']==7
assert '最终答案' in browser['manual_rl']['metric']
assert browser['shape_limit']['rejected'] and browser['shape_limit']['unchanged']
assert browser['cot_off']['no_think']
assert browser['play_correct']['raw'].startswith('<|im_start|>user\n')
assert browser['play_long']['a_length']>10 and browser['play_long']['check']
assert browser['early']['raw'] and browser['early']['check']
assert all(r['overflow'] in ('auto','scroll') for r in browser['scroll'])
assert not browser['errors']
wasm=json.loads((WEB/'build/validation-outcome-wasm.json').read_text());assert wasm['protocol_passed']
parity=json.loads((WEB/'build/parity-v6/report.json').read_text());assert parity['complete']
for n in ('validation-outcome-native.log','validation-outcome-asan.log','validation-outcome-chat.log','validation-outcome-chat-asan.log'):assert '100% tests passed' in (WEB/'build'/n).read_text(encoding='utf-8')
report=dict(delivery_passed=True,scope='Offline delivery, protocol and actual training checks; arithmetic and numeric divergence are reported independently.',parameters=payload['dialogue_stages'][6]['parameters'],stage_count=7,preserved_checkpoints=browser['after']['snapshots'],all_backend_numeric_parity=parity['passed'],wasm_generation_equivalence=wasm['generation_equivalence_passed'],scores=payload['dialogue_test'],winner=payload['outcome_comparison']['winner'],browser=browser['script_fingerprints'])
(WEB/'build/validation-outcome-delivery.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
