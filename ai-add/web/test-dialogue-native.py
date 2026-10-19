"""Independent whole-answer greedy tests. Arithmetic is only a scorer here."""
import argparse
import json
from pathlib import Path
import random
import re
import subprocess

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent

SPECIAL = {256:'<|im_start|>',257:'<|im_end|>',258:'<|endoftext|>',259:'<think>',260:'</think>'}


def decode(tokens):
    return ''.join(SPECIAL.get(t, chr(t) if t == 10 or 32 <= t < 127 else f'\\x{t:02x}') for t in tokens)


def audit_fields(a,b,tokens):
    """Diagnose generated fields; never feed this arithmetic back to inference."""
    text=decode(tokens)
    body=text.split('<think>',1)[-1].split('</think>',1)[0] if '<think>' in text else ''
    lines=body.strip().splitlines()
    result=dict(lines=len(lines),parsed=0,position=0,operands=0,incoming_carry=0,transition=0,full_step=0)
    carry=0
    reference=[]
    for i in range(max(len(a),len(b))):
        ea=int(a[-1-i]) if i<len(a) else 0
        eb=int(b[-1-i]) if i<len(b) else 0
        reference.append((ea,eb,carry))
        carry=(ea+eb+carry)//10
    for i,line in enumerate(lines):
        m=re.fullmatch(r'pos=(\d+) a=(\d) b=(\d) cin=([01]) -> (\d) ([01])',line)
        if not m or i>=max(len(a),len(b)):continue
        pos,da,db,ci,d,co=map(int,m.groups())
        ea,eb,carry=reference[i]
        result['parsed']+=1
        checks=dict(position=pos==i,operands=(da,db)==(ea,eb),incoming_carry=ci==carry,transition=d+10*co==da+db+ci)
        for key,correct in checks.items():result[key]+=int(correct)
        result['full_step']+=int(all(checks.values()))
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--version',choices=('v3','v4','v5','v6'),default='v6')
    ap.add_argument('--stage',type=int)
    ap.add_argument('--variant',choices=('selected','entry','last'),default='selected')
    args=ap.parse_args()
    selected_stage=args.stage or (7 if args.version=='v6' else 6)
    if args.variant!='selected' and (args.version!='v6' or selected_stage!=7):
        ap.error('Entry/last comparison is available for dialogue-v6 stage 7')
    prefix='dialogue-'+args.version
    protocol='chat-v1' if args.version=='v3' else 'chat-v2'
    extra=['--compact'] if args.version=='v3' else []
    variant_suffix='' if args.variant=='selected' else '-'+args.variant
    OUT=WEB/('build/'+prefix+('-tests' if args.stage is None else f'-tests-stage-{selected_stage}')+variant_suffix)
    MODEL=WEB/f'build/{prefix}-export/{prefix}-stage-{selected_stage}{variant_suffix}.cn'
    OUT.mkdir(exist_ok=True)
    rng = random.Random(938234)
    cases = [('complaint', '7140345462', '1095133250'), ('carry10', '9999999999', '1'),
             ('max10', '9999999999', '9999999999'), ('short', '99', '1'),
             ('fields', '90', '70')]
    for width in (1,2,4,6,10,12,20):
        for i in range(20):
            lo = 10**(width-1) if width > 1 else 0
            cases.append((f'w{width}-{i}', str(rng.randrange(lo, 10**width)), str(rng.randrange(lo, 10**width))))
    cases.append(('rollover', '9'*100, '1'))
    results = []
    for name,a,b in cases:
        subprocess.run([str(ROOT / 'build/chat_eval.exe'),str(MODEL),'1',str(OUT),name,a,b,*extra],check=True)
        p=json.loads((OUT/(name+'.json')).read_text())
        raw=(OUT/(name+'.txt')).read_text(encoding='utf-8')
        prompt=f'<|im_start|>user\n{a}+{b}<|im_end|>\n<|im_start|>assistant\n'
        assert decode(p['input_ids']) == prompt
        assert raw == prompt + decode(p['tokens'])
        assert p['protocol']==protocol and p['columns']==[]
        assert p['message_complete'] == (p['stop']=='endoftext')
        assert '# problem' not in raw and 'scheduled' not in raw and '[stop=' not in raw
        expected=str(int(a)+int(b))
        result=dict(name=name,width=max(len(a),len(b)),a=a,b=b,expected=expected,
                    correct=p['valid'] and p['answer'].lstrip('0')==expected.lstrip('0'), **p)
        # Retain old SFT diagnostics only for historical stages. Outcome RL
        # neither evaluates nor reports the scratchpad's internal algorithm.
        if protocol=='chat-v2' and selected_stage<7:
            result['field_audit']=audit_fields(a,b,p['tokens'])
        results.append(result)
    groups={str(w):{'correct':sum(r['correct'] for r in results if r['width']==w),
                    'total':sum(r['width']==w for r in results)} for w in sorted({r['width'] for r in results})}
    report=dict(passed=True,scope='Protocol and raw-token fidelity passed; task scores below are measured, not a convergence claim.',
                seed=938234,protocol=protocol,variant=args.variant,groups=groups,
                random_groups={str(w):dict(correct=sum(r['correct'] for r in results if r['name'].startswith(f'w{w}-')),total=20) for w in (1,2,4,6,10,12,20)},results=results)
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(groups,indent=2))


if __name__=='__main__':
    main()
