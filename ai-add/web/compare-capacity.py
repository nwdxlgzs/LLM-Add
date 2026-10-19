"""Compare held-out whole-dialogue outputs; arithmetic is evaluation only."""
import json
from pathlib import Path
import importlib.util

WEB=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('dialogue_tests',WEB/'test-dialogue-native.py')
evaluation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def summarize(rows):
    total=sum(max(len(r['a']),len(r['b'])) for r in rows)
    audits=[evaluation.audit_fields(r['a'],r['b'],r['tokens']) for r in rows]
    result={k:sum(a[k] for a in audits) for k in ('parsed','position','operands','incoming_carry','transition','full_step')}
    complete=[r for r,a in zip(rows,audits) if a['lines']==a['full_step']==max(len(r['a']),len(r['b']))]
    result.update(questions=len(rows),correct=sum(r['correct'] for r in rows),expected_columns=total,
                  all_steps_correct=len(complete),answer_correct_when_all_steps_correct=sum(r['correct'] for r in complete))
    return result


def main():
    old=json.loads((WEB/'build/dialogue-v4-tests/report.json').read_text(encoding='utf-8'))
    new=json.loads((WEB/'build/dialogue-v5-tests/report.json').read_text(encoding='utf-8'))
    assert old['seed']==new['seed'] and old['protocol']==new['protocol']=='chat-v2'
    assert [(r['name'],r['a'],r['b']) for r in old['results']]==[(r['name'],r['a'],r['b']) for r in new['results']]
    report=dict(scope='Same random questions, protocol and unconstrained greedy generation; no correction or fallback.',
                old_parameters=57388,new_parameters=550556,seed=new['seed'],groups={},examples={})
    for width in (1,2,4,6,10,12,20):
        report['groups'][str(width)]={name:summarize([r for r in source['results'] if r['name'].startswith(f'w{width}-')])
                                     for name,source in [('old',old),('new',new)]}
    for name in ('complaint','fields','carry10','max10','short'):
        report['examples'][name]={label:dict(answer=r['answer'],correct=r['correct'],audit=evaluation.audit_fields(r['a'],r['b'],r['tokens']))
                                 for label,source in [('old',old),('new',new)] for r in source['results'] if r['name']==name}
    (WEB/'build/capacity-v5/comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for w,g in report['groups'].items():
        print(f"{w} digits: answer {g['old']['correct']}/20 -> {g['new']['correct']}/20; copied operand pairs {g['old']['operands']}/{g['old']['expected_columns']} -> {g['new']['operands']}/{g['new']['expected_columns']}")


if __name__=='__main__': main()
