"""Export real whole-dialogue weights, stage metadata and self-describing bundles."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent



def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--version',choices=('v3','v4','v5','v6'),default='v6')
    args=ap.parse_args()
    prefix='dialogue-'+args.version
    protocol='chat-v1' if args.version=='v3' else 'chat-v2'
    extra=['--compact'] if args.version=='v3' else []
    RUN=WEB/('build/'+prefix+'-run')
    OUT=WEB/('build/'+prefix+'-export')
    OUT.mkdir(exist_ok=True)
    subprocess.run([str(ROOT / 'build/chat_export.exe'), str(RUN), str(OUT), prefix, str(7 if args.version=='v6' else 6)], check=True)
    rows = list(csv.DictReader((RUN / 'progress.csv').open(encoding='utf-8', newline='')))
    stages = []
    import base64
    for stage in range(1, 8 if args.version=='v6' else 7):
        name = f'{prefix}-stage-{stage}'
        subprocess.run([str(ROOT / 'build/chat_eval.exe'), str(OUT / (name+'.cn')), '1', str(OUT), name, *extra, *(['--outcome'] if stage==7 else [])], check=True)
        metrics = json.loads((OUT / (name+'-metrics.json')).read_text())
        last = [r for r in rows if int(r['stage']) == stage][-1]
        stages.append(dict(stage=stage, validation_kind='greedy_final_answer' if stage==7 else 'teacher_forced_exact', steps=int(last['step']), best_step=int(last['best_step']),
                           last_correct=int(last['last_correct']), **metrics))
        for variant in ('', '-last', '-entry'):
            source = RUN / f'stage-{stage}{variant}.cn'
            # Preserve optimizer state, protocol and variant for further experiments.
            bundle = dict(format='aiadd-experiment-v1', protocol=protocol, cot=True,
                          stage=stage, variant=variant or '-selected',
                          model_base64=base64.b64encode(source.read_bytes()).decode())
            source.with_suffix('.aiadd').write_text(json.dumps(bundle), encoding='utf-8')
        bundle['variant'] = '-selected'
        bundle['model_base64'] = base64.b64encode((OUT / (name+'.cn')).read_bytes()).decode()
        (OUT / (name+'.aiadd')).write_text(json.dumps(bundle), encoding='utf-8')
    (OUT / 'stages.json').write_text(json.dumps(stages, indent=2), encoding='utf-8')
    print(json.dumps(stages, indent=2))


if __name__ == '__main__':
    main()
