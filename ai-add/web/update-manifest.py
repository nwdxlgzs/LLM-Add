"""Refresh hashes, excluding only explicitly recorded privacy-cleanup removals."""
import hashlib
import json
from pathlib import Path

WEB=Path(__file__).resolve().parent
ROOT=WEB.parents[1]


def main():
    target=WEB/'dist/sha256.json'
    old=json.loads(target.read_text(encoding='utf-8'))
    cleanup_path=WEB/'build/privacy-cleanup.json'
    cleanup=json.loads(cleanup_path.read_text(encoding='utf-8')) if cleanup_path.exists() else {}
    removed=set(cleanup.get('removed_files',[]))
    for relative in removed:
        path=ROOT/relative
        assert path.resolve().parent==(WEB/'build').resolve()
        assert path.name.startswith('browser-') and path.suffix in ('.json','.png','.jpg','.jpeg')
        assert not path.exists(), f'Cleanup removal still exists: {relative}'
    paths={ROOT/r['path'] for r in old['files'] if r['path'] not in removed}
    for folder in ('include','src','tools','tests'):
        paths.update(p for p in (WEB.parent/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.update(p for p in WEB.iterdir() if p.is_file())
    for name in ('dialogue-v5-run','dialogue-v5-export','dialogue-v5-tests','capacity-v5','parity-v5','dialogue-v6-run','dialogue-v6-export','dialogue-v6-tests','dialogue-v6-tests-stage-6','dialogue-v6-tests-last','outcome-v6','parity-v6'):
        paths.update(p for p in (WEB/'build'/name).rglob('*') if p.is_file())
    paths.update(p for p in (WEB/'build').glob('*outcome*') if p.is_file())
    if cleanup_path.exists(): paths.add(cleanup_path)
    paths.add(WEB/'dist/ai-add.html')
    paths.discard(target)
    records=[]
    for path in sorted(paths):
        data=path.read_bytes()
        records.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    old['files']=records
    old['current_delivery']='dialogue-v6 / chat-v2; seven stages and final-answer-only scratchpad RL; arithmetic and FP4 numerical failures retained'
    old['latest_delivery']=old['current_delivery']
    target.write_text(json.dumps(old,indent=2),encoding='utf-8')
    for row in records:
        data=(ROOT/row['path']).read_bytes()
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    print(f"Verified {len(records)} artifacts; excluded {len(removed)} explicitly recorded privacy removals.")


if __name__=='__main__':main()
