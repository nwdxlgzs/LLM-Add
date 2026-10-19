"""Verify the embedded offline delivery and preserve a separate chat report."""
import ast
import argparse
import base64
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess

WEB = Path(__file__).resolve().parent
ROOT = WEB.parents[1]


class Scripts(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.blocks = {}
        self.current = None
        self.external = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script":
            self.current = attrs.get("id", "app")
            assert self.current not in self.blocks
            self.blocks[self.current] = ""
        if tag in ("script", "link", "img"):
            url = attrs.get("src") or attrs.get("href")
            if url and not url.startswith(("data:", "#", "blob:")):
                self.external.append(url)

    def handle_data(self, data):
        if self.current:
            self.blocks[self.current] += data

    def handle_endtag(self, tag):
        if tag == "script":
            self.current = None


def fingerprint(text):
    data = text.encode("utf-16-le")
    value = 2166136261
    for i in range(0, len(data), 2):
        value = ((value ^ (data[i] | data[i + 1] << 8)) * 16777619) & 0xFFFFFFFF
    return {"length": len(data) // 2, "fnv1a_utf16": f"{value:08x}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--browser-report', default='browser-chat.json')
    parser.add_argument('--report', default='validation-chat-delivery.json')
    parser.add_argument('--native-log', default='validation-chat-native.log')
    parser.add_argument('--asan-log', default='validation-chat-asan.log')
    parser.add_argument('--wasm-report', default='validation-chat.json')
    parser.add_argument('--checks', type=int, default=14)
    args = parser.parse_args()
    page = Scripts()
    page.feed((WEB / "dist/ai-add.html").read_text(encoding="utf-8"))
    assert set(page.blocks) == {"payload", "engines", "worker-source", "app"}
    assert not page.external, page.external
    payload = json.loads(page.blocks["payload"])
    models = []
    for name, item in payload["models"].items():
        data = base64.b64decode(item["base64"], validate=True)
        digest = hashlib.sha256(data).hexdigest()
        folder = WEB / 'build/column-v2-export' if name.startswith('column-v2-') else WEB.parent / 'run-final'
        assert data == (folder / name).read_bytes()
        assert digest == item["sha256"] and len(data) == item["bytes"]
        models.append({"file": name, "bytes": len(data), "sha256": digest})
    assert len(models) == 18
    assert sum(item['file'].startswith('column-v2-') for item in models) == 9
    assert payload['structured_training'] == (WEB / 'build/column-v2-run/progress.csv').read_text(encoding='utf-8')
    engines = json.loads(page.blocks["engines"])
    for name, source in engines.items():
        assert source == (WEB / "build" / f"engine-{name}.js").read_text(encoding="utf-8")
    assert set(engines) == {"simd", "scalar"}
    assert json.loads(page.blocks["worker-source"]) == (WEB / "worker.js").read_text(encoding="utf-8")
    assert page.blocks["app"] == "\n".join((WEB / name).read_text(encoding="utf-8") for name in ("app.js", "datasets.js"))
    for source in [page.blocks["app"], json.loads(page.blocks["worker-source"]), *engines.values()]:
        subprocess.run(["node", "--check", "-"], input=source, text=True, encoding="utf-8", check=True)
    for name in ("build.py", "verify-chat-delivery.py"):
        ast.parse((WEB / name).read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (WEB / "build/chat-dataset-export.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 512
    for row in rows:
        user, assistant = row["messages"]
        assert (user["role"], assistant["role"]) == ("user", "assistant")
        assert "<think>" not in user["content"] and "</think>" not in user["content"]
        assert assistant["content"].count("<think>") == assistant["content"].count("</think>") == 1
    bundle = json.loads((WEB / "build/browser-chat-stage6.aiadd").read_text(encoding="utf-8-sig"))
    assert bundle["protocol"] == "chat-v1" and bundle["cot"] is False and bundle["stage"] == 6
    assert len(base64.b64decode(bundle["model_base64"], validate=True)) > 1000
    for name in (args.native_log, args.asan_log):
        log = (WEB / "build" / name).read_text(encoding="utf-8", errors="replace")
        assert log.count(" Testing: ") == args.checks and "Test Failed" not in log
        assert log.count("Test Passed.") + log.count("Test Pass Reason:") == args.checks
    wasm = json.loads((WEB / 'build' / args.wasm_report).read_text(encoding="utf-8"))
    assert wasm["passed"] and len(wasm["reports"]) in (2, 6)
    report = {"passed": True, "models": models, "engines": list(engines), "external_resources": page.external,
              "script_fingerprints": {key: fingerprint(value) for key, value in page.blocks.items()},
              "syntax_checks": "node --check on app, worker, both engines; Python AST",
              "exported_dataset_rows": len(rows), "native_checks": args.checks, "asan_checks": args.checks,
              "chat_v1_convergence": "not established; retained as a comparison"}
    browser = json.loads((WEB / 'build' / args.browser_report).read_text(encoding="utf-8"))
    assert browser["script_fingerprints"] == report["script_fingerprints"]
    if 'model_picker' in browser:
        picker = browser['model_picker']
        expected = [f'column-v2-stage-{i}.cn' for i in range(1, 6)] + ['column-v2-final-fp4.cn']
        assert [row['value'] for row in picker['before']['builtin']] == expected
        assert len(picker['before']['mine']) == picker['before']['snapshots']
        assert not picker['empty']['mine'] and picker['empty']['snapshots'] == 0
        assert len(picker['empty']['builtin']) == picker['empty']['options'] == 6
        assert picker['empty']['selected'] == expected[-1]
        assert '正确' in picker['empty']['answerCheck']
        assert picker['afterUndo']['snapshots'] == picker['before']['snapshots']
        assert picker['afterReload']['snapshots'] == picker['before']['snapshots']
        assert len(picker['afterReload']['builtin']) == 6
        assert len(picker['afterReload']['mine']) == picker['before']['snapshots']
        assert set(row['value'] for row in picker['afterReload']['mine']) == set(row['value'] for row in picker['before']['mine'])
        assert len(picker['stageCards']) == 6 and not any('旧协议' in text for text in picker['stageCards'])
        report['model_picker'] = {'builtin_stages': 6, 'user_checkpoints': picker['afterReload']['snapshots'],
                                  'delete_all_leaves_only_six_builtins': True, 'undo_and_reload_preserve_user_models': True}
    elif 'column' in browser:
        column = browser['column']
        assert column['afterTraining'] - column['beforeTraining'] == 19
        assert column['afterReload'] == column['afterTraining']
        assert column['downloadedReport']['protocol'] == 'column-v2'
        assert column['downloadedReport']['snapshotCount'] == column['afterReload']
        exported_report = json.loads((WEB / 'build/browser-column-experiment.json').read_text(encoding='utf-8-sig'))
        assert exported_report['structured_training_csv'] == payload['structured_training']
        assert len(column['newEntries']) == 19 and all('结构化逐位' in row for row in column['newEntries'])
        assert '未达标' in column['trainingLabel']
        assert column['ten']['answer'] == '19999999998' and '正确' in column['ten']['check']
        assert '错误' in column['early']['check']
        assert '正确' in column['final']['check'] and '结构化逐位' in column['final']['note']
        assert all(row['overflow'] == 'auto' for row in column['scroll'])
        assert any(row['scroll'] > row['client'] for row in column['scroll'])
        structured = [json.loads(line) for line in (WEB / 'build/column-dataset-export.jsonl').read_text(encoding='utf-8-sig').splitlines()]
        assert len(structured) == 200
        saved = json.loads((WEB / 'build/browser-column-stage6.aiadd').read_text(encoding='utf-8-sig'))
        assert saved['protocol'] == 'column-v2' and saved['cot'] is True and saved['stage'] == 6
        assert len(base64.b64decode(saved['model_base64'], validate=True)) > 1000
        for row in structured:
            user, assistant = row['messages']
            assert (user['role'], assistant['role']) == ('user', 'assistant')
            assert '<think>' not in json.dumps(row) and '</think>' not in json.dumps(row)
            fields = dict(part.split('=') for part in user['content'].split()[1:])
            total = int(fields['a']) + int(fields['b']) + int(fields['cin'])
            assert assistant['content'] == f'{total % 10} {total // 10}'
        parity = json.loads((WEB / 'build/parity-v2/report.json').read_text(encoding='utf-8'))
        assert parity['passed'] and len(parity['results']) == 16
        assert all(row['wasm_to_native_roundtrip'] for row in parity['results'])
        assert all(match['informative_steps'] > 0 for row in parity['results'] if row['name'] == 'cot_rloo' for match in row['nativeMatches'])
        assert len(wasm['reports']) == 6
        for suffix in ('fp32', 'fp8', 'fp4'):
            evaluation = json.loads((WEB / f'build/column-v2-export/evaluation-{suffix}.json').read_text(encoding='utf-8'))
            assert evaluation['passed'] and len(evaluation['backends']) == 8
            assert all(item['correct'] == item['total'] == 200 for item in evaluation['backends'])
            assert evaluation['carry_1000_digits']
            if suffix == 'fp4':
                assert sum(item['total'] for item in evaluation['random_tests']) == 10000
                assert all(item['correct'] == item['total'] for item in evaluation['random_tests'])
        report['column_v2'] = {'dataset_rows': len(structured), 'parity_cases': 16, 'wasm_precision_backend_pairs': 6,
                               'native_fp4_random_problems': 10000, 'preserved_browser_snapshots': column['afterReload']}
    elif 'repair' in browser:
        repair = browser['repair']
        assert repair['deleted']['count'] == repair['beforeDelete'] - 1
        assert not repair['deleted']['optionExists'] and repair['deleted']['undo']
        assert repair['afterUndo'] == repair['beforeDelete']
        assert repair['afterReload'] == repair['afterTraining']
        assert repair['afterTraining'] - repair['beforeTraining'] == 19
        assert all(row['overflow'] == 'auto' for row in repair['scroll'])
        assert any(row['scroll'] > row['client'] for row in repair['scroll'])
        assert '未达标' in repair['trainingLabel']
    else:
        assert browser["finalImport"]["cot"] is False and "完整对话自回归" in browser["finalImport"]["note"]
        assert browser["finalDataset"]["restoredSnapshots"] == 34
    assert not browser['logs']
    report["browser_dom_matches_delivery"] = True
    path = WEB / 'build' / args.report
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(models)} preserved models, both engines, all embedded scripts and chat exports: {path}")


if __name__ == "__main__":
    main()
