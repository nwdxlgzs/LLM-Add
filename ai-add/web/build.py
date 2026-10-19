"""Build a self-contained HTML. Python/Emscripten are developer dependencies only."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emsdk", type=Path, default=os.environ.get("EMSDK"))
    ap.add_argument("--output", type=Path, default=WEB / "dist" / "ai-add.html")
    ap.add_argument("--engines-only", action="store_true", help="Compile engines before training artifacts are published")
    ap.add_argument("--reuse-engines", action="store_true", help="Reuse already compiled engines for HTML-only edits")
    ap.add_argument("--version", choices=("v4", "v5", "v6"), default="v6")
    args = ap.parse_args()
    emcc = shutil.which("emcc") or shutil.which("emcc.bat")
    sdk = args.emsdk or (Path(emcc).resolve().parents[2] if emcc else None)
    if not sdk or not (sdk / "upstream/emscripten/emcc.py").is_file():
        ap.error("Emscripten SDK required to rebuild. Use --emsdk PATH; end users only need the HTML.")
    build = WEB / "build"
    build.mkdir(exist_ok=True)
    config = build / "emscripten_config.py"
    config.write_text(
        f"LLVM_ROOT = {(sdk / 'upstream/bin').as_posix()!r}\n"
        f"BINARYEN_ROOT = {(sdk / 'upstream').as_posix()!r}\n"
        f"NODE_JS = [{shutil.which('node')!r}]\n"
        f"PYTHON = {sys.executable!r}\n", encoding="utf-8")
    sources = [ROOT / "../cairn/src" / p for p in
               ["cairn.c", "quant.c", "tokenizer.c", "cpu.c", "cpu_wasm.c"]]
    sources += [ROOT / "src/add.c", ROOT / "src/train.c", ROOT / "src/experiment.c", ROOT / "src/chat.c",
                ROOT / "src/column.c", ROOT / "tools/engine-parity.c", WEB / "bridge.c"]
    exports = ["web_outcome_reward", "web_init_arch", "web_shape", "web_rollouts", "web_init", "web_load", "web_mode", "web_begin", "web_tick", "web_status", "web_certify",
               "web_infer", "web_predict", "web_save", "web_evaluate", "web_protocol", "web_cache", "web_example", "web_example_format",
               "web_column_example", "parity_run"]
    engines = {}
    for variant in ["simd", "scalar"]:
        target = build / f"engine-{variant}.js"
        cmd = [sys.executable, str(sdk / "upstream/emscripten/emcc.py"),
               "--em-config", str(config), *map(str, sources), "-I" + str(ROOT / "include"),
               "-I" + str(ROOT / "../cairn/include"), "-O3", "-std=c11", "-ffp-contract=off",
               "-sMODULARIZE=1", "-sEXPORT_NAME=createEngine", "-sENVIRONMENT=worker,node",
               "-sALLOW_MEMORY_GROWTH=1", "-sINITIAL_MEMORY=33554432", "-sMAXIMUM_MEMORY=536870912",
               "-sSTACK_SIZE=2097152", "-sSTACK_OVERFLOW_CHECK=2",
               "-sFILESYSTEM=1", "-sSINGLE_FILE=1", "-sNO_EXIT_RUNTIME=1",
               "-sEXPORTED_RUNTIME_METHODS=" + json.dumps(["ccall", "FS", "UTF8ToString"]),
               "-sEXPORTED_FUNCTIONS=" + json.dumps(["_" + p for p in exports]),
               "-o", str(target)]
        if variant == "simd":
            cmd.append("-msimd128")
        build_env = os.environ.copy()
        build_env.update(EMSDK_PYTHON=sys.executable, EM_CONFIG=str(config),
                         EM_CACHE=str(build / "emscripten-cache"))
        if not args.reuse_engines:
            subprocess.run(cmd, check=True, cwd=ROOT, env=build_env)
        engines[variant] = target.read_text(encoding="utf-8")
    if args.engines_only:
        return
    models = {}
    prefix="dialogue-"+args.version
    stages=json.loads((build/(prefix+"-export/stages.json")).read_text())
    for name in [f"{prefix}-stage-{i}.cn" for i in range(1, 8 if args.version=="v6" else 7)]:
        data = (build / (prefix+"-export") / name).read_bytes()
        models[name] = {"base64": base64.b64encode(data).decode(), "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(), "protocol": "chat-v2", "cot": True}
    variants={}
    if args.version=='v6':
        for variant in ('entry','last'):
            name=f'{prefix}-stage-7-{variant}.cn'
            data=(build/(prefix+'-export')/name).read_bytes()
            variants[name]={"base64":base64.b64encode(data).decode(),"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"protocol":"chat-v2","cot":True}
    evidence = {}
    for name in ["metrics.csv", "qat-metrics.csv", "training.json", "qat-training.json",
                 "certificate-fp32.json", "certificate-fp8-file.json", "certificate-fp4-file.json",
                 "certificate-fp4.json", "evaluation-fp4.json"]:
        evidence[name] = (ROOT / "run-final" / name).read_text(encoding="utf-8-sig")
    datasets = {name: (ROOT / "run-final" / name).read_text(encoding="utf-8")
                for name in ["local.tsv", "curriculum.tsv"]}
    payload = {"models": models, "variants":variants, "evidence": evidence, "datasets": datasets}
    payload["model_prefix"]=prefix
    payload["default_config"]={"dim":stages[-1].get("dim",24),"ff":stages[-1].get("ff",24),"layers":stages[-1].get("layers",6),"seed":20261018}
    if stages[-1]["parameters"]>500000:
        payload["default_config"].update(dim=56,ff=128,layers=6)
    payload["dialogue_training"] = (build/(prefix+"-run/progress.csv")).read_text(encoding="utf-8")
    payload["dialogue_stages"] = stages
    payload["dialogue_test"] = json.loads((build/(prefix+"-tests/report.json")).read_text(encoding="utf-8"))["random_groups"]
    payload["previous_scores"] = json.loads((build/"dialogue-v5-tests/report.json").read_text())["random_groups"]
    payload["capacity_baseline"] = json.loads((build/"dialogue-v4-tests/report.json").read_text(encoding="utf-8"))["random_groups"]
    payload["outcome_before"] = json.loads((build/(prefix+"-tests-stage-6/report.json")).read_text())["random_groups"] if args.version=="v6" else None
    payload["outcome_last"] = json.loads((build/(prefix+"-tests-last/report.json")).read_text())["random_groups"] if args.version=="v6" else None
    payload["outcome_parity"] = json.loads((build/"parity-v6/report.json").read_text()) if args.version=="v6" else None
    payload["outcome_comparison"] = json.loads((build/"outcome-v6/comparison.json").read_text()) if args.version=="v6" else None
    payload["outcome_rollouts"] = json.loads(sorted((build/(prefix+"-run")).glob("stage-7-rollouts-*.json"),key=lambda p:int(p.stem.split("-")[-1]))[-1].read_text()) if args.version=="v6" else None
    payload["capacity_pilots"] = json.loads((build/"capacity-v5/pilots.json").read_text(encoding="utf-8"))
    payload["capacity_comparison"] = json.loads((build/"capacity-v5/comparison.json").read_text(encoding="utf-8")) if args.version=="v5" else None
    wasm_report=build/("validation-outcome-wasm.json" if args.version=="v6" else "validation-capacity-wasm.json")
    payload["wasm_validation"]=json.loads(wasm_report.read_text(encoding="utf-8")) if args.version in ("v5","v6") and wasm_report.exists() else None
    payload["compact_baseline"] = json.loads((WEB / "build/dialogue-v3-tests/report.json").read_text(encoding="utf-8"))["random_groups"]
    worker = WEB.joinpath("worker.js").read_text(encoding="utf-8")
    template = WEB.joinpath("index.html").read_text(encoding="utf-8")
    if args.version!="v6":
        template=template.replace('<option value="{{MODEL_PREFIX}}-stage-7.cn" selected>07 · 自由草稿 RL 阶段选择</option>','')
    for key, name in [("DATA_SECTION", "datasets.html"), ("PLAY_SECTION", "play.html"),
                      ("DETAIL_CSS", "details.css")]:
        template = template.replace("{{" + key + "}}", WEB.joinpath(name).read_text(encoding="utf-8"))
    if args.version!="v6":
        template=template.replace('<option value="{{MODEL_PREFIX}}-stage-7.cn" selected>07 · 自由草稿 RL 阶段选择</option>','').replace('-stage-6.cn">06 · QAT','-stage-6.cn" selected>06 · QAT')
    for key,value in {"MODEL_PREFIX":prefix,"MODEL_PARAMETERS":f'{stages[-1]["parameters"]:,}',
                      "MODEL_SPLIT":f'{stages[-1]["layers"]//2} + {stages[-1]["layers"]//2}', "MODEL_LAYERS":stages[-1]["layers"], "MODEL_STAGES":len(stages), "MODEL_PARAMETER_LABEL":f'{stages[-1]["parameters"]/10000:.1f} 万'}.items():
        template=template.replace("{{"+key+"}}",str(value))
    # Data live in inert JSON script elements. Escape script terminators in JS.
    parts = {"PAYLOAD": payload, "ENGINES": engines, "WORKER": worker}
    for key, value in parts.items():
        template = template.replace("{{" + key + "}}", json.dumps(value, ensure_ascii=False).replace("<", "\\u003c"))
    template = template.replace("/* APP_SCRIPT */", "\n".join(
        WEB.joinpath(name).read_text(encoding="utf-8") for name in ["app.js", "datasets.js"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template, encoding="utf-8")
    print(f"Built {args.output} ({args.output.stat().st_size:,} bytes), no external requests.")


if __name__ == "__main__":
    main()
