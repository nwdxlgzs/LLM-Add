"""Write user-facing evidence after all v6 training/engine reports exist."""
import csv,json
from pathlib import Path
WEB=Path(__file__).resolve().parent;ROOT=WEB.parent;B=WEB/'build'
stages=json.loads((B/'dialogue-v6-export/stages.json').read_text());config=stages[-1]
comparison=json.loads((B/'outcome-v6/comparison-final.json').read_text())
new=json.loads((B/'dialogue-v6-tests/report.json').read_text());before=json.loads((B/'dialogue-v6-tests-stage-6/report.json').read_text())
last=json.loads((B/'dialogue-v6-tests-last/report.json').read_text())
parity=json.loads((B/'parity-v6/report.json').read_text());wasm=json.loads((B/'validation-outcome-wasm.json').read_text())
rows=list(csv.DictReader((B/'dialogue-v6-run/progress.csv').open()))
phase7=[r for r in rows if r['stage']=='7'];p=phase7[-1]
result=f"{config['layers']} 层、宽度 {config['dim']}、FFN {config['ff']}，{config['parameters']:,} 参数"
summary=f"当前 `dialogue-v6`：**{result}**，在 500,000 参数上限内。预置七个阶段；新增阶段 7 为自由草稿 RLOO，只按最终答案与合法消息控制符奖励，草稿内容不评分。SFT 的逐位字段只是示范格式，不是这一阶段的规则。"
doc=f'''# 自由 CoT 草稿 RL 与 50 万以内的深宽实训

{summary}

打开 [离线网页](dist/ai-add.html)。测试区的“自由草稿 RL 的版本”可以分别测试验证选择、最后更新和训练前；失败权重也保留。训练区显示同题四份实际采样与奖励，长文本在框内滚动。用户原有检查点不自动删除。

## 新增阶段 7 的实际目标

模型输入只有一条 `user: A+B` 和 assistant 前缀。在全部 261 类词表自由采样四个完整回答，不屏蔽输出 token、不按列调用网络、不强行补齐思考块。CoT 模式要求一个正常的 `<think>…</think>` 边界；草稿可为空、换算法或包含错误中间计算。只要最终数字正确且消息合法，奖励均为 1。错误答案、重复消息控制符、裸 `/think` 等明显错误的控制符、缺少结束标记得 0。允许答案前后空白与前导零。

不检查 `pos/a/b/cin`、草稿行数、步骤正确率、与示范 CoT 的相似度，也不对草稿计算监督 CE。四份采样做 on-policy RLOO（temperature 1、无 top-k）与冻结参考策略 KL / 熵正则。四份奖励相同时不调用优化器。阶段 7 保留输入模型的精度（量化模型继续 STE 更新），避免把精度切换当作 RL 收益。阶段 5 的步骤奖励保留为历史对照；阶段 7 的采样、奖励和选模全部走最终答案标准。

选择验证也改为真实 greedy 自回归生成，不 teacher-force 标准草稿、不按固定 CoT 字符串匹配。验证集共 200 题，1/2/4/6/10 位各 40 题，含零、连续进位与最大数。这是选模数据；独立随机测试另用种子 938234，每种实际长度 20 题。

训练时每条提示词与回答合计最多 512 token，以控制 CPU / WASM 内存。推理不设置输入位数上限，超出上下文使用滚动窗口；接受长题不代表学会长题。真值加法仅在标签、奖励与独立评分中使用，不回传为推理答案。

## 深宽比较

同种子 20261018、batch 4，分别从零训练 300 / 1200 / 2000 / 4000 步的四段课程。以下分数只比较最终答案，内部草稿不判分。层数改变时同时重建 CED 分界、Full/Reuse/Reindex 的 KV 与索引来源，保留 SWA、CSA2、MoE 2+1、Engram、mHC 和 RoPE。

| 候选 | 层 / 主干 / FFN | 参数 | 自由答案 / 200 | 各长度 / 40 | 实训与评测分钟 |
| --- | --- | ---: | ---: | --- | ---: |
'''
for c in comparison['candidates']:
 doc+=f"| {c['name']}{'（采用）' if c['name']==comparison['winner'] else ''} | {c['layers']} / {c['dim']} / {c['ff']} | {c['parameters']:,} | {c['correct']} | {' / '.join(map(str,c['width_correct']))} | {c['seconds']/60:.1f} |\n"
doc+='\n按本轮预算下的真实答案正确率选择架构，同分看耗时。候选在同机并行运行，耗时不是独占机器的通用性能基准。参数相近但不完全相同，结论限于这些配置和训练预算。赢家保留试训选择权重和优化器，从第 4000 次尝试继续混合长度训练至 8000 次（采样 RNG 为新的确定性流），再执行 200 步步骤 RL、600 步 QAT、400 步自由草稿 RL；试训检查点另外保留。\n\n## 七阶段最终训练记录\n\n| 阶段 | 步数 | 采用步 | 选择 / 200 | 末次 / 200 | 验证指标 | 精度 |\n| --- | ---: | ---: | ---: | ---: | --- | --- |\n'
for s in stages:
 doc+=f"| {s['stage']} | {s['steps']} | {s['best_step']} | {s['correct']} | {s['last_correct']} | {'自由最终答案' if s['stage']==7 else '固定整段匹配'} | {['FP32','FP8','FP4'][s['precision']]} |\n"
doc+=f"\n阶段 7 有 {p['informative']} 轮出现奖励差异。它的分数与前六阶段固定整段匹配的口径不同，不能直接把阶段 6→7 的验证数字变化当作能力提升。阶段 7 若选择第 0 步，就是保留训练前模型；并不删除末次更新。\n\n## 独立同题答案测试\n\n| 实际位数 | RL 前 | RL 后选择 | RL 末次 |\n| --- | ---: | ---: | ---: |\n"
for w,r in new['random_groups'].items():doc+=f"| {w} | {before['random_groups'][w]['correct']}/20 | {r['correct']}/20 | {last['random_groups'][w]['correct']}/20 |\n"
doc+='\n所有原始输出与 token 在 `build/dialogue-v6-tests`；错误结果保持原样。本轮没有宣称保证十位或任意长度全对，也没有用认证限制推理。\n\n## C / WASM 与内存检查\n\n'
doc+=f"原生 CTest 15/15，Linux ASan / UBSan / leak 15/15；新增聊天目标测试另有修订后的 sanitizer 记录。WASM 标量与 SIMD 真实运行七阶段 CoT / direct 更新、保存三份权重，检查不同草稿同等奖励、错误控制符、参数上限与两种拓扑。协议检查 `{wasm['protocol_passed']}`；自由生成逐 token 等效 `{wasm['generation_equivalence_passed']}`；12 组数值对照整体 `{parity['passed']}`，FP32 匹配后端 `{parity['fp32_backend_aligned_passed']}`。容差通过不是长期训练轨迹相同的证明；所有失败差异保存在报告中，不改写为通过。\n"
doc+='''
## 复现与路径

```powershell
cmake --build ai-add/build -j4
python ai-add/web/run-outcome-pilots.py
python ai-add/web/finish-outcome-training.py
python ai-add/web/build.py --emsdk D:/project/emsdk-3.1.69
```

脚本拒绝覆盖已有的最终训练目录；复现请在独立工作区或使用新的输出目录。手动浏览器训练不需要上述环境。新建和训练严格限制 500,000 参数；旧大模型可加载测试。

- `src/chat.c`：合法消息解析、只按结果奖励、自由采样与实际生成选模。
- `src/train.c`：阶段 7、冻结参考模型、入口/末次/选择保存。
- `web/build/outcome-v6`：三个候选、训练日志和比较依据。
- `web/build/dialogue-v6-run`：七阶段优化器检查点及实际 RL 采样。
- `web/build/parity-v6/report.json`：数值容差与反例。
- `web/build/validation-outcome-wasm.json`：实际 WASM 推理和训练。
'''
(WEB/'OUTCOME_RL.md').write_text(doc,encoding='utf-8')
p=ROOT/'README.md';s=p.read_text(encoding='utf-8');start=s.index('**网页默认使用') if '**网页默认使用' in s else s.index('当前 `dialogue-v6`');end=s.index('\n历史局部模型',start);s=s[:start]+summary+'\n\n详见 [自由草稿 RL 与深宽实训](web/OUTCOME_RL.md)。前轮 550,556 参数模型、旧逐列模型和全部旧记录仍保留为历史，不能把旧局部认证当作整式自由生成能力。\n'+s[end:];p.write_text(s,encoding='utf-8')
p=WEB/'README.md';s=p.read_text(encoding='utf-8');s=s.replace('六阶段整式模型','七阶段整式模型').replace('## 六阶段训练','## 七阶段训练');start=s.index('当前默认使用') if '当前默认使用' in s else s.index('当前 `dialogue-v6`');end=s.index('## 真实训练格式');s=s[:start]+summary+'\n\n模型输入整题，assistant 连续生成，不拆列调用或拼接答案。上下文 512，测试位数不限。详见 [本轮实训与限制](OUTCOME_RL.md)。内置只列七个阶段；我的训练单列用户检查点，默认不清理旧记录。阶段 7 还有入口、末次、选择版本可测试。\n\n'+s[end:]
if '| 自由草稿 RLOO |' not in s:s=s.replace('| QAT | 从一开始混合全部长度，模拟量化下继续完整对话监督 |','| QAT | 从一开始混合全部长度，模拟量化下继续完整对话监督 |\n| 自由草稿 RLOO | 仅按最终答案与消息控制符奖励，草稿自由，真实生成选模 |')
start=s.index('\nCoT RLOO') if '\nCoT RLOO' in s else s.index('\n阶段 5 的 CoT RLOO');end=s.index('\n检查点可勾选',start)
s=s[:start]+'''
阶段 5 的 CoT RLOO 使用步骤正确比例和最终答案共同奖励；阶段 7 只检查最终答案与消息控制符，草稿内容不评分。无 CoT 时不要求思考块。所有对话阶段按设置步数执行，零分与失败不阻止测试。

前六阶段的曲线是固定 1/2/4/6/10 位各 40 题的完整 assistant 目标匹配数，含 CoT 和结束标记；阶段 7 使用真实 greedy 连续生成，只按最终答案与合法控制符判分，没有固定 CoT 匹配、逐步正确率或草稿 CE。两种指标不能直接比较。固定验证集用于选权重；独立同题随机测试另行报告。

每次运行保存开始前权重，以及各阶段训练前、真实最后更新、验证选择三个版本。预训练采用最后更新；阶段 2–6 按完整匹配数和验证 CE 选择，阶段 7 按自由生成最终答案正确数选择，同分保留较早版本。QAT 也与转换前模型比较；量化退步时可保留 FP32，FP4 末次仍保留。阶段 7 保持输入精度。验证选择可能为第 0 步，失败的末次更新照常可测试。
''' + s[end:];p.write_text(s,encoding='utf-8')
p=WEB/'VALIDATION.md';s=p.read_text(encoding='utf-8').replace('## 当前：六层扩容 dialogue-v5','## 历史：六层扩容 dialogue-v5');at=s.index('\n## ')
if s[at:].startswith('\n## 当前：自由草稿 RL dialogue-v6'):s=s[:at]+s[s.index('\n## ',at+4):]
s=s[:at]+'\n## 当前：自由草稿 RL dialogue-v6\n\n'+summary+'\n\n实际深宽比较、七阶段权重、答案成绩和所有数值差异见 [OUTCOME_RL.md](OUTCOME_RL.md)。验收脚本为 `verify-outcome-delivery.py`；它检查交付完整性，不把算术失败或数值反例改成通过。\n'+s[at:];p.write_text(s,encoding='utf-8')
print('Updated v6 documentation from actual artifacts.')
