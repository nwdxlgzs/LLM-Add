# Web 版实际验收（2026-10-18）

## 当前：自由草稿 RL dialogue-v6

当前 `dialogue-v6`：**6 层、宽度 64、FFN 104，467,612 参数**，在 500,000 参数上限内。预置七个阶段；新增阶段 7 为自由草稿 RLOO，只按最终答案与合法消息控制符奖励，草稿内容不评分。SFT 的逐位字段只是示范格式，不是这一阶段的规则。

实际深宽比较、七阶段权重、答案成绩和所有数值差异见 [OUTCOME_RL.md](OUTCOME_RL.md)。验收脚本为 `verify-outcome-delivery.py`；它检查交付完整性，不把算术失败或数值反例改成通过。

## 历史：六层扩容 dialogue-v5

新模型仍为同一六层 DSV4.1F 缩小结构，dim64、FFN128、550,556 参数，chat-v2 显式字段 CoT、512 上下文。六阶段实际执行 600/2400/4000/8000/200/600，耗时 67.5 分钟。选择整段匹配 0/25/72/153/153/154（满分 200），最终选择 FP4；RLOO 有 35 轮信号但未改善选模。所有阶段入口/末次/选择及 optimizer 包均保留，原 v4 历史不删除。

独立真随机同题原生 C 测试：一位 20/20、两位 20/20、四位 20/20、六位 19/20、十位 3/20、十二/二十位 0/20。十位取数正确列从 8/200 升到 181/200，但仍未收敛。详见 [CAPACITY_TRAINING.md](CAPACITY_TRAINING.md)。

Windows CTest、Linux ASan/UBSan/leak 各 15/15。新增测试覆盖实际扩容后长样本反向更新、权重/优化器保存互载和前向逐 bit 相同；原配置兼容保留。WASM 直接调用同一 `add_chat_resize`，网页支持主干16..128、专家16..512常用档位。

FP32 的 C/WASM 预填、随机 masked SFT、已训 SFT 和实际 RLOO 数值对照通过声明容差；随机 FP4 QAT 跨后端不一致，最终 FP4 的跨后端 logits 也未全过容差。百位滚动和部分随机题存在真实 token 分歧，不能宣称完全等效，失败记录保存在 `parity-v5` 与 `capacity-v5`。流程与协议检查单列通过，token 等效检查仍记失败。

网页更新后只内置六个扩容阶段，原有本机用户检查点保留；增加扩容前后同题取数/步骤/答案表，C 与 WASM 成绩分别显示。各阶段可自由测试，保持不设测试位数上限、独立正确性提示和内部滚动。

## 历史：显式字段 CoT dialogue-v4

`dist/ai-add.html` 约 2.85 MB，六个新阶段，chat-v2，512 上下文，57,388 参数。实际 assistant 输出 `pos=N a=A b=B cin=C -> D CO`，从整道 user 问题连续生成；无调度拼接。阶段、目标和导出包均同步更新。训练配方及未收敛结果见 [DIALOGUE_TRAINING.md](DIALOGUE_TRAINING.md)。独立随机一位 20/20，两位 10/20，四/六/十/十二/二十位各 0/20，未学会十位加法；最终选择 FP32 66/200，FP4 最后 55/200。

- `validation-fields-native.log`、`validation-fields-asan.log`：原生与 sanitizer 各 15/15。
- `validation-fields-regression.json`：标量/SIMD，历史协议兼容、失败生成与六阶段更新。
- `validation-fields-wasm.json`：六个新阶段、5 个实际生成与 C 完全一致、旧紧凑格式兼容、CoT 开关及阶段权重保存。
- `parity-v4/report.json`：相同后端数值对照通过；跨后端 FP4 QAT 有反例，整体 `passed=false`，不得声称全部后端训练等效。
- `browser-fields.json` 与截图：新格式输入/目标/实际原文、独立判错、随机长题、失败阶段、CoT 目标切换、内部滚动、六个内置模型及 72 个用户检查点保留。
- `validation-fields-delivery.json`：实际浏览器脚本指纹、权重、C/WASM、课程、HTML 无外部资源与数据一致。该通过仅表示交付完整，不将模型算术失败或跨后端 QAT 失败改成通过。

HTTP 本地预览已检查；不将 HTTP 测试说成实际双击本地文件测试。离线 HTML 内嵌全部引擎与权重，用户 IndexedDB 需另存模型包迁移。

## 历史：整题单轮对话 dialogue-v3

当前交付 `dist/ai-add.html` 为约 2.85 MB 的单文件，内嵌六个整式阶段及两种 WASM 引擎。默认路径 user 输入整题，assistant 连续生成所有 CoT 与最终答案；移除了程序逐列调度后拼接的伪完整轨迹。原始文本与实际 token 一致；停止及校验信息独立展示。

本轮从零六阶段实训 15,800 次请求更新；固定选模 74/200，独立随机一位 19/20、两位 11/20，四/六/十/十二/二十位各 0/20。**模型仍未学会十位加法。** 详细配方、阶段精度和失败样例见 [DIALOGUE_TRAINING.md](DIALOGUE_TRAINING.md)。QAT 最后 64/200，最终保留较好的 FP32，并未标成 FP4 成品。

- `validation-dialogue-native.log`、`validation-dialogue-asan.log`：Windows 与 Linux ASan/UBSan/leak 各 15/15（引擎/协议及旧兼容回归，不是算术成绩）。
- `parity-v3/report.json`：12 组原生 C / WASM 容差对照，含真实整式 SFT、RLOO（6 次有效更新）、FP4 QAT、Adam 与互载。
- `validation-dialogue-wasm.json`：六阶段逐一实际推理，四个代表样例生成 token 与 C 完全一致，CoT/无 CoT 各六阶段更新与三版本保存。
- `browser-dialogue.json`：实际浏览器单轮原始消息、用户原题判错、99+1 正确、20 位随机题、早期失败检查点、CoT 开关改目标、内部滚动、六个内置阶段与原有 72 个本机模型完整保留。未产生新的浏览器训练存档。控制台无错误。
- `validation-dialogue-delivery.json`：实际 HTML 四段脚本 DOM 指纹、六份权重哈希、引擎源、在线课程及上述测试证据一致，无外部资源。

```powershell
python ai-add/web/verify-dialogue-delivery.py
```

HTTP 本地预览已实际验证；这里不把 HTTP 验证说成实际双击文件运行。HTML 无外部资源且全部权重内嵌，保存页面不含用户 IndexedDB，用户模型需另存包。

## 以下为旧版本验证记录



交付文件：`dist/ai-add.html`，约 9.5 MB。包含全部模型、历史记录、SIMD/标量引擎；不需要独立 `.wasm`、模型下载或后端服务。

## 模型选择器整理（当前界面）

“使用的模型”只保留两组：预训练好的六个阶段，以及本机保存的“我的训练”。六项依次为预训练、三段 SFT、RLOO、QAT 后 FP4 成品；不再重复列出旧协议、随机初始化和其他精度副本。阶段卡片也使用这六个模型，默认曲线为本轮原生训练。当前训练模型仅在实际创建、载入或训练后出现，清空检查点时一并移除。

浏览器实际选择一个本地模型后删除全部 72 个检查点，选择器只剩六项，自动切回 FP4，十位混合进位题正确；随后撤销删除并刷新，全部 72 个检查点与对应选项恢复。阶段 1 的未完成输出也保留。没有为本次界面验收新增训练记录。记录与截图为 `build/browser-model-picker.json` 和 `build/browser-model-picker.png`，最终脚本/权重/数据一致性报告为 `build/validation-model-picker-delivery.json`。下方各轮模型与训练结果保留各自原始报告。

## 规范逐位对话与原生训练（当前版本）

- 根因与完整配方见 [STRUCTURED_TRAINING.md](STRUCTURED_TRAINING.md)。此前修正格式时同时改成整式自由生成，给小模型额外增加了定位、循环和长回答任务。本轮保留同一 57,388 参数网络，恢复局部转移，并采用真实 user/assistant 消息。模型从全词表预测结果位、空格、进位及两个正常结束标记，局部不使用 think 标记。
- 原生 C 从零执行 200 步预训练、三段各 3000 步 SFT、400 轮 RLOO、2000 步 QAT。各阶段选中权重依次 1/90/152/200/200/200，共 200 种局部状态。RL 前已收敛，本次 RL 没有奖励差异、没有有效更新。FP8 直接转换 197/200，额外 25 步 QAT 恢复 200/200。失败转换、六阶段训练前/最后/选中版本均保留。
- FP32/FP8/FP4 在 scalar、SSE2、SSSE3、SSE4.1、AVX、AVX2、AVX2-FMA、AVX512 八种本机后端均 200/200。FP4 独立种子 20261022，1/2/4/6/10 位各 2000 道，共 10,000 道全对；另通过千位连续进位。FP32/FP8 原生随机题数为 0，其随机整题测试在 WASM 执行。原生报告为 `build/column-v2-export/evaluation-fp*.json`。
- 同一个 C 探针链接原生与真正交付 WASM；从同一序列化参数/RNG/优化器对比完整 logits、loss、梯度范数、全参数和 Adam m/v，包括有效 RLOO、SFT、QAT、三种成品前向与优化器互载。16 个 WASM 组合与原生 scalar/auto 比较，5856 个 argmax 全部一致，最大 logits 差 8.34465027e-5、权重差 3.88734043e-6。按报告声明的绝对/相对容差通过，不宣称 bit 相同或长期训练路径相同。报告 `build/parity-v2/report.json`。
- 实际交付 WASM scalar/SIMD × 三种精度，各通过 200 种局部转移、200 道整题和千位进位；检查真实角色消息、正常结束及仅一对整题展示包装，报告 `build/validation-column-wasm.json`。Windows Release、Linux ASan/UBSan/leak 均 15/15，日志为 `build/validation-column-native.log` 与 `build/validation-column-asan.log`。
- 网页默认新 FP4，新增六个阶段可测试，旧九个权重字节保留；允许失败模型照常作答。浏览器实际完成六阶段各一步的流程回归，本机检查点 53→72，保留原来的 53 个，刷新恢复 72 个。这个短训练 0/200 是流程回归，未作为收敛模型。数据导出 200 条规范逐位 messages JSONL，阶段模型包保留 column-v2、CoT、stage 等元数据。最大十位数和二十位题实际作答并独立判对；早期阶段的错误答案原样显示。详见 `build/browser-column-v2.json` 与截图 `build/browser-column-v2.png`。
- 原始消息、数据 token 与轨迹继续限制高度并内部滚动；本地检查点手动选择删除/撤销保留。`build/validation-column-delivery.json` 将最终浏览器四段脚本指纹、18 个权重、源码、引擎、导出数据及上述报告逐项核对。旧 chat-v1/legacy-v1 实验记录仍在下方，不能当作本轮结果。

最终交付核对：

```powershell
python ai-add/web/verify-chat-delivery.py --checks 15 --browser-report browser-column-v2.json --report validation-column-delivery.json --native-log validation-column-native.log --asan-log validation-column-asan.log --wasm-report validation-column-wasm.json
```

## 训练退化与长内容修复（前一版本，chat-v1）

- 详细根因、两次完整六阶段训练及统一验证表见 [TRAINING_DIAGNOSIS.md](TRAINING_DIAGNOSIS.md)。修复课程十位只在最后一步出现、长度切换无重放、无奖励差异的 RL 仍更新，以及最后权重覆盖较好权重。新对话配置使用 64 token 局部窗口，参数数目不变。
- 同一新版验证集上，修改前最终 1/200，修复后选中模型 45/200；后者分组为 40/2/1/1/1（每组 40），**未学会通用十位加法**。QAT 最后 FP4 31/200，因此采用输入 FP32 的 45/200，两个版本均保留。完整原生复跑各阶段 batch=4，网页默认 QAT batch=8，未声称两者相同。
- Windows Release 14/14、Linux ASan/UBSan/leak 14/14。覆盖五个时间档及十位的持续出现、SFT 重放、RL 无信号模型字节不变、验证选择不差于最后更新。最终日志分别为 `build/validation-training-repair-native.log`、`build/validation-training-repair-asan.log`。WASM SIMD/标量短流程均通过，报告为 `build/validation-training-repair.json`。
- 数据样本、原始 JSON、token 列表、推理轨迹和答案限制高度并内部滚动。用户阶段检查点可勾选、全选、删除和当前页面撤销最近一次删除；重置不清空历史。每个新对话阶段分别保留训练前、最后更新和验证选择，阶段结束显示“未达标”而不是冒充训练成功。
- 最终浏览器实际执行有 CoT 六阶段各 1 步、SFT batch=1、QAT batch=8 的流程回归：新增 19 个检查点，34→53，刷新全部恢复。实际勾选删除一个 53→52，选择器同步移除，再撤销回 53，未丢弃原有实验。导入完整原生复跑的 `stage-6.aiadd` 后，`99+1` 真实对话回答 100；十位混合进位题输出格式失败，页面保留原文和独立答案 11111111100。短流程不作为收敛证据。
- 最终数据 token 面板实际内容高 3267px、显示高 320px，内部滚动已操作验证；回答、JSON 等同样有高度上限。截图为 `build/browser-training-repair-scroll.png` 与 `build/browser-training-repair-checkpoints.png`。操作、完整原始输出、空控制台错误/警告列表和四段内嵌脚本指纹在 `build/browser-training-repair.json`；与最终 HTML 逐段一致性、九个旧权重保持不变、无外部资源的结果在 `build/validation-training-repair-delivery.json`。

## 对话格式与量化速度修正（前一版本）

- 新建实验为 `chat-v1`，完整 user 消息和 assistant 前缀真实进入网络，然后连续生成完整 assistant 回答。每段 CoT 目标只有一对思考标记；预训练及无 CoT 目标没有思考标记。数据预览与导出直接调用 C 训练器的 `add_chat_example`。
- 旧的 `run-final` 权重没有改动，保留 `legacy-v1` 原协议，界面明确区分旧协议历史和新对话模型。旧千位题、200 种局部状态等证据只属于旧模型，不能移用于新对话格式。
- Windows Release 及 Linux ASan/UBSan/leak detection 均 **14/14**。新增测试检查完整角色和消息边界、单个 CoT 块、无 CoT/预训练零思考标记、十位目标容纳、两种真实六阶段更新、量化缓存前后 200 种旧模型状态正确。日志为 `build/validation-chat-native.log` 和 `build/validation-chat-asan.log`。
- WASM SIMD 与标量通过相同的新格式检查及两种六阶段短训练，仍保留旧格式和长输入回归检查。报告为 `build/validation-chat.json`。这些短实验用于流程回归，不是对话模型收敛或任意位正确性证明。
- 固定相同 20 位题、WASM SIMD128、各路径预热 4 次再计时 15 次，以中位数比较：本次 FP32 约 4.4–4.6 ms，FP8 紧凑路径约 15.9 ms、缓存约 8.5 ms，FP4 紧凑约 15.8 ms、缓存约 8.2 ms。缓存改善约 1.9 倍，但仍慢于 FP32。`build/benchmark-quant-before.json` 保存修改前记录，`build/benchmark-quant-after.json` 保存本轮全部重复计时。数字是本机测量，不是跨设备承诺。
- 新缓存将量化权重一次解码为 FP32 计算缓存，保留量化后的权重值与激活量化。约增加 200 KiB，磁盘文件仍保持量化存储；优化器/重新量化会使缓存失效。页面可切换速度优先和紧凑权重，没有用原始全精度权重替换量化模型。

## 实际浏览器操作

### 对话格式与模型包（前一版本，chat-v1）

- 浏览器实际各完成一次有 CoT、无 CoT 六阶段短实验，每阶段 1 步、SFT batch 1。开始前及每阶段模型共保存 14 项；两个实验均未收敛，零分阶段照样保留，没有当作可用加法成品。
- 刷新最终网页后，从 IndexedDB 恢复全部 34 个旧、新阶段模型。分别测试保存的对话 CoT 阶段和导入的无 CoT 阶段，真实输入为完整 user/assistant 提示词；达到生成预算的失败输出、原始 token 和独立正确答案均保留。
- 下载 512 条两位 CoT JSONL，逐条检查 user/assistant 角色、user 无思考标记、assistant 仅一对思考标记。导出文件保存为 `build/chat-dataset-export.jsonl`，直接来自 C 样本函数。
- 实际通过文件选择器导入 `build/browser-chat-stage6.aiadd`，恢复 `chat-v1` 和无 CoT 设置，并在完整对话模式推理。最终“下载模型包”也实际下载并读取验证，保留协议及 CoT 元数据，保存为 `build/browser-chat-weights.aiadd`；权重包不带优化器，训练断点另可带优化器。
- 最终页面展示 `99+1` 的完整 user 消息、assistant 前缀和单个思考块目标。截图为 `build/browser-chat.jpg`；阶段训练日志、恢复与导入记录、失败推理和空控制台错误/警告列表在 `build/browser-chat.json`。四段 DOM 内嵌脚本的长度与 FNV-1a UTF-16 校验值均与最终离线文件一致。
- `python web/verify-chat-delivery.py` 检查九个旧权重字节与 SHA-256 保持不变，两个引擎、Worker、应用脚本与源码一致，无外部资源，以及导出数据、模型包和原生/WASM 验收记录。结果为 `build/validation-chat-delivery.json`。

### 不限位数与开放实验（前一版，旧逐列协议）

本节为新版行为；后面初版中的“未训练拒绝”“长度拒绝”属于旧界面验收，不再适用于网页实验接口。原生 CLI 的旧约束测试继续保留。

- 新 C 实验接口动态分配结果数组，允许任意长度的非空十进制输入，不做认证门禁。Windows Release CTest 为 **13/13**；Linux ASan、UBSan、leak detection 也为 **13/13**，没有错误报告。新增测试覆盖 1,000 位进位、直接回答上下文滚动、随机模型失败输出和无 CoT 六阶段更新。
- WASM SIMD 与标量分别通过同样的长输入和失败推理检查，并实际执行有 CoT、无 CoT 的六阶段短训练。无 CoT 即使零分也跑完所设步数，阶段文件实际写出。报告为 `build/validation-open-experiments.json`。
- 实际浏览器输入 1,000 个 9 加 1，FP4 成品输出 1 后 1,000 个 0，独立校验为正确，原始文本保留 1,000 列预测，页面解释展开前 200 列。
- 实际选择随机初始化模型测试 `9+9`，保留非法字节 token、停止原因和独立正确答案 18，没有拒绝推理。
- CoT 开关切换会刷新样本提示词、监督目标、课程名称和测试指标。无 CoT 预览显示整式输入及最终答案目标；不把模板预览冒充真实某一步的随机样本。
- 浏览器实际各用每阶段 2 步、SFT batch 1 完成有/无 CoT 的六阶段短实验，两者均零分但都保存并可独立推理。CoT RL/QAT 的最后更新和最佳检查点分别保留。这些是流程验收模型，不作为训练收敛证据。
- 早期 SFT 模型把 `9+9` 输出为 `9`，页面明确保留原答案并显示正确答案 18；没有因未通过检查而拒绝。无 CoT QAT 和 CoT QAT 的非法输出也分别实际展示。
- 重置后七个无 CoT 阶段仍保留，刷新后从 IndexedDB 恢复；实际载入已保存模型续训，追加记录而不覆盖原项。最终重新载入后，续训默认推荐下一阶段，实验编号继续递增。
- 实际下载无 CoT 阶段 `.aiadd` 模型包，读取磁盘文件确认模式、阶段、权重字节和 SHA-256，再经网页文件选择器重新导入，恢复无 CoT 模式及对应数据预览。验收模型包保存在 `build/browser-direct-stage6.aiadd`。
- 最终界面实际随机生成并算对 1,001 位加法。直接回答模式也接受相同长输入，报告上下文滚动 186 次，达到生成预算后保留失败输出。最终随机 20 位题正确，截图为 `build/browser-open-experiments.jpg`。
- 浏览器 15 项操作记录、四段 DOM 脚本校验及空错误/警告日志保存在 `build/browser-open-experiments.json`。四段脚本与最终离线 HTML 完全一致；9 个内置模型及两种引擎的字节一致性、语法检查结果保存在 `build/validation-open-delivery.json`。原生与内存检查日志分别为 `build/validation-open-native.log`、`build/validation-open-asan.log`。

新版仍是同一 6 层 DSV4.1F 缩小网络。以上验证表明接口接受长输入，并验证若干实例正确；没有宣称无 CoT 或任意实验模型能保证所有长度、所有算式都正确。

### 数据与原始输出改版（前一版界面）

- 新增六阶段数据浏览，实际检查数量依次为 200/55/100/200/2560/200。200 条预训练项为按在线生成规则枚举的可能样本，SFT/RL 则直接读取内置原始 TSV。
- 实际选择预训练并翻页，检查复制目标与逐 token 监督位置；选择 RL 十位课程样本，保留前导零，点击交给成品模型，输出 `10000000000`，显示十列实际 CoT 与逐列 token ID。
- 实际点击 1 位、2 位、10 位与随机难度的“随机出题”，再运行模型，全部正确。原始 C 文本、逐列 token、可读解释与回答记录在 `build/browser-ui-update.json`。
- 可见页面中不再有“证书”“认证”文字和操作，200 格证书区已移除。正确性检查在引擎内部自动执行，开发者测试仍保留。
- 重建 SIMD/标量引擎后，独立接口测试均通过，覆盖所有成品精度、输入拒绝、六阶段实际更新，以及新增原始文本和预测字段的一致性；报告为 `build/validation-ui-update.json`。

### 初版完整训练验收（下述证书界面现已移除）

在当前 Chromium 内置浏览器中打开只读本机预览页面：

- 启动自动选择 `wasm-simd128`，内置 FP4 文件重载后在 scalar 与 SIMD128 均为 200/200，min margin=2.84558105。
- `9999999999+1` 输出 `10000000000`；两个最大十位数输出 `19999999998`，展开了实际逐位预测。
- 点击“从零完成全流程”，在第 62 个预训练更新暂停，再继续；之后完成全部六个阶段。
- 完成 SFT 后 200/200，继续真实 RLOO，再做 QAT。QAT 第 25 步达到 200/200，min margin=2.82568359。
- 选择“我的实验模型”，最大十位加法正确。对实际自训文件重载认证，两个浏览器后端均通过。
- 下载实验报告及带优化器的训练断点，重新导入断点，重新认证通过；再手动运行单独 QAT 阶段。
- 对浏览器自训 FP4 执行完整算式验收：两位全集 10,000/10,000、随机十位 10,000/10,000、边界 440/440，全通过，报告为 `build/browser-evaluation.json`。
- 修改设计为 dim16 / FFN16 后实际创建 32,444 参数模型，再恢复默认 57,388 参数；选择未训练历史模型时明确拒绝推理。
- 最终版本“保存完整网页”为内存 Blob 下载链接，已通过浏览器下载事件验收。浏览器 DOM 中四段内嵌脚本/数据与交付源文件的长度和校验值一致（标准化 HTML 换行后）；记录在 `build/download-validation.json`。下载事件返回的路径没有作为磁盘文件可读性验证的依据。
- 最终页面重新认证通过，并输出 `9999999999+9999999999=19999999998`。控制台没有错误或警告；成品界面截图为 `build/browser-final.jpg`。

本机预览仅监听 `127.0.0.1:8765`，只提供一个只读 HTML 路由，没有文件浏览能力。交付不依赖该服务。自动化浏览器的协议策略不允许访问 `file://`，因此没有把 HTTP 预览测试描述成真实双击本地文件的测试；构建中模型、WASM 字节和 Worker 均已内嵌，支持普通浏览器离线打开。

完整浏览器实验报告在 `build/browser-experiment.json`，包含真实逐检查点损失、正确数、间隔、时间与浏览器证书。历史记录不会被这次浏览器实验覆盖。

## 初版独立引擎测试（历史证据）

`node web/test-engine.cjs --pipeline` 已执行。SIMD 与标量引擎分别加载 FP32、FP8、FP4 成品，全部实际文件认证通过；最大十位数、前导零、混合进位、未训练拒绝及非法输入拒绝均通过。六阶段增量接口确实执行更新。

SIMD 从随机初始化跑完的结果：

| 阶段 | 更新次数 | 局部正确数 |
| --- | --- | --- |
| 预训练 | 200 | 2/200 |
| 无进位 SFT | 1,200 | 62/200 |
| 输出进位 SFT | 800 | 112/200 |
| 输入进位 SFT | 1,600 | 200/200 |
| RLOO | 400 | 200/200 |
| FP4 QAT | 25 | 200/200 |

WASM 的浮点和初始化 libm 与原生版本有细微差别，因此其阶段步数不同于原生历史；页面分别标记历史记录与本次实验。

WASM SIMD 导出 `build/wasm-trained-simd-fp4.cn` 还经过原生 C 引擎全部 8 个本机后端重新认证，均 200/200，报告在 `build/native-certificate-wasm-trained.json`。

另执行 `--scalar-pipeline`，兼容标量引擎也从零跑完六阶段：更新数依次为 200/1400/400/1200/400/175，最终 FP4 达到 200/200、最小间隔约 2.042。报告在 `build/validation-scalar.json`。标量自训 FP4 同样经原生全部 8 个后端认证通过。

## 初版原生与内存检查（历史证据）

- Windows Release CTest：12/12，通过全部原测试及增量六阶段接口测试。
- Linux ASan + UBSan + leak detection：12/12，无地址、未定义行为或泄漏报告。
- 新增 Wasm 后端只在 `__wasm_simd128__` 下编译，不改变原生 ISA 选择。

Linux 参数沿用项目已验证的 `-no-pie` 与 `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:handle_segv=0`，避免本环境的 ASan 信号递归问题。

## 交付完整性

`dist/sha256.json` 记录离线 HTML、网页源码、C 引擎与接口、测试证据及浏览器训练产物的 SHA-256。最终构建的 HTML 已检查九个内置模型的字节数与 SHA-256、双 WASM 引擎内嵌数据及 JavaScript 语法；开发脚本也已通过语法检查。
