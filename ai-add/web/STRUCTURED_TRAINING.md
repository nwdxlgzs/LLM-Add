# 规范逐位对话与 C / WASM 等效检查

> 历史局部实验，已退出默认路径。外部逐列调度的成功不能作为整式 assistant 自回归能力的证据。当前默认见 [DIALOGUE_TRAINING.md](DIALOGUE_TRAINING.md)。

此前把可工作的逐位运算改成整串算式自由生成，同时要求小模型自行找位、循环、传递进位并组织长答案，改变了任务难度。只修复遗忘和权重选择无法解决这个变化。整式自由生成实验仍保留作对照。

本轮保持同一个 DSV4.1F 缩小网络（6 层、3+3 CED、24 维、57,388 参数），恢复位置无关的逐位转移。每次真实网络调用为：

```text
<|im_start|>user
add a=7 b=1 cin=0<|im_end|>
<|im_start|>assistant
8 0<|im_end|><|endoftext|>
```

网络从全 261 类词表连续生成结果位、空格、进位和两个结束 token；没有限制数字词表或强制结束。局部回答不用思考标记充当结束。调度器只取位、传递**模型生成的进位**和拼接**模型生成的结果位**；推理不计算结果位，没有算术兜底。

`pos` 只作为轨迹元数据，不送入局部网络。每列运算与位置无关，省掉不必要的位置计数学习，也避免长输入产生未训练的位置字符串。整题呈现：

```text
<think>
pos=0 a=7 b=1 cin=0 -> 8 0
pos=1 a=9 b=7 cin=0 -> 6 1
</think>
```

这是调度器将真实预测排成的轨迹。页面同时保留每次实际收到/生成的角色消息和 token，不把它冒充一次自由生成的完整回答。

## 从零完成六阶段

```powershell
cmake --build ai-add/build -j 4
ai-add/build/chat_run.exe ai-add/web/build/column-v2-run 1 200 3000 3000 3000 400 2000 4 new 1 2 8
```

种子 20261018，SFT batch=4、QAT batch=8；末尾 `2` 是 column-v2，`1` 是完整 chat-v1。上下文 256，局部窗口 64，无新增参数。

| 阶段 | 更新数 | 保留的逐位完整匹配 / 200 |
| --- | ---: | ---: |
| 复制与角色格式预训练，无加法标签 | 200 | 1 |
| SFT 无进位，cin=0 且 a+b<10 | 3000 | 90 |
| SFT 输出进位，cin=0，a/b 全覆盖 | 3000 | 152 |
| SFT 输入进位，全部 200 种状态 | 3000 | 200 |
| CoT RLOO，2→4→10 位轨迹 | 400 | 200 |
| FP4 QAT，全部状态 | 2000 | 200 |

SFT 输入进位第 1200 步首次 200/200。本次模型在 RL 前已经学会局部运算，400 轮 RL 均没有奖励差异，因此跳过更新，不能描述为 RL 提升。另用未收敛 SFT 1 检查点验证了有奖励差异的实际 RLOO 更新，纳入 C/WASM 对照。

每阶段的训练前、最后更新、验证选择保存 `.cn` 和带 column-v2 元数据的 `.aiadd`，位于 `build/column-v2-run`。网页内嵌六个阶段选中版本，可分别测试。QAT 的最佳选择仍为输入 FP32（CE 更低），实际 FP4 最后更新另存，正确率同为 200/200。

FP8 直接转换为 197/200，失败文件保留为 `column-v2-fp8-before-qat.cn`。从实际 FP4 QAT 权重继续 FP8 QAT，25 步恢复 200/200。

FP8 修复命令如下（最多 1000 步，实际第 25 步达到完整正确并停止）：

```powershell
ai-add/build/column_quant.exe ai-add/web/build/column-v2-run/stage-6-last.cn ai-add/web/build/column-v2-export/column-v2-final-fp8.cn 1 1000 8 0.00005
```

| 最终权重（build/column-v2-export） | 字节数 | 本机八后端最小间隔下界 |
| --- | ---: | ---: |
| column-v2-final.cn | 237,044 | 8.73828 |
| column-v2-final-fp8.cn | 84,187 | 5.44919 |
| column-v2-final-fp4.cn | 74,647 | 7.28956 |

FP4 仍为专家 MXFP4、其他矩阵 FP8、敏感参数 FP32 的混合表示。网页默认加载本轮 FP4，不覆盖旧 run-final 文件。

## 原生 C 与实际交付 WASM

`tools/engine-parity.c` 同一份 C 探针分别链接原生引擎和实际交付 WASM；`node web/test-parity.cjs` 从**同一序列化权重**、token、RNG、优化器开始，对照原生 scalar/auto 和 WASM scalar/SIMD128：

- 完整对话及逐位对话每个位置、全部 261 类 logits，包含 CED 与缓存重用路径。
- prompt mask、三步 SFT、三步 FP4 QAT、12 轮含有效梯度的 CoT RLOO。
- 每步 loss、梯度范数、奖励、更新步数，以及全部参数、Adam 一阶矩和二阶矩。
- 实际成品 FP32/FP8/FP4 前向；WASM 优化器文件由原生 C 重载，再次前向。

报告 `build/parity-v2/report.json` 逐项要求误差 ≤ `atol + rtol*max(abs(a),abs(b))`：初始 logits 容差 2e-4/2e-4，更新后 5e-4/5e-4，参数及 Adam 状态 2e-5/2e-4；所有探针 argmax 一致。

最终 16 个模式/后端组合、32 组原生比较：最大 logits 差 8.34465027e-5，最大权重差 3.88734043e-6，5856 个被比较位置的 argmax 全部一致。有信号 RLOO 的 12 轮中确实发生两次有效更新，各路径计数相同。

这是容差内实测等效，不宣称 bit 相同、任意硬件相同或长期训练轨迹逐步相同。独立随机初始化的 libm 舍入可能不同；迁移时使用同一个检查点。后续用原生 C 快训，保存权重和优化器，再用 WASM 交叉评估与演示，不必重复在浏览器完整训练。

## 实际模型检查

- FP32、修复后的 FP8、FP4：scalar/SSE2/SSSE3/SSE4.1/AVX/AVX2/AVX2-FMA/AVX512 八后端均 200/200。
- 原生 FP4：1/2/4/6/10 位各 2000 题，共 10,000 题全对，另有千位连续进位。独立测试种子 20261022。
- 实际交付 WASM 标量/SIMD：每种精度分别 200 种转移、200 道整题及千位进位全对。检查每次角色消息和结束标记，局部无思考 token，整题轨迹只有一对包装。
- Windows CTest、Linux ASan/UBSan/leak 各 15/15。

报告：`build/column-v2-export/evaluation-fp32.json`、`evaluation-fp8.json`、`evaluation-fp4.json`、`build/validation-column-wasm.json`、`validation-column-native.log`、`validation-column-asan.log`。FP32/FP8 原生报告的随机题数为 0，仅检查八后端与千位题；其整题随机检查由 WASM 执行。

对于已经穷举通过的确定性局部策略，每列数字与进位都落在这 200 种状态内，按进位归纳可覆盖任意有限位数的十进制加法，含前导零和最终进位；执行受内存/时间限制。这依赖当前权重、精度、后端、prompt 和调度代码，不能套用到修改后的实验权重或完整自由生成模型。页面仍允许失败模型推理。

取消 CoT 会训练完整算式直接回答作对照，需要另训，不能期待逐位成品取消 CoT 后仍能直接回答。页面展示实际协议与原始输出。
