# ai-add：缩小的 DSV4.1F 十位加法模型

这是用现有 Cairn **纯 C11、CPU Transformer 引擎**训练的专用加法模型。网络由 `src/add.c::add_config()` 的 `cn_config` 结构体定义；没有另外设计 MLP，也没有把加法答案写进推理网络。

**免安装 Web 版**：直接用浏览器打开 [`web/dist/ai-add.html`](web/dist/ai-add.html)。单个离线文件包含模型设计、历史数据、真实手动训练和不限十位的实验推理。新建实验使用真实 user/assistant 对话及单个 CoT 包装，独立判对错，保留并测试各阶段；旧逐列协议成品明确标注，未训练与失败模型仍可试。量化增加可选解码权重缓存，但不保证比 FP32 快。复制这个 HTML 给没有开发环境的人即可使用，详见 [网页使用说明](web/README.md)。

当前 `dialogue-v6`：**6 层、宽度 64、FFN 104，467,612 参数**，在 500,000 参数上限内。预置七个阶段；新增阶段 7 为自由草稿 RLOO，只按最终答案与合法消息控制符奖励，草稿内容不评分。SFT 的逐位字段只是示范格式，不是这一阶段的规则。

详见 [自由草稿 RL 与深宽实训](web/OUTCOME_RL.md)。前轮 550,556 参数模型、旧逐列模型和全部旧记录仍保留为历史，不能把旧局部认证当作整式自由生成能力。

历史局部模型（不属于当前整式网页）：`run-final/final.cn`（FP32，237,044 字节）、`run-final/final-fp8.cn`（FP8，84,187 字节）和 **`run-final/final-fp4.cn`（MXFP4 专家混合精度，74,647 字节）**。三者均已认证。FP4 经 50 步量化感知训练修复，实际文件重新加载后，8 个本机 CPU 后端全部 200/200，最小 logit 间隔为 2.84558105。

可直接使用 `build/ai_add.exe add run-final/final-fp4.cn 9999999999 1`（在本目录运行），输出 `10000000000`。FP4 文件包含 20,736 个 MXFP4 权重、30,520 个 FP8 权重和 6,132 个 FP32 参数；沿用原架构精度分工，不是全模型 4 bit。

原生 CLI 的旧验收范围：两个 `0..9999999999` 的自然数，各 1–10 个十进制字符，允许前导零。旧 run-final 成品采用**固定逐位 CoT 协议**：每列调用同一 Transformer 预测结果位、进位和结束标记，由调度器传递预测进位。网页的新实验接口不设十位上限与认证门禁；另可训练同一网络整式直接回答作为无 CoT 对照。接受长输入不等于保证任意实验模型全部算对，页面展示真实答案、失败原因及独立正确答案。

## 构建与分步训练

在 `D:/project/git/LLM-add` 下运行：

```powershell
cmake -S ai-add -B ai-add/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build ai-add/build -j 4
ctest --test-dir ai-add/build --output-on-failure

# 生成全部局部训练状态和 1/2/4/6/10 位课程算式
ai-add/build/ai_add.exe prepare ai-add/my-run
# 预训练 200 步、每个 SFT 课程上限 3000 步、RL 400 步、SFT batch=4
ai-add/build/ai_add.exe train ai-add/my-run 200 3000 400 4
# 同一网络做 FP4 QAT；通过所有可用后端认证才返回成功
ai-add/build/ai_add.exe qat ai-add/my-run/final.cn ai-add/my-run 2000 8

# 认证所有本机可用后端的 200 种状态，再独立检查完整算式
ai-add/build/ai_add.exe certify ai-add/my-run/final-fp4.cn ai-add/my-run/certificate-fp4-file.json keep all
ai-add/build/ai_add.exe evaluate ai-add/my-run/final-fp4.cn ai-add/my-run/evaluation-fp4.json 10000
ai-add/build/ai_add.exe add ai-add/my-run/final-fp4.cn 9999999999 1
```

Linux 使用同样 CMake 命令，去掉 `.exe`。依赖标准 C 库与 libm，训练不依赖 Python、PyTorch 或 GPU。项目通过 `../cairn` 链接原引擎，因此两个目录要放在一起。

也可运行 `tools/run.ps1` 分步执行 `build/data/train/qat/validate`，`all` 包含 QAT 和实际 FP4 文件验证。默认单线程；小矩阵由 Cairn 运行时探测并选择 SIMD 后端。x86 提供 scalar、SSE2、SSSE3、SSE4.1、AVX、AVX2、AVX2-FMA、AVX512，ARM/RISC-V 使用原引擎的 NEON/SVE/RVV 路径；证书只声称实际运行过的后端。

## 训练与正确性

- **预训练**：从随机初始化做数字字节的自回归复制/格式预训练。没有加法标签，不是通用文本预训练。
- **SFT**：依次学习无进位、单列可进位、带输入进位。只对两个回答字节及 `</think>` 求 CE，prompt 保持可反传但不计 loss。最后阶段均匀样本与错误状态各占一半，后期降学习率。
- **CoT RL**：实际采样 2→4→10 位轨迹，验证每列和完整答案，用 RLOO 策略梯度更新；冻结 SFT 参考模型，加 KL 和熵项。不是“正确样本再做 SFT”。
- **FP4 QAT**：保留同一网络，权重与激活模拟实际量化，FP32 主权重用 STE 更新；均匀状态与错误/低间隔状态重采样。每次更新重打包，最终保存并重新加载实际 FP4 文件再认证。
- **认证**：在当前检查点、精度、后端上穷举 `10×10×2=200` 个固定局部 prompt，要求三个自由 argmax token 都正确、间隔为正。根据进位归纳覆盖所有十位数对。修改权重、量化、prompt 或后端后须重新认证。

`add` 命令每次加载模型都先穷举认证；失败则拒绝给答案，**不会使用整数加法兜底**。`add_local`/`add_predict` 是未认证的底层推理接口，调用方应先认证。真值加法只在训练、奖励与测试函数中使用。

`certify` 默认使用 `keep`：保持载入文件的实际表示，不重新打包。显式 `fp32/fp8/fp4` 用于转换精度后检查；这类报告的文件指纹指向输入文件，转换发生在内存。交付文件认证请用 `keep`。

详见 [架构](docs/architecture.md)、[分阶段流程](docs/training.md)、[正确性证明](docs/proof.md)、[实际结果](docs/results.md)。
