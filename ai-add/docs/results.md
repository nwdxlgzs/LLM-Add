# 2026-10-18 实际训练与验证

最终模型位于 `run-final/final.cn`，**57,388 参数，237,044 字节（约 231.5 KiB）**。机器是 i7-11800H，Windows GCC 14.2，CPU 自动选择 AVX512，单线程。此次训练耗时 **37.33 秒**，含阶段诊断与检查点保存；不是不同机器的耗时承诺。

打包 FP8 文件为 `run-final/final-fp8.cn`，**84,187 字节（约 82.2 KiB）**；重新从文件加载后，8 个后端全部 200/200 通过，另存 `certificate-fp8-file.json`。例如：`ai_add add final-fp8.cn 9999999999 1` 输出 `10000000000`。

**FP4 已通过 QAT 修复并交付**：`run-final/final-fp4.cn`，**74,647 字节（约 72.9 KiB）**。同一 6 层网络继续训练 50 步，含最终全部后端认证共 3.665 秒。第 25 步为 199/200，第 50 步达到 200/200、最小 logit 间隔 **2.84558105**。文件重新加载后保持原打包表示，8 个本机后端均通过；报告为 `certificate-fp4-file.json`，日志为 `qat-metrics.csv`、`qat-training.json`。

| 阶段结束 | 实际 updates | 全局局部状态正确数 | 备注 |
| --- | --- | --- | --- |
| 初始化 | 0 | 0/200 | 格式也尚未学会 |
| 数字/格式预训练 | 200 | 0/200 | 未使用加法标签 |
| SFT：无进位 | 800 | 58/200 | 已学全当前无进位课程 |
| SFT：单列可进位 | 1,000 | 117/200 | 已学全 cin=0 课程 |
| SFT：含输入进位 | 1,200 | 200/200 | min margin=4.1248641 |
| RLOO 完整课程 | 400 | 200/200 | 最终选择第 100 步检查点 |

SFT 后每列三个正确 token 的平均联合概率为 0.998910501，选择的 RL 检查点为 0.998942678。RL 400 次更新中 **9 次出现组内奖励差异**，平均采样轨迹奖励 0.996375。这个任务在 SFT 后已通过贪心认证，RL 收益很小；不能据此宣称 RL 是保证正确的必要条件。完整轨迹采样没有 100% 正确保证。

## 完整算式检查

模型选择后，独立执行 `evaluate`，没有用这些结果选择权重：

| 集合 | FP32 正确 / 总数 | 实际 FP4 文件正确 / 总数 |
| --- | --- | --- |
| 0..99 两位加法全部组合 | 10,000 / 10,000 | 10,000 / 10,000 |
| 不同种子随机十位范围算式 | 10,000 / 10,000 | 10,000 / 10,000 |
| 零、交换顺序、最高进位、各位置连续进位 | 440 / 440 | 440 / 440 |

随机测试半数为两个恰好十位的数，半数覆盖 `0..9999999999`，均按十位对齐。边界实例 `9999999999+9999999999` 输出 `19999999998`，完整轨迹见 `run-final/example.txt`。

FP4 的独立完整算式报告为 `run-final/evaluation-fp4.json`，最大进位示例为 `run-final/example-fp4.txt`。模型权重只按局部全集选择；完整算式测试不参与选择。全集是数学归纳的基础，不将随机测试通过本身当作保证。

## 后端与量化认证

实际运行的 8 个后端：scalar、SSE2、SSSE3、SSE4.1、AVX、AVX2、AVX2-FMA、AVX512。

| 权重模式 | 各后端认证 | 最小 token logit 间隔 |
| --- | --- | --- |
| FP32 | 均 200/200 | 约 4.09895 |
| FP8 E4M3FN | 均 200/200 | 约 2.73666 |
| MXFP4 直接量化（旧 PTQ 诊断） | 均 196/200，未认证 | 存在错误 |
| MXFP4 QAT 后实际文件 | **均 200/200** | **2.84558105** |

缓存保持 FP32（`quant_cache=0`）。FP8/MXFP4 的非量化矩阵规则沿用 Cairn，包括 mHC 和分组输出投影保持 FP32。FP4 文件实际包含 **20,736 个 MXFP4、30,520 个 FP8、6,132 个 FP32 参数**，不是全模型 4 bit。旧的直接量化失败证书 `certificate-fp4.json` 保留供核对，交付验证使用新的 `certificate-fp4-file.json`。QAT 并未将失败的 FP4 专家改回 FP8。

认证 JSON/逐状态 CSV 在 `run-final/certificate-*.json*`。正确性范围和归纳关系见 [proof.md](proof.md)，随机测试通过本身不构成全输入保证。

## 代码检查

- Windows release CTest：11/11 通过，包括新目标函数梯度、回答 mask、FP32/FP8/FP4 文件认证、FP4 最大进位、实际打包权重检查、拒绝未训练模型、拒绝超过十位输入及原引擎测试。
- Linux GCC 9.4，ASan + UBSan + leak detection：11/11 通过，读取同一个最终检查点重新认证。
- FP8/MXFP4 QAT 与实际打包推理的逐 token logits 一致；更新后的重打包一致性也通过。FP4 文件测试释放量化矩阵的 FP32 副本后仍通过 200/200，验证实际使用了打包权重。
- 负优势/KL/熵的有限差分检查最大相对误差约 0.0226；独立回答交叉熵与全序列 CE 等价检查通过。

Linux 检查使用非 PIE 链接，并关闭 ASan 的 SIGSEGV handler，以规避本环境先前观察到的 PIE/信号递归异常；地址/未定义行为插桩与泄漏检测仍启用。具体构建参数：

```sh
cmake -S ai-add -B /tmp/ai-add-dsv41-asan -G Ninja \
  -DCAIRN_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_EXE_LINKER_FLAGS=-no-pie
cmake --build /tmp/ai-add-dsv41-asan -j 4
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:handle_segv=0 \
UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir /tmp/ai-add-dsv41-asan --output-on-failure
```

`run/` 保留第一次未通过认证的 199/200 结果；`run-v2/` 是修正错误重采样后的中间实验；`run-final/` 使用文档中的学习率、RL 优化器重置及信息量记录，是本次交付的模型。前两个实验也使用同一 6 层 DSV4.1F 网络，没有使用之前错误的 MLP 实现。
