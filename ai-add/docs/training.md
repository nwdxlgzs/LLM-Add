# 分阶段训练记录与复现

固定种子 `20261018`。`prepare` 生成 `local.tsv` 全部 200 种局部输入及正确回答，并生成 `curriculum.tsv` 的 2,560 道训练课程算式：1/2/4/6/10 位各 512 道，含全 9 加 1 的长进位。实际训练读取这两个文件；预训练的随机数字流由相同种子确定。

局部状态是**有限训练全集**，认证也覆盖这个全集，不能将 200/200 称为未见局部样本泛化。完整算式组合的空间极大，测试采用不同种子的独立数对；正确性保证依据固定协议的穷举与归纳，而非随机准确率。

| 阶段 | 数据/目标 | 默认上限 | 检查点 |
| --- | --- | --- | --- |
| 初始化 | 随机权重 | — | `00-initial.cn` |
| 预训练 | 随机数字复制序列，全序列 next-token CE；不包含加法标签 | 200 updates × 4 samples | `01-pretrain.cn` |
| SFT 0 | cin=0、a+b<10 | 3,000 updates | `02-sft-0.cn` |
| SFT 1 | cin=0、所有 a/b | 3,000 updates | `02-sft-1.cn` |
| SFT 2 | 全部 200 状态，错误重采样 | 3,000 updates | `02-sft-2.cn` |
| RL | 整个逐位 CoT 的轨迹奖励；2 位→4 位→10 位课程 | 400 updates | `03-rl-last.cn` |
| 选择 | 每 50 步比较认证正确数，再比较正确状态的联合概率 | — | `03-best.cn`、`final.cn` |
| FP4 QAT | 同一网络的量化前向 + STE，均匀/难例各半 | 2,000 updates × 8 samples | `04-qat-best.cn`、`04-qat-last.cn`、`final-fp4.cn` |

前两阶段 SFT 学全当前局部集合便晋级；SFT 2 要求 200/200 且最小 logit 间隔≥0.5。每 200 步检查，若未学全，以 50% 原始均匀状态 + 50% 错误状态继续训练；第 1,000 步后学习率降为 0.0002。为稳定离散专家选择，最后 SFT 阶段停止负载偏置调整，专家本身仍参与训练。上限耗尽也保存结果，但最终不满 200/200 则进程失败，推理拒绝认证。

当前日志中的 `curriculum10_accuracy` 是固定 40 道**训练课程**诊断，不能称为验证集成绩。独立测试由 `evaluate` 在选好模型后执行，不用于选择检查点。

## SFT 与策略梯度

引擎新增 `cn_objective_backward()`，对位置 i 的真实 target token 定义：

```text
L_i = ce_i * [-log p(token_i)]
    + kl_i * KL(p_i || frozen_reference_i)
    - entropy_i * H(p_i)
```

系数为 0 的 prompt 不计算直接损失，但回答仍通过注意力、Engram 等反传到 prompt 表示。系数按样本/回答 token 数显式归一化；多个序列先累积，再调用一次 AdamW。

RL 每组为同一道算式的 4 条独立 on-policy 采样轨迹。采样使用完整 261 类分布、temperature=1，不截 top-k。奖励为：

```text
R = 0.5 * 正确局部状态比例 + 0.5 * 完整CoT与最终答案同时正确
A_i = R_i - mean(R_j, j != i)
L_policy = -A_i * sum_t log p(token_t)
```

非法 digit/carry/end token 使轨迹无效、奖励为 0。验证器检查每列输入数字、进位连续性、局部算式、答案与轨迹一致，以及独立 uint64 真值。完整答案正确但中间轨迹错误不能取得满分。

每组只更新一次，优势与奖励 stop-gradient，不用 PPO 多轮复用。SFT 模型是冻结参考，KL=0.001，熵系数=0.00001；RL 开始重置 Adam 状态、学习率=0.00003。`rl_informative_updates` 记录同组奖励存在差异的更新次数。奖励全部为 1 时策略梯度为 0，仍有正则梯度；不能把执行步数都说成有效奖励学习。

## 保存与验证

阶段 `.cn` 保存优化器和权重，最终 `final.cn` / `final-fp4.cn` 只保存推理所需权重，格式带校验。每阶段日志在 `metrics.csv`，超参数/实际步数在 `training.json`；QAT 另存 `qat-metrics.csv`、`qat-training.json`。不完整检查点不能用于认证。模型可通过 Cairn API 继续训练，但实验驱动当前不提供精确恢复数据迭代位置的断点续跑命令；复现完整训练请使用新目录。

`cn_objective_gradient_check()` 用有限差分检查负优势、KL、熵和 response mask 对各参数张量的梯度。测试还比较新目标与原始全序列 CE、独立计算的回答 CE、全 mask 时的零梯度，以及无效输入拒绝。

## FP4 量化感知训练

```powershell
ai-add/build/ai_add.exe qat ai-add/run-final/final.cn ai-add/run-final 2000 8
```

调用 `cn_model_set_qat(m, CN_MXFP4)`，不改网络配置与参数量。路由专家矩阵前向使用 MXFP4 E2M1/E8M0，其他可量化矩阵使用 FP8，mHC/分组输出投影按引擎规则保留 FP32；激活模拟 FP8，缓存仍为 FP32。反向穿过量化使用 STE，Adam 更新 FP32 主权重，每次更新后重新打包用于推理检查。测试比较 FP8 和 MXFP4 模式的训练前向与实际打包推理，以及更新后的一致性。

重新初始化 Adam 状态，learning_rate=0.00005、beta2=0.99、weight_decay=0、balance_rate=0。超过请求步数的一半后降至 0.00002。每 25 步构造 50% 均匀全集 + 50% 错误或 margin<2 的状态池，并在实际打包推理路径上检查全部 200 种状态。按正确数优先、最小 margin 次之选择检查点；200/200 且最小 margin≥2 时提前结束。当前运行第 25 步 199/200，第 50 步 200/200、margin=2.84558105。

`04-qat-best.cn` 保存未舍入的 FP32 主权重与优化器，`final-fp4.cn` 保存实际打包权重。重新读取最终文件后，不进行重新量化，穷举本机所有可用后端；任一后端失败则 QAT 命令返回失败，CLI `add` 也会拒绝未认证模型。`certify ... keep all` 可独立复查文件表示。
