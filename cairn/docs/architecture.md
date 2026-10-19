# 架构实现与资料对齐

参考资料来自同工作区 `deepseek-v4.1-flash/models/DeepSeek-V4.1-Flash/` 的技术报告、`inference/model.py`、`inference/kernel.py` 和 `inference/engram.py`。上游资料入口为 [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)。Cairn 使用自己的 C 计算图、内存管理和训练实现。

## 已实现的结构语义

| 模块 | 具体行为 |
| --- | --- |
| CED | 下层因果编码器；上层全局 KV 投影最终编码器状态，局部 SWA KV/查询仍来自当前解码器层输入 |
| 低秩 latent attention | `wq_a → RMSNorm → wq_b`；跨头共享 KV；RoPE；可学习零值 sink；输出逆 RoPE；分组低秩输出投影 |
| CSA2 连接 | 每层 SWA；Full/Reindex/Reuse；共享全局 KV 和 index K；共享或重新选择 top-k |
| 压缩器 | 逐通道 learned softmax pooling；完整分组才可见；组首位置 RoPE；不完整分组跨 decode 步保留 |
| 索引器 | 多头 `ReLU(q·k)` 加权和；因果 top-k；解码器候选块按块最大分数筛选，最新部分块始终保留 |
| MoE | `sqrt(softplus(router))`；correction bias 只用于选专家；原始分数混合；多专家时归一化；共享专家；SwiGLU 非对称截断 |
| Single-Pass mHC | 当前子层消费上一个子层预测的 pre mix；当前 post/comb 用于输出；Sinkhorn 约束残差混合 |
| Engram | 每层每阶每头独立质数区间；有界奇数乘法 + XOR 哈希；拼接查表；逐残差分支 key、共享 value；归一化有符号开方门控；不含旧版短卷积 |
| 位置编码 | 相邻通道对 RoPE；压缩层和 SWA-only 分别使用基频；可选 YaRN |
| 低比特 | E4M3FN、E2M1、E8M0 实际编码、分组标度和 nibble 打包；量化权重、缓存与 STE 训练 |

默认 4 层示例的连接是：

```text
字节 embedding → 2 路 residual stream
  Encoder 0: SWA
  Encoder 1: Engram + SWA + Full(global compression=2)
  -------- 最终编码器状态 --------
  Decoder 2: SWA + Full(global compression=1，源为编码器状态)
  Decoder 3: SWA + Reindex(共享 Decoder 2 的 global KV)
  最新 pre mix collapse → RMSNorm → 261 类输出
```

每层注意力和 FFN 都有独立 mHC，FFN 为一个共享专家加四个路由专家中的两个。

mHC 初始 pre mix 为选择第 0 个残差流的 one-hot。混合矩阵使用 `comb[来源流,目标流]`，即输出目标流 r 汇总 `comb[k,r] * residual[k]`；测试包含非对称置换矩阵，防止把这两个轴转置。

## CED 预填充的精确性

普通 `cn_prefill` 逐 token 执行全部层，计算完整位置 logits，并作为缓存一致性参考。

两阶段预填充先运行所有 prompt 的编码器，同时建立全部解码器 global KV。第二阶段运行解码器时，global 可见长度仍受当前查询位置限制，不能读到未来位置。

设解码器层数为 D，窗口包含当前 token、宽度 W。最后位置的精确依赖历史长度为 `1 + D*(W-1)`；精确模式回放这个后缀，并保留各解码层后续解码需要的窗口。早期回放位置允许缺少后缀之前的局部状态，但这些误差不在最终位置及下一步所需状态的依赖范围内。测试比较两阶段结果、后续 decode 和完整序列路径。

有界模式只回放 W 个位置。多层解码器在后缀最前端的局部状态不完整，误差可能传到最后输出；它是显式选择的速度/精度折中，不能声称与完整 prefill 等价。

## Engram 可复现性

默认 token map 为字节 ID 恒等映射，不对单个 UTF-8 字节做 Unicode 词元归一化。自定义 `token_map[]`、`primes[]`、`multipliers[]` 可导入外部定义。哈希乘积受 int64 上界约束，序列起点使用 `pad_id`，上下文重置不允许读到上一个序列。

默认乘数由 C 的 SplitMix64 生成，种子包含模型种子和层号；官方参考使用 NumPy PCG。默认哈希地址因此不是官方地址。若导入布局，必须同时提供一致的 token map、所有乘数和质数，并检查词表及表形状，而不能仅使用相同层号。

## 训练语义与尚未覆盖的部分

语言模型交叉熵会反传到 embedding、查询/KV/压缩器、输出投影、选中专家、共享专家、Engram 表及投影、mHC 和归一化参数。专家 top-k 和稀疏位置选择是离散操作，不对选中位置本身求导。索引器额外使用全局注意力分布的 detach teacher 和 KL 目标训练；这个目标是 Cairn 的可用训练设计，不等于复现未公开的 V4.1 训练流水线。

| 项目 | 当前边界 |
| --- | --- |
| 优化器 | AdamW + clipping + 路由负载偏置；未实现官方 Muon、表/输出头 Sinkhorn 优化器及完整预训练配方 |
| 训练并行 | 单机 CPU，逐序列梯度累积；没有数据/张量/专家/流水线并行、通信重叠或 optimizer sharding |
| 内核 | 运行时多 ISA CPU 分派、SIMD GEMV/量化解码、分块 online softmax、融合 Sinkhorn/SwiGLU、OpenMP；详见 [cpu.md](cpu.md) |
| 混合精度 | FP32 主权重、CPU 模拟低比特前向与 STE；不是官方完整混合精度训练复现，mHC/分组输出矩阵保持 FP32 |
| 官方检查点 | `.cn` 为本引擎格式，没有 safetensors 转换器；261 词表与官方 tokenizer/embedding 不兼容 |
| 规模 | 提供可缩放结构和显式配置限制；不宣称已训练或运行完整 748B 级模型 |
| 其他任务 | 不含视觉塔、MTP/DeepSpec 多 token 预测、OPD 或服务端连续批调度；已增加显式 token 目标以支持 response-only SFT 和策略梯度，实际加法课程/RLOO 见同工作区 `ai-add` 项目 |

这些边界不会影响当前框架从随机初始化进行文本训练、保存、量化和增量推理，但决定了它与官方生产系统之间的距离。
