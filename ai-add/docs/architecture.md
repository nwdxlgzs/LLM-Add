# 网络结构与原架构的对应关系

`src/add.c::add_config()` 调用 Cairn 的 `cn_config_tiny()`，再用结构体显式定义更小的 6 层配置。所有前向、反向、优化器与检查点均来自 `cairn/src/cairn.c`。57,388 是 `cn_parameter_count()` 实测值，包含 Engram 表和路由偏置。

| 项目 | 本任务配置 |
| --- | --- |
| 词表 | 256 字节 + `<|im_start|>`、`<|im_end|>`、`<|endoftext|>`、`<think>`、`</think>` |
| 主干 | 6 层，前 3 编码器、后 3 解码器；dim=24 |
| 注意力 | 2 头，head_dim=8，RoPE=4，q_rank=8，o_rank=8 |
| 专家 | 每层 2 路由专家、激活 1 个，1 共享专家；FFN=24、SwiGLU |
| mHC | 2 残差流，4 次 Sinkhorn |
| Engram | 第 1 层，2/3/4-gram，2 哈希头，每头 2 通道，buckets=101 |
| SWA/上下文 | window=8，max_seq=32；局部训练样本实际 8 tokens |
| 稀疏索引 | 1 头，dim=4，top-k=4，候选 4 blocks × 2 positions |
| 训练 | FP32 预训练/SFT/RLOO，再做 FP4 QAT；AdamW、梯度裁剪、索引器辅助训练、CPU SIMD |

| 层号 | 分区 | CSA2 模式 | 全局 KV / 索引来源 |
| --- | --- | --- | --- |
| 0 | Encoder | SWA | 无全局 |
| 1 | Encoder | Full + Engram | 自建，compression=2 |
| 2 | Encoder | Reuse | KV=1，index=1 |
| 3 | Decoder | Full | 自建，输入来自最终 encoder，compression=1 |
| 4 | Decoder | Reuse | KV=3，index=3 |
| 5 | Decoder | Reindex | KV=3，index=5 |

这保留了 DSV4.1F 的模块语义和连接种类，但层数、窗口、压缩分组数量和专家数量都为本任务缩减；不是完整 40 层生产拓扑或官方训练配方的等比例复刻。参考实现与本地归档的对应关系见 `cairn/docs/architecture.md`。

## 固定 CoT 协议

对一列 `(a,b,cin)`，模型实际看到 5 个 token：

```text
[<|im_start|>, ascii(a), ascii(b), ascii(cin), <think>]
```

随后在 **261 类全词表**自由 argmax，生成：

```text
[ascii(digit), ascii(cout), </think>]
```

不使用数字候选掩码。下一列重新开始同样的固定 prompt，cin 使用上一列模型生成的 cout。调度器从低位到高位读取输入，最后倒序拼接结果；显示的 `pos=...` 等文本由模板格式化，模板不负责计算 digit/cout。每列生成结束符也经过学习和认证。

此协议通过外部调度控制位置与列顺序，避免小模型还要从整串输入学习位数对齐与任意自然语言格式。它的保证只适用于该固定协议，不能延伸到任意改写的聊天 prompt。
