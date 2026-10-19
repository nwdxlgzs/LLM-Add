# Cairn：纯 C 文本模型训练与推理引擎

C11 实现，使用结构体定义网络；运行时只依赖 C 标准库、数学库和可选 OpenMP。包含可用的反向传播、AdamW、批训练、增量解码、CED 两阶段预填充、Engram、MoE、mHC、FP8/FP4 和检查点。没有视觉塔。

这是可以从随机参数开始训练的 CPU 引擎。架构与数值格式按已收集的 DeepSeek V4.1 文本资料实现，但不是官方权重加载器，也不是 GPU/分布式训练系统。默认 261 词表与官方词表不同，不能直接运行官方预训练权重。细节与范围见 `docs/architecture.md`；实测见 `docs/validation.md`。

## 构建

Windows PowerShell，需要 GCC、CMake 和 Ninja 在 PATH 中：

```powershell
cd D:\project\git\LLM-add\cairn
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Linux：

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

默认构建运行时探测并分派 scalar、SSE2、SSSE3、SSE4.1、AVX、AVX2、AVX2/FMA、AVX-512；AArch64 提供 NEON/可选 SVE，RISC-V 提供可选 RVV。专用 ISA 与基线代码隔离，旧 `CAIRN_NATIVE` 开关已忽略。`CAIRN_OPENMP=OFF` 可禁用 OpenMP。源码没有 Python、PyTorch 或 C++ 运行依赖。后端、官方优化移植、验证范围见 [CPU 文档](docs/cpu.md)。

## 训练、续训、推理

```powershell
# 示例语料只验证流程；20 步不会得到可用的语言模型。
.\build\cairn.exe train examples\train.txt build\demo.cn 20 32 1
.\build\cairn.exe resume build\demo.cn examples\train.txt build\resumed.cn 20 32
.\build\cairn.exe generate build\resumed.cn '<|im_start|>assistant' 64 0.8

# 混合量化：路由专家 MXFP4，其他可量化矩阵 FP8。
.\build\cairn.exe quantize build\resumed.cn build\demo-fp4.cn fp4
.\build\cairn.exe generate build\demo-fp4.cn '<|im_start|>assistant' 64 0
.\build\cairn.exe inspect build\demo-fp4.cn
.\build\cairn.exe cpu
.\build\cairn.exe kernels 1024 100
.\build\cairn.exe bench 1 192 32 auto f32
.\build\cairn.exe bench 1 64 128 avx512 fp4
.\build\cairn_custom.exe
```

`train CORPUS MODEL [步数] [序列长度] [线程数]` 使用默认 130,176 参数配置，按 UTF-8 原始字节读取文件，识别五个特殊字符串。CLI 按 `<|endoftext|>` 截断样本，跨训练调用重置状态；长文按最大训练长度分段，段首不保留前段上下文。续训恢复权重、Adam 一二阶矩、步数和路由平衡偏置；CLI 的语料游标从头开始。学习率、采样和数据调度可通过库接口自行控制。

## 结构体定义网络

公开接口在 `include/cairn.h`。完整的可编译例子在 `examples/custom.c`：

```c
cn_config config;
cn_config_tiny(&config);       /* 初始化全部默认字段 */
config.dim = 64;
config.ff_dim = 96;
config.hc = 4;
config.layer[3].index_source = 2;  /* 复用第 2 层的稀疏选择 */
config.layer[1].engram = (cn_engram_config){
    .enabled = 1, .max_order = 4, .heads = 2,
    .head_dim = 8, .buckets = 127, .pad_id = CN_EOT
};
char error[256];
cn_model *model = cn_model_create(&config, error, sizeof(error));
/* 检查 model 是否为空；返回的配置包含实际生成的质数和哈希乘数。 */
```

逐层控制：

| 配置 | 含义 |
| --- | --- |
| `kv_source=-1, index_source=-1` | 仅滑动窗口注意力 |
| `kv_source=本层, index_source=本层` | Full：生成全局 KV，并选择稀疏位置 |
| `kv_source=较早层, index_source=本层` | Reindex：共享全局 KV/索引 K，重新选择 |
| `kv_source=较早层, index_source=较早索引层` | Reuse：复用 KV 和选中位置 |
| `encoder_layers` | 编码器/解码器分界；解码器全局 KV 来自最终编码器状态 |
| `compression` | KV 所有者的压缩分组；编码器通常 2，解码器必须 1 |
| `experts=0, active_experts=0, shared_experts=1` | 不使用路由的普通 SwiGLU FFN |
| `hc=1` | 单残差流；仍使用可学习混合系数 |
| `qat=1, quant_cache=1` | 权重/激活 STE 量化训练，以及低比特缓存模拟 |

顶层结构体统一规定层宽、注意力维度、专家维度；逐层结构体规定连接拓扑、压缩和 Engram。配置校验会拒绝非法维度、未来层引用、不同缓存之间的索引复用，以及非法 Engram 布局。

对已训练模型调用 `cn_model_set_qat(model, CN_MXFP4)` 或 `CN_FP8` 可启动对应精度的 QAT，更新后自动重打包推理权重；`CN_F32` 关闭 QAT 并清除打包表示。`cn_model_precision()` 和 `cn_model_quantized_parameters()` 用于核对载入文件的混合精度分工。`ai-add` 已提供十位加法模型的实际 FP4 QAT、文件重载认证及完整算式检查。

`cn_config_v41_topology(&config)` 提供官方 40 层、20/20 编解码分界及 Full/Reindex/Reuse 调度，Engram 位于第 1、14 层（均从 0 编号）。宽度、专家数和哈希表规模仍采用适合 CPU 的小尺寸，不是官方完整参数配置。

## Tokenizer

| token | ID |
| --- | --- |
| 任意原始字节 | 0–255 |
| `<|im_start|>` | 256 |
| `<|im_end|>` | 257 |
| `<|endoftext|>` | 258 |
| `<think>` | 259 |
| `</think>` | 260 |

`cn_encode`、`cn_decode` 接受显式字节长度，支持 NUL 和任意二进制。`parse_special=0` 时所有输入按原始字节编码；识别特殊 token 需要显式设置为 1。传入空输出、容量 0 可先查询所需长度；缓冲不足时只写容量内的数据，仍返回所需长度，不自动追加 NUL。`render_special=0` 的解码会跳过特殊 token。字节 tokenizer 允许模型输出不完整 UTF-8；流式调用者应自行缓冲字节序列。

## 训练与上下文接口

`cn_train_step` 对一条序列做完整反向传播和更新。`count` 包含最后一个 target，预测对数为 `count-1`。`cn_train_batch` 接受连续存放的多条不等长序列和 `lengths[]`，按预测 token 数加权平均梯度，并在每条序列之间重置 KV、n-gram 和残差状态。它通过序列间累积梯度实现批训练，不是跨 batch 的大 GEMM 调度器。

`cn_loss_backward` 和 `cn_optimizer_step` 可分开调用。优化器为带梯度裁剪的 AdamW；路由 correction bias 仅控制专家选择，并按负载独立调整。索引 top-k 是离散操作，通过独立的、teacher 已 detach 的注意力分布蒸馏损失训练索引器。

`cn_objective_backward` 支持逐目标 token 的显式 CE/KL/熵系数，用于回答区间 SFT 和带负优势的策略梯度。多个序列可累积梯度后统一更新；`cn_optimizer_reset` 在新训练阶段清空 Adam 状态。完整 CPU 加法课程、CoT RLOO 和认证示例见同工作区的 `ai-add/README.md`。

每个 `cn_context` 有自己的位置、token 历史和缓存。多个推理上下文可交错访问同一静态模型。模型更新、量化、冻结或释放时，调用方必须确保没有正在访问它的上下文；同一个模型的训练操作须串行。

```c
cn_context *ctx = cn_context_create(model, 0);
const float *logits;
cn_forward(ctx, token, &logits);  /* 仅计算新 token；不会重算整个前缀 */
cn_context_reset(ctx);           /* 新序列：同时重置 Engram 和 KV */
cn_context_free(ctx);
```

`cn_prefill` 返回每个位置的精确 logits。`cn_prefill_ced(..., 0, last)` 在完整编码器阶段后，仅回放重建最后输出和后续解码所需的解码器历史；返回最后位置 logits。`cn_prefill_ced(..., 1, last)` 仅回放最后一个窗口，是**近似模式**。两阶段接口要求空的推理上下文。超过 `max_seq` 返回错误，不会默默截断；失败后应重置上下文。

## 量化与持久化

- FP8 为 E4M3FN；密集权重按 32×32 共用 E8M0 标度，激活按 32 通道分组。
- 路由专家使用真实打包的 FP4 E2M1，每字节两个值，每 32 通道一个 E8M0 标度；共享专家保留 FP8。
- `quant_cache=1` 时，局部 SWA KV 使用 FP8，全局 KV 在 RoPE 后使用每 16 通道一个 E4M3 标度的 E2M1，索引 q/k 使用每 32 通道 E8M0 标度的 MXFP4。
- 分组输出投影和 mHC 预测矩阵保留 FP32。推理端 CPU 解码量化数据并以 FP32 累加；这不是原生 FP4 矩阵乘硬件内核，量化不保证更快。
- `cn_model_quantize` 保留主权重以便继续训练；`cn_model_freeze` 释放梯度、优化器和已量化矩阵的 FP32 主权重，该实例随后只能推理。
- `.cn` v2 采用逐字段小端序列化、维度/长度校验、有限值检查、FNV-1a 完整性校验及临时文件替换。校验用于检测损坏，不是密码学签名。
- `cn_save(..., 1)` 保存 FP32 主权重及优化器，可恢复训练；`cn_save(..., 0)` 保存实际打包的推理权重。推理文件不保留训练步数/优化器语义，加载后再训练会从量化重建的主权重和新优化器开始。

优化器存储仅在开始训练时分配。FP32 训练的权重、梯度和 Adam 状态合计约每参数 16 字节，另加激活和缓存。反向图按序列存储，没有激活重计算或外存卸载。单个张量最多 `INT_MAX` 个元素；不能在本实现中直接分配官方 196B 参数 Engram 表。

## 文件

```text
include/cairn.h       公共 C API 和网络结构体
include/cairn_cpu.h   CPU 探测、运行时后端选择
src/cairn.c          张量反向图、网络、缓存、训练、检查点
src/quant.c          FP8 / FP4 编解码与打包
src/cpu*.c           指令集探测、SIMD 点积/AXPY/GEMV/量化内核
src/tokenizer.c      261 token 字节 tokenizer
src/main.c           train/resume/generate/quantize/bench/inspect/cpu/kernels
examples/custom.c    自定义网络的训练与 CED 推理示例
examples/train.txt   流程测试语料
tests/test.c         数值、因果、缓存、训练、持久化等测试
tests/cpu_test.c     各后端数值对照、长注意力、多线程训练
tools/benchmark.ps1  重复基准与 CSV 导出
tools/check-cross.sh ARM/RISC-V 交叉编译检查
docs/cpu.md          CPU 优化、指令集、使用和边界
docs/architecture.md 架构对齐和边界
docs/validation.md   验证及性能记录
```
