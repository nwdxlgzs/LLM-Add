# CPU 后端与优化

只使用 CPU。默认构建把通用代码和专用 SIMD 函数放在同一库中，在创建上下文时探测并选择内核。模型保存格式和参数不依赖构建机器的指令集。

## 指令集与运行时分派

| 后端 | FP32 点积 / AXPY / 四行 GEMV | 打包权重点积 | 构建与验证范围 |
| --- | --- | --- | --- |
| scalar | 顺序参考循环 | E4M3FN / E2M1 + 标度 | Windows GCC、Linux GCC/Clang 已运行 |
| sse2 | 128 位，多累加器 | SIMD 位构造解码 | 本机已运行 |
| ssse3 | 128 位 | FP4 字节 shuffle 查表 | 本机已运行 |
| sse4.1 | 128 位 | 字节直接扩展、FP4 shuffle | 本机已运行 |
| avx | 256 位，无 FMA | 两组 128 位解码后合并 | 本机已运行 |
| avx2 | 256 位，无 FMA | FP8 位构造、FP4 permute 查表 | 本机已运行 |
| avx2-fma | 256 位 FMA | 解码融合 FMA | 本机已运行 |
| avx512 | 512 位 FMA，要求 AVX512F | FP8 位构造、FP4 16 项 permute | 本机已运行 |
| neon | AArch64 128 位 FMA | 字节取值、向量解码及累加 | Clang 18 交叉编译通过，未真机运行 |
| sve | 可变向量宽度，掩码尾部 | 每组解码到 32 元素缓冲后向量累加 | Clang 18 交叉编译通过，未真机运行 |
| rvv | RVV 1.0 可变宽度，保留尾部累加器 | 同上 | Clang 18 交叉编译通过，未真机运行 |

完整 x86 多版本构建使用 GCC/Clang 函数 target 属性，MSVC 原生编译目前提供 scalar/SSE2；本次 Windows 验证使用 MinGW GCC。SVE/RVV 由 CMake 检查编译器支持，仅对专用源文件添加 ISA 参数。AArch64 通用源文件不启用 SVE，RISC-V 通用代码保留构建工具链的基线 ISA。不要通过全局 CFLAGS 强制更高 ISA，再把该程序作为通用二进制分发。

x86 使用 CPUID，只有 OSXSAVE 为真时执行 XGETBV。AVX 需要 XMM/YMM 状态，AVX-512 还需要 opmask/ZMM 状态；强制不支持的后端返回失败。ARM Linux 使用 HWCAP/HWCAP2，RISC-V Linux 使用 HWCAP 并尊重线程的 vector OFF 控制。自动模式按可用后端优先级选最高一项，没有隐藏的启动时性能调优。

额外报告 SSE3、SSE4.2、F16C、AVX/AVX512 VNNI、BF16、AMX、SVE2 等能力。这些名称不代表都有专用内核：当前计算保留 FP32 累加，不把 FP8/E2M1 变成 INT8/INT4 来调用 VNNI，不用 BF16 舍入替换 FP32；AMX 未请求线程权限或投入计算。`cpu` 命令分别列出硬件、OS 可用状态、已编译后端及实际可选后端。

```powershell
.\build-cpu\cairn.exe cpu
.\build-cpu\cairn.exe kernels 1024 100
.\build-cpu\cairn.exe bench 1 64 128 avx2-fma f32
.\build-cpu\cairn.exe bench 1 64 128 avx512 fp4
```

`kernels` 是单线程 GEMV 微基准，计入量化解码成本；`GFLOP/s` 以每权重两次浮点运算计算，不包含解码整数运算。`bench` 包含网络、缓存和内存管理，最后两个参数为可选后端和权重格式；其训练部分仍使用 FP32 主权重，量化模式还计入每步更新后重新打包权重的成本，输出标为 `f32+repack`。数据依赖的解码、softmax、路由、图管理使网络速度不会按微基准倍数增长。

```c
#include "cairn_cpu.h"
cn_cpu_info cpu;
cn_cpu_detect(&cpu);
/* model 已由 cn_model_create 创建；先检查返回值。 */
int ok = cn_model_set_backend(model, CN_CPU_AUTO);
cn_context *ctx = cn_context_create(model, 0);
/* 空上下文可以单独选择；模型默认值只影响之后创建的上下文。 */
ok = ok && ctx && cn_context_set_backend(ctx, CN_CPU_SCALAR);
```

分派表是静态只读数据，每个上下文持有独立指针；没有可变全局 CPU 后端开关。修改模型默认值须由调用方与模型操作同步，已有上下文不会随之改变。后端选择不写入检查点。

## 从官方实现移植的原则

参照归档的 [官方 kernel.py](../../deepseek-v4.1-flash/models/DeepSeek-V4.1-Flash/inference/kernel.py)、[model.py](../../deepseek-v4.1-flash/models/DeepSeek-V4.1-Flash/inference/model.py) 和 [engram.py](../../deepseek-v4.1-flash/models/DeepSeek-V4.1-Flash/inference/engram.py)。GPU 的线程块、张量核和异步流水线不能直接映射到 CPU，以下是实际的 C 实现：

| 参考与思路 | CPU 实现 |
| --- | --- |
| `sparse_attn_kernel` 的 online softmax | 推理按 32 个稀疏 KV 分块，维护 running max / sum / weighted value；learnable sink 从零值、单位概率质量起始。省去 heads×keys 概率缓存；训练保留概率供反向。额外 softmax 工作区为 32 个 float，KV 指针列表仍为 O(keys)。 |
| `hc_split_sinkhorn_kernel` | 推理融合 pre/post sigmoid、行 softmax 和交替列/行归一化，保留相同 epsilon、迭代数和矩阵方向。训练仍记录可微操作。 |
| FP4 分组计算、输入复用 | FP8/FP4 直接从打包数据解码到 SIMD 寄存器，按原分组标度累加，无整矩阵反量化副本。FP32 GEMV 一次处理四行、共享输入向量加载。 |
| SwiGLU / MoE 融合 | 只运行被选中专家；推理 gate/up 共用一次激活量化，并融合 clamp、SiLU 和乘法。gate/up 的两个矩阵投影仍分别执行。 |
| CSA2 / 跨层 KV 复用 | Reindex 推理只计算已选候选块内的索引分数；训练保留完整索引评分供蒸馏。打包 KV 在同一 token 的多个消费者间只解码一次，按 arena epoch 失效，写入立即失效。 |
| Engram 随机访问 | 提前计算当前层全部哈希行号，prefetch 后再收集 embedding；不改变哈希地址。 |
| 公共算子与内存复用 | 缓存 RoPE 频率和最近位置的 sin/cos；训练保存独立角度快照。复用同形状量化存储及上下文临时量化缓冲，按操作决定是否清零输出，减少 malloc 和内存写。 |

量化编码也使用等价的分段阈值/位舍入替换逐码搜索，穷举验证全部相邻码点的 midpoint 及两侧值，保持 nearest-even。E8M0 标度直接构造 binary32 位，包含 2^-127 的次正规情况。

## CPU 训练并行

反向的 dW 按输出行分配工作；dX 按 256 列块分配，每个线程独占一段结果并按同样行顺序累加。无需原子加法和不确定顺序的归约。只有矩阵至少 262,144 个元素时启动 OpenMP，前向按 32 输出行分块。默认单线程，不建立嵌套线程池。

当前训练仍是逐 token 的反向图和逐序列梯度累积，未引入跨 token/batch GEMM 调度、NUMA 放置、分布式训练、Muon、MTP 或 speculative decoding。性能记录和验证范围见 [validation.md](validation.md)。
