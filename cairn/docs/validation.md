# 验证与性能记录

记录日期：2026-10-18。测试针对本目录 C 源码；此前下载的 Python 参考实现测试不计入本引擎的验证。

## 构建与功能检查

Windows x64，GCC 14.2.0、CMake 3.31.3、Ninja、OpenMP，Release、通用基线和运行时多版本内核，不使用 `-march=native`。库、CLI、测试和自定义配置例子在 `-Wall -Wextra -Wpedantic` 下构建无警告。

`ctest --test-dir build --output-on-failure` 三项通过：引擎、自定义配置例子、逐 CPU 后端测试。Linux Clang 18.1.8 Release / 无 OpenMP 构建也通过全部三项。引擎覆盖：

1. 全部 256 个字节和 5 个特殊 token 的编码/解码，嵌入 NUL、容量查询和非法 ID。
2. E4M3FN 全有限码点往返、E2M1 所有码点、ties-to-even、FP4 打包向量、奇数维度和 32×32 标度边界。
3. 网络参数、KV/index 源、Engram 质数及 40 层拓扑配置校验；40 层小尺寸模型实际执行与 CED 对照。
4. 完整反向图前向与增量缓存结果一致；压缩边界、窗口绕回、后续 token 变化不影响较早 logits；上下文独立及重置。
5. FP8/FP4 缓存路径、两阶段精确 CED 与后续 decode 一致；近似有界回放结果有限。
6. 每个参数张量中最大梯度元素的中心差分抽查，另对索引器 q/k/权重逐元素差分；离散选择固定，detach teacher 不参与数值差分。
7. 重复序列训练降低交叉熵；保存/加载后的 logits 一致、优化器续训一致；量化打包文件往返一致、冻结推理一致；损坏/截断检查点拒绝加载。
8. 单流、无路由 FFN、Reuse、多 Engram 层及第 0 层 Engram、YaRN、QAT；QAT 与打包权重前向一致；不等长批训练按 token 加权；批内序列隔离及非法优化器输入。

稀疏 top-k 使用有界 min-heap，时间复杂度 `O(n log k + k log k)`。测试用包含并列分数及候选掩码的穷举排名结果对照堆选择，覆盖 k 大于可见位置数。

初版 Windows 重复序列的 45 步 AdamW 检查：交叉熵 **5.546764 → 1.446649**。本轮 Linux GCC 检查构建为 **5.546764 → 1.447836**、梯度最大归一化误差 **0.0177889**，阈值 0.075；分母使用 `max(0.01, |解析梯度|+|数值梯度|)`，并非所有参数的相对误差。差分受 FP32 舍入、极小 embedding 初始化和离散路由边界影响；没有声称穷举验证全部梯度。

Linux WSL Ubuntu / GCC 9.4.0，以 `CAIRN_OPENMP=OFF`、`CAIRN_SANITIZE=ON` 构建。运行时启用 ASan、UBSan 和 leak detection，引擎、CPU 后端测试和自定义例子均退出 0，无检查器报告。无 OpenMP 构建由 CPU 测试显式覆盖标量后备路径和全部本机 ISA。具体命令：

```sh
cmake -S /mnt/d/project/git/LLM-add/cairn -B /tmp/cairn-cpu-asan -G Ninja \
  -DCAIRN_OPENMP=OFF -DCAIRN_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_EXE_LINKER_FLAGS=-no-pie
cmake --build /tmp/cairn-cpu-asan
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:handle_segv=0
export UBSAN_OPTIONS=halt_on_error=1
timeout -k 2 30 /tmp/cairn-cpu-asan/cairn_tests
timeout -k 2 30 /tmp/cairn-cpu-asan/cairn_cpu_tests
timeout -k 2 20 /tmp/cairn-cpu-asan/cairn_custom
```

本 WSL 环境的原 PIE 检查程序曾出现间歇 SIGSEGV、ASan 信号处理递归打印，未产生可定位的插桩报告；上面的非 PIE 检查构建及关闭 SIGSEGV 处理后全部通过。内存、UB 和泄漏插桩保持开启，系统 SIGSEGV 仍导致非零退出；这不等于已证明原信号异常的根因。

逐后端测试已实际运行 scalar、SSE2、SSSE3、SSE4.1、AVX、AVX2、AVX2/FMA、AVX-512。覆盖未对齐指针、0/1/奇数长度与向量尾部、四行 GEMV 尾行、33 行跨标度块、FP4 跨行奇数 nibble、完整码点、全部 midpoint 两侧舍入、量化存储复用/失败保留、无 OSXSAVE/不完整 XCR0 的模拟屏蔽，以及不支持后端的拒绝。最长 70 token 的网络对照开启量化 KV 与 YaRN，窗口 40，跨越 online softmax 的 32-key 块；FP32 logits 对标量最大差异在本次运行约 7e-6，打包权重约 1.2e-7。512 维网络触发大矩阵并行分支，Windows OpenMP 1/4 线程的训练 loss、grad norm 和更新后 logits 逐位一致。

ARM64 NEON/SVE、RISC-V RVV 1.0 及两平台通用源、CLI、CPU 测试已由 Clang 18.1.8 无警告交叉编译为目标对象。复现入口为 `tools/check-cross.sh`，需要对应 sysroot。尚未链接并在 ARM/RISC-V 真机上运行，不能把交叉编译当作这些平台的数值或性能验证。

## CLI 集成检查

已运行示例文件训练 20 步、加载续训 2 步、混合 FP8/FP4 导出、检查模型和贪心生成。默认配置文件大小：

| 文件 | 字节数 |
| --- | ---: |
| FP32 权重及 Adam 状态训练检查点 | 1,568,028 |
| 混合 FP8/FP4 打包推理检查点 | 123,185 |

推理文件不含原始 FP32 主权重和优化器。该语料和步数仅验证命令链路，输出没有语言质量保证。格式 v2 为当前版本，早期开发期间生成的 v1 文件不兼容。

## 本轮运行时多 ISA 基准

同一 i7-11800H，Windows GCC 14.2、1 线程。旧版 `cairn-before-cpu.exe` 为上轮保留的 `-O3 -march=native` 程序；新版为 portable Release，自动选择 AVX-512。固定 token、模型 seed、5 次推理和 3 次训练的 CLI 内部测量不变。外层运行 3 轮、交替新旧程序顺序，以下为每组吞吐的中位数。后台编译和检查已结束，笔记本仍可能因温度/功耗产生波动。

| 维度 / prompt | 项目 | 旧 native | 新 runtime | 变化 |
| --- | --- | ---: | ---: | ---: |
| 32 / 192 | prefill token/s | 3144.4 | 3400.3 | +8.1% |
| 32 / 192 | decode token/s | 3151.4 | 3231.2 | +2.5% |
| 32 / 192 | train token/s | 2128.6 | 2142.1 | +0.6% |
| 128 / 64 | prefill token/s | 1860.5 | 2188.1 | +17.6% |
| 128 / 64 | decode token/s | 1924.6 | 2126.5 | +10.5% |
| 128 / 64 | train token/s | 783.0 | 836.3 | +6.8% |

dim 128 的新版本单线程 prefill：FP32 auto **2188.1**、scalar **801.9**；FP8 auto **1123.1**、scalar **427.3**；混合 FP4 auto **1242.2**、scalar **461.6** token/s。量化路径相对各自标量明显加速，但这个小网络的量化推理仍慢于 FP32，不能据此宣称低比特端到端更快。量化模式的训练数字还含更新后重新打包的成本，不能和纯 FP32 训练直接比较。

1024×1024、预热一次、100 次单线程 GEMV 的实测 GFLOP/s（包含打包数据解码）：

| 内核 | FP32 | FP8 | FP4 |
| --- | ---: | ---: | ---: |
| scalar | 1.864 | 1.283 | 1.555 |
| SSE2 | 16.365 | 2.671 | 2.749 |
| SSSE3 | 15.808 | 2.886 | 3.063 |
| SSE4.1 | 17.796 | 3.329 | 3.439 |
| AVX | 21.758 | 3.034 | 2.725 |
| AVX2 | 21.200 | 6.169 | 6.746 |
| AVX2/FMA | 26.361 | 5.833 | 6.814 |
| AVX-512 | 22.221 | 8.722 | 9.845 |

AVX-512 在这次 FP32 微基准中慢于 AVX2/FMA，说明最高 ISA 不保证每个尺寸都最快；可以通过 API/bench 显式选择。auto 当前按能力优先级分派，没有按 CPU 型号或矩阵尺寸做经验性能调参。

完整数据保存在 [cpu-benchmarks.csv](cpu-benchmarks.csv)、[cpu-kernels.txt](cpu-kernels.txt)，复现：

```powershell
.\tools\benchmark.ps1 -Binary .\build\cairn.exe `
    -Baseline .\build\cairn-before-cpu.exe -Repeats 3 -Output .\docs\cpu-benchmarks.csv
.\build\cairn.exe kernels 1024 100
```

### 初版测量详情

设备：Intel Core i7-11800H，8 核 / 16 逻辑处理器。下表均为 **1 线程**、FP32 主计算、关闭权重和缓存量化，4 层、2 层编码器、窗口 16、2 个注意力头、4 个路由专家中选 2，另加 1 个共享专家。编译为 GCC 14.2.0 `-O3 -march=native`。默认 Engram 和 mHC 开启。

预热后测 5 次推理，训练测 3 次完整更新；decode 在指定 prompt 后生成 32 步，使用固定 token 流。吞吐是累计 token 数除以累计耗时，包含执行图/缓存管理，不包含模型初始化和文件 IO。

| 配置 | 130,176 参数、dim 32、prompt 192 | 1,783,152 参数、dim 128、prompt 64 |
| --- | ---: | ---: |
| 普通精确 prefill，token/s | 3,562.7 | 2,338.7 |
| 增量 decode，token/s | 3,365.3 | 2,234.5 |
| 训练，token/s | 2,346.8 | 923.0 |
| CED 精确两阶段 prefill，token/s | 6,040.7 | 3,105.4 |
| CED 有界近似 prefill，token/s | 6,662.9 | 3,620.4 |
| 训练 arena 已用峰值，MiB | 39.35 | 25.47 |

对应命令：

```powershell
.\build\cairn.exe bench 1 192 32
.\build\cairn.exe bench 1 64 128
```

精确 CED 接口只输出最后位置 logits，普通 prefill 可输出全部位置，工作量不同。arena 指标不含权重、Adam、缓存、slab 预留空间和系统分配器开销。笔记本功耗与并行构建会明显影响结果；这些是一次本机测量，不是跨硬件或大型模型性能承诺。小矩阵通常不会触发 OpenMP 并行，线程数增加也不保证加速。低比特格式在 CPU 上解码计算，未声称其吞吐超过 FP32。
