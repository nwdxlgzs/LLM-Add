# LLM-Add

纯 C11、CPU 加法模型训练实验。包含 Cairn 模型引擎、分阶段训练代码，以及可离线使用的单页 Web 实验室。

**免安装使用**：下载仓库后，用浏览器打开 [ai-add/web/dist/ai-add.html](ai-add/web/dist/ai-add.html)。页面内嵌 WASM 标量 / SIMD 引擎和七阶段模型，支持浏览数据集、查看原始输入输出、自由测试、手动训练、CoT 开关及检查点管理。只复制这一个 HTML 文件也能使用。

当前模型为 6 层、宽度 64、FFN 104，共 **467,612 参数**。阶段依次为数字复制预训练、三个 SFT 课程、步骤 RLOO、QAT、自由草稿 RLOO。最后一阶段仅按最终答案及消息控制符奖励，不检查草稿算法或中间步骤。

## 实际结果

当前选择模型在独立随机测试中：1 / 2 / 4 / 6 / 10 位分别为 20/20、20/20、19/20、18/20、0/20。它尚未学会可靠的十位加法；测试接受长输入不代表任意长度都正确。末次 RL 权重与训练前权重也保留，失败回答可以直接观察。

原生与 sanitizer 检查各 15/15 通过，所测 FP32 C / WASM 输出逐 token 一致；FP4 部分数值对照存在差异，原始报告保留。详见 [训练与架构对比](ai-add/web/OUTCOME_RL.md) 和 [验证说明](ai-add/web/VALIDATION.md)。

## 工程内容

| 路径 | 内容 |
| --- | --- |
| [cairn](cairn/) | C11 引擎、结构体网络配置、CPU 指令集探测、训练与量化 |
| [ai-add](ai-add/) | 加法数据、SFT / RLOO / QAT、原生 CLI 和测试 |
| [ai-add/web](ai-add/web/) | 单页界面、WASM 桥接、构建及验证脚本 |
| [训练记录](ai-add/web/OUTCOME_RL.md) | 七阶段训练结果、深宽比较与独立测试成绩 |

仓库仅发布源码、文档和离线网页成品；七阶段推理模型已内嵌在 HTML 中。所有 `build` 目录、编译缓存、独立权重、优化器断点和原始测试文件均留在本地，不纳入版本控制。本机浏览器里的用户检查点保留在原来源的 IndexedDB，不随 Git 同步；跨设备请在网页下载 `.aiadd` 模型包或训练断点。文档中的 `build/...` 路径指本地训练产物，重新运行脚本后生成。

## 原生构建

需要 C11 编译器、CMake 和 Ninja；不依赖 PyTorch 或 GPU。在仓库根目录运行：

```sh
cmake -S ai-add -B ai-add/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build ai-add/build -j 4
ctest --test-dir ai-add/build --output-on-failure
```

网页用户不需要这些环境。重新构建 WASM 需要 Emscripten；训练脚本与历史协议边界见 [ai-add 说明](ai-add/README.md)、[网页说明](ai-add/web/README.md) 和 [Cairn 说明](cairn/README.md)。
