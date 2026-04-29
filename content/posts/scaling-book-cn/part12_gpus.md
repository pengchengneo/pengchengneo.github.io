---
title: "GPUs"
date: 2026-04-29
draft: false
math: true
weight: 13
---

{{< katex >}}

# 如何理解 GPU（How to Think About GPUs）

[How To Scale Your Model](/scaling-book) 第 12 部分（[Part 11: Conclusion](../part11_conclusion) | The End）

我们在 Google 钟爱 TPU，但 GPU 同样出色。本章深入探讨 GPU 的世界——每块芯片如何工作、它们如何互联，以及这些对 LLM 意味着什么，特别是与 TPU 的对比。虽然 NVIDIA、AMD、Intel 等厂商有多种 GPU 架构，本文聚焦 NVIDIA GPU。本节建立在[第 2 章](https://jax-ml.github.io/scaling-book/tpus/)和[第 5 章](https://jax-ml.github.io/scaling-book/training)的基础上，建议先阅读它们。

**目录**

- [什么是 GPU？（What Is a GPU?）](#什么是-gpu)
  - [内存（Memory）](#内存)
  - [GPU 规格汇总（Summary of GPU specs）](#gpu-规格汇总)
  - [芯片层面 GPU 与 TPU 对比（GPUs vs. TPUs at the chip level）](#芯片层面-gpu-与-tpu-对比)
  - [测验 1：GPU 硬件](#测验-1gpu-硬件)
- [网络（Networking）](#网络)
  - [节点层级（At the node level）](#节点层级)
  - [测验 2：GPU 节点](#测验-2gpu-节点)
  - [节点之上（Beyond the node level）](#节点之上)
  - [测验 3：节点之上](#测验-3节点之上)
- [GPU 上的集合通信如何工作？（How Do Collectives Work on GPUs?）](#gpu-上的集合通信如何工作)
  - [节点内集合通信（Intra-node collectives）](#节点内集合通信)
  - [跨节点集合通信（Cross-node collectives）](#跨节点集合通信)
  - [测验 4：集合通信](#测验-4集合通信)
- [GPU 上 LLM 扩展的 Roofline（Rooflines for LLM Scaling on GPUs）](#gpu-上-llm-扩展的-roofline)
  - [数据并行（Data Parallelism）](#数据并行)
  - [张量并行（Tensor Parallelism）](#张量并行)
  - [专家并行（Expert Parallelism）](#专家并行)
  - [流水线并行（Pipeline Parallelism）](#流水线并行)
  - [示例（Examples）](#示例)
  - [GPU 上 LLM 扩展的总结（TLDR）](#gpu-上-llm-扩展的总结)
  - [测验 5：LLM Roofline](#测验-5llm-roofline)
- [致谢与延伸阅读（Acknowledgements and Further Reading）](#致谢与延伸阅读)
- [附录（Appendix）](#附录)
  - [附录 A：GB200 带来什么变化？](#附录-agb200-带来什么变化)
  - [附录 B：更多网络细节](#附录-b更多网络细节)

## 什么是 GPU？

现代 ML GPU（如 H100、B200）本质上是一堆专门做矩阵乘法的计算核心（称为 **流式多处理器（Streaming Multiprocessors）** 或 **SM**）连接到一条高速内存（称为 **HBM**）。下面是一张示意图：

[![](/scaling-book/assets/gpu/gpu-diagram.png)](/scaling-book/assets/gpu/gpu-diagram.png)

**图：** H100 或 B200 GPU 的抽象布局示意图。H100 有 132 个 SM，而 B200 有 148 个。我们这里宽泛地用"Warp Scheduler"一词来描述 32 个 CUDA SIMD 核心 _以及_ 向其分派工作的调度器。注意它和 TPU 何其相似！

每个 SM 与 TPU 的 Tensor Core 类似，都有一个专用的矩阵乘法核心（不幸也叫做 **Tensor Core**——GPU Tensor Core 是 SM 中的矩阵乘法子单元，而 TPU TensorCore 是包含 MXU、VPU 等组件的总体单元）、一个向量算术单元（称为 **Warp Scheduler**——NVIDIA 没有为它取个好名字，所以我们只能选择最不糟糕的一个。Warp Scheduler 主要是向一组 CUDA 核心分派工作的单元，但我们这里用它来描述控制单元和它所控制的核心组）以及一块快速片上缓存（称为 **SMEM**）。与最多有 2 个独立"Tensor Core"的 TPU 不同，现代 GPU 有超过 100 个 SM（H100 上是 132 个）。每个 SM 远比 TPU Tensor Core 弱，但整个系统更灵活。每个 SM 大致完全独立，所以一个 GPU 可以同时做数百个不同任务。虽然 SM 是独立的，但它们经常被迫协调以达到峰值性能，因为它们共享一个容量受限的 L2 缓存。

让我们更详细地看看 H100 的一个 SM：

[![](/scaling-book/assets/gpu/blackwell-sm.png)](/scaling-book/assets/gpu/blackwell-sm.png)

**图：** H100 SM 示意图（[来源](https://wccftech.com/nvidia-hopper-gh100-gpu-official-5nm-process-worlds-fastest-hpc-chip-80-billion-transistors-hbm3-memory/)），展示 4 个 _子分区（subpartitions）_，每个包含一个 Tensor Core、Warp Scheduler、寄存器文件以及不同精度的 CUDA Cores。底部附近的"L1 Data Cache"就是 256kB 的 SMEM 单元。B200 看起来类似，但增加了相当数量的 Tensor Memory（TMEM）来给庞大的 Tensor Core 喂数据。

每个 SM 被分成 4 个相同的象限，NVIDIA 称之为 **SM 子分区（SM subpartitions）**，每个包含一个 Tensor Core、16k 个 32 位寄存器，以及一个 SIMD/SIMT 向量算术单元 Warp Scheduler，其车道（ALU）NVIDIA 称为 **CUDA Cores**。每个分区的核心组件可以说是 Tensor Core，它执行矩阵乘法并贡献绝大部分 FLOPs/s，但它不是唯一值得关注的组件。

- **CUDA Cores：** 每个子分区包含一组称为 CUDA Cores 的 ALU，做 SIMD/SIMT 向量算术。每个 ALU 通常每个周期可以做 1 个算术操作，例如 f32.add。新的 GPU 支持 FMA（融合乘加）指令，技术上每周期做两个 FLOPs，NVIDIA 残忍地利用这一点把它们公布的规格翻倍。每个子分区包含 32 个 fp32 核心（以及更少的 int32 和 fp64 核心），它们在每个周期都执行相同指令。像 TPU 的 VPU 一样，CUDA cores 负责 ReLU、逐点向量操作和归约（求和）。在引入 Tensor Core 之前的历史上，CUDA cores 是 GPU 的主要组件，用于渲染，包括光线-三角形相交和着色。在如今的游戏 GPU 上，它们仍然承担大量渲染工作，而 TensorCores 用于上采样（DLSS），让 GPU 以较低分辨率渲染（更少像素 = 更少工作）然后用 ML 上采样。

- **Tensor Core（TC）：** 每个子分区有自己的 Tensor Core，这是一个像 TPU MXU 一样的专用矩阵乘法单元。Tensor Core 占 GPU FLOPs/s 的绝大部分（例如在 H100 上，我们有 990 bf16 TC TFLOP/s，而 CUDA cores 仅有 66 TFLOPs/s）。

  - [990 bf16 TFLOPs/s](https://www.nvidia.com/en-us/data-center/h100/) 配 132 个 SM 以 1.76GHz 运行，意味着每个 H100 TC 每周期可做 `7.5e12 / 1.76e9 / 4 ~ 1024` bf16 FLOPs，大约一个 8x8x8 矩乘。NVIDIA 不分享太多 TC 硬件细节，所以这更多是猜测而非确定的事实——当然，它没有说明 TC 是如何实现的。我们知道 V100 每 TC 每周期可做 256 FLOPs。A100 可以做 512，H100 可以做 1024，虽然 B200 细节未公布，看起来大约 2048 FLOPs/TC/周期，因为 `2250e12 / (148 * 4 * 1.86e9)` 约为 2048。更多细节见[这里](https://forums.developer.nvidia.com/t/how-to-calculate-the-tensor-core-fp16-performance-of-h100/244727)。
  - 像 TPU 一样，GPU 可以以更高吞吐量做较低精度矩乘（例如 H100 的 fp8 FLOPs/s 是 fp16 的 2 倍）。低精度训练或推理可以显著更快。
  - 自 Volta 以来每代 GPU 都增大了 TC 的尺寸（[这篇好文](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)）。在 B200 上 TC 已经大到无法在 SMEM 中容纳其输入，所以 B200 引入了一个新内存空间叫 TMEM。在 Ampere 中，Tensor Core 可由单个 warp 喂数据，在 Hopper 中需要整个 SM（warpgroup），在 Blackwell 中由 2 个 SM 共同喂数据。Blackwell 中矩乘也变得如此之大，参数（特别是累加器）已不能装入寄存器内存/SMEM，所以 Blackwell 添加了 TMEM。

**CUDA cores 比 TPU 的 VPU 更灵活：** GPU CUDA cores（自 V100 起）使用所谓 SIMT（_单指令多线程_）编程模型，相比之下 TPU 是 SIMD（_单指令多数据_）模型。像 TPU VPU 中的 ALU 一样，子分区内的 CUDA cores 必须在每个周期执行相同操作（例如，如果一个核心在加两个浮点数，那么子分区中所有其他 CUDA cores 也必须这样做）。然而与 VPU 不同的是，每个 CUDA core（或 CUDA 编程模型中的"线程"）有自己的指令指针，可以独立 _编程_。当同一 warp 中的两个线程被指示执行不同操作时，你实际上 _两个_ 操作都做，把不需要执行分歧操作的核心屏蔽掉。

![](/scaling-book/assets/gpu/warp-divergence.png)

**图：** 一组线程内 warp divergence 的例子（[来源](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)）。空白表示物理 CUDA cores 至少部分停顿。

这在线程层面实现了灵活编程，但代价是如果 warp 分歧太频繁会悄悄降低性能。线程在可访问内存方面也可以更灵活；VPU 只能操作连续内存块，而 CUDA cores 可以访问共享寄存器中的单个浮点数并维护每线程状态。

**CUDA core 调度也更灵活：** SM 运行起来有点像多线程 CPU，因为它们可以并发"调度"许多程序（**warps**，每个 SM 最多 64 个），但每个 _Warp Scheduler_ 在每个时钟周期只执行单个程序。在给定 SM 上调度的 warp 称为"驻留"。Warp Scheduler 自动在活动 warp 之间切换以隐藏 I/O 操作如内存加载。相比之下 TPU 通常是单线程的。

### 内存

除计算单元外，GPU 还有内存层级，最大的是 HBM（主 GPU 内存），然后是一系列较小缓存（L2、L1/SMEM、TMEM、寄存器内存）。

- **寄存器（Registers）：** 每个子分区有自己的寄存器文件，在 H100/B200 上包含 16,384 个 32 位字（每个 SM `4 * 16384 * 4 = 256kiB`），可被 CUDA cores 访问。
  - 每个 CUDA core 一次最多只能访问 256 个寄存器，所以虽然每个 SM 可以调度多达 64 个"驻留 warp"，但如果每个线程使用 256 个寄存器，一次只能装下 8 个（`256 * 1024 / (4 * 32 * 256)`）。

- **SMEM（L1 Cache）：** 每个 SM 有自己的 256kB 片上缓存称为 SMEM，可被程序员控制为"共享内存"或被硬件用作片上缓存。SMEM 用于存储激活和 TC 矩乘的输入。

- **L2 Cache：** 所有 SM 共享（技术上 L2 缓存被分成两半，所以一半 SM 在 H100 上每个可访问 25MB。两半之间有一条链接相连，但带宽较低）一个相对较大的 ~50MB L2 缓存，用于减少主内存访问。
  - 这与 TPU 的 VMEM 大小相似，但 **慢得多** 且不受程序员控制。这导致一些"远程作用"现象，程序员需要修改内存访问模式来确保 L2 缓存被充分利用。L2 缓存被所有 SM 共享这一事实，实际上迫使程序员以相当协调的方式运行 SM，尽管原则上它们是独立单元。
  - NVIDIA 不公布其芯片的 L2 带宽，但已[测量](https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth)约为 5.5TB/s。这大约是 HBM 带宽的 1.6 倍，但它是全双工的，所以有效双向带宽接近 3 倍。相比之下，TPU 的 VMEM 大 2 倍 _且_ 带宽高得多（约 40TB/s）。

- **HBM：** 主 GPU 内存，用于存储模型权重、梯度、激活等。
  - HBM 容量从 Volta 的 32GB 大幅增加到 Blackwell（B200）的 192GB。
  - 从 HBM 到 CUDA Tensor Core 的带宽称为 HBM 带宽或内存带宽，H100 上约 3.35TB/s，B200 上约 9TB/s。

### GPU 规格汇总

下面是近期型号的 GPU 规格汇总。给定 GPU 的 SM 数、时钟速度和 FLOPs 在不同变体之间略有不同。这是内存容量数字：

| GPU | 代际 | 时钟速度 | SMs/芯片 | SMEM 容量/SM | L2 容量/芯片 | HBM 容量/芯片 |
|-----|-----------|-------------|----------|-------------------|-------------------|-------------------|
| V100 | Volta | 1.25GHz/1.38GHz | 80 | 96kB | 6MB | 32GB |
| A100 | Ampere | 1.10GHz/1.41GHz | 108 | 192kB | 40MB | 80GB |
| H100 | Hopper | 1.59GHz/1.98GHz | 132 | 256kB | 50MB | 80GB |
| H200 | Hopper | 1.59GHz/1.98GHz | 132 | 256kB | 50MB | 141GB |
| B200 | Blackwell | ? | 148 | 256kB | 126MB | 192GB |

所有代际每个 SM 都有 256kB 寄存器内存。Blackwell 每个 SM 还增加了 256kB 的 TMEM。下面是每个芯片的 FLOPs 和带宽数字：

| GPU | 代际 | HBM BW/芯片 | FLOPs/s/芯片 (bf16/fp16) | FLOPs/s/芯片 (fp8/int8) | FLOPs/s/芯片 (fp4) |
|-----|-----------|-------------|--------------------------|-------------------------|---------------------|
| V100 | Volta | 9.0e11 | — | — | — |
| A100 | Ampere | 2.0e12 | 3.1e14 | 6.2e14 | — |
| H100 | Hopper | 3.4e12 | 9.9e14 | 2.0e15 | — |
| H200 | Hopper | 4.8e12 | 9.9e14 | 2.0e15 | — |
| B200 | Blackwell | 8.0e12 | 2.3e15 | 4.5e15 | 9.0e15 |

我们排除了 B100，因为它没有量产。NVIDIA 虽然制造了 B100 代际，但它们只短暂销售和生产，据称由于设计缺陷使其无法接近声称的规格运行。它们由于热和功耗问题难以在不降频的情况下达到峰值 FLOPs。某些规格略微取决于 GPU 的具体版本，因为 NVIDIA GPU 不像 TPU 那样标准。

下面是一份对比 GPU 与 TPU 组件的实用速查表：

| GPU | TPU | 是什么？ |
|-----|-----|-------------|
| Streaming Multiprocessor (SM) | Tensor Core | 包含其他单元的核心"细胞" |
| Warp Scheduler | VPU | SIMD 向量算术单元 |
| CUDA Core | VPU ALU | SIMD ALU |
| SMEM (L1 Cache) | VMEM | 快速片上缓存内存 |
| Tensor Core | MXU | 矩阵乘法单元 |
| HBM (aka GMEM) | HBM | 高带宽高容量内存 |

### 芯片层面 GPU 与 TPU 对比

GPU 起源于渲染电子游戏，但自 2010 年代深度学习起飞以来，它们越来越像专用矩阵乘法机器——换句话说，越来越像 TPU。在深度学习浪潮之前，GPU（"图形处理单元"）做的就是图形——主要用于电子游戏。电子游戏用数百万个小三角形表示物体，游戏将这些三角形渲染（或"光栅化"）成 2D 图像，每秒在屏幕上显示 30-60 次（这个频率叫帧率）。光栅化涉及将这些三角形投影到相机的坐标系，并计算哪些三角形与哪些像素重叠，每秒数十亿次。可以想象，这非常昂贵，而这只是开始。然后必须通过组合可能与光线相交的多个半透明三角形的颜色给每个像素着色。GPU 被设计为极快地完成这些操作，注重通用性；你需要同时运行许多不同的 GPU 工作负载（称为"shaders"），没有单一操作占主导。因此，面向消费图形的 GPU 可以做矩阵乘法，但这不是它们的主要功能。在某种程度上，这段历史解释了现代 GPU 为何如此。它们不是纯粹为 LLM 或 ML 模型设计的，而是作为通用加速器，硬件追求一定程度的"通用性"，这既是福也是祸。GPU 应用于新任务时更经常"就能工作"，且不像 TPU 那样依赖好的编译器。但这也使它们更难推理或获得 roofline 性能，因为太多编译器特性会引发瓶颈。

**GPU 更模块化。** TPU 有 1-2 个大 Tensor Core，而 GPU 有数百个小 SM。同样，每个 TC 有一个由 4 个独立可编程 8x128 单元组成的大 VPU（共 4096 个 ALU）；相比之下，H100 有 132 * 4 = 528 个独立 SIMD 单元，每个 32 宽（共 16k 个 ALU）。下面是一份 GPU 与 TPU 的 1:1 对比，突出这一点：

| GPU | TPU | H100 # | TPU v5p # |
|-----|-----|--------|-----------|
| SM (streaming multiprocessor) | Tensor Core | 132 | 2 |
| Warp Scheduler | VPU slots | 528 | 8 |
| SMEM (L1 cache) | VMEM | 32MB | 128MB |
| Registers | Vector Registers (VRegs) | 32MB | 256kB |
| Tensor Core | MXU | 528 | 8 |

模块化上的差异一方面使 TPU 制造更便宜、更易理解，但也将更多负担放在编译器上做正确的事。因为 TPU 有单一控制线程且只支持向量化的 VPU 宽度指令，编译器需要手动流水化所有内存加载和 MXU/VPU 工作以避免停顿。GPU 程序员可以直接启动数十个不同 kernel，每个运行在完全独立的 SM 上。另一方面，那些 kernel 可能性能糟糕，因为它们在抖动 L2 缓存或未能合并内存加载；因为硬件控制了运行时的大部分，很难推理幕后发生了什么。结果，TPU 通常能以更少工作接近峰值 roofline 性能。

**历史上，单个 GPU 比可比 TPU 更强大（也更贵）：** 单个 H200 的 FLOPs/s 接近 TPU v5p 的 2 倍，HBM 是其 1.5 倍。同时，Google Cloud 上 H200 的标价约 \$10/小时，TPU v5p 约 \$4/小时。TPU 通常更依赖将多个芯片联网，而 GPU 较少。

**TPU 有更多快速缓存内存。** TPU 也比 GPU 的 SMEM（+TMEM）有更多 VMEM，这块内存可用于以非常快的方式存储和加载权重和激活。如果你能持续将模型权重存储或预取到 VMEM，这能使 TPU 在 LLM 推理上更快。

### 测验 1：GPU 硬件

下面是一些用以测试上述内容的题目。提供了答案，但最好先尝试回答问题，纸笔在手。

**问题 1 [CUDA cores]：** 一个 H100 有多少 fp32 CUDA cores（ALU）？B200 呢？这与 TPU v5p 中独立 ALU 的数量相比如何？

<details>
<summary>点击查看答案。</summary>

**答案：** H100 有 132 个 SM，每个有 4 个子分区，每个包含 32 个 fp32 CUDA cores，所以共 `132 * 4 * 32 = 16896` 个 CUDA cores。B200 有 `148` 个 SM，所以共 `18944` 个。TPU v5p 有 2 个 TensorCore（通常通过 Megacore 连接），每个有一个 (8, 128) 车道的 VPU 且每车道 4 个独立 ALU，所以 `2 * 4 * 8 * 128 = 8192` 个 ALU。这大约是 H100 向量车道数的一半，运行频率大致相同。
</details>

**问题 2 [向量 FLOPs 计算]**：单个 H100 有 132 个 SM，时钟速度 1.59GHz（最高 1.98GHz boost）。假设每个 ALU 每周期可做一个向量操作。每秒可做多少向量 fp32 FLOPs？带 boost 呢？这与矩乘 FLOPs 相比如何？

<details>
<summary>点击查看答案。</summary>

**答案：** `132 * 4 * 32 * 1.59e9 = 26.9TFLOPs/s`。带 boost 是 33.5 TFLOPs/s。这是[规格表](https://www.nvidia.com/en-us/data-center/h100/)所报数字的一半，因为技术上我们可以一周期做 FMA（融合乘加）算两个 FLOPs，但大多数情况下没用。我们可做 990 bfloat16 矩乘 TFLOPs/s，所以忽略 FMA，Tensor Core 的 FLOPs/s 大约多 30 倍。
</details>

**问题 3 [GPU 矩乘 intensity]：** H100 上峰值 fp16 矩乘 intensity 是多少？B200 呢？fp8 呢？_所谓 intensity 我们是指矩乘 FLOPs/s 与内存带宽的比率。_

<details>
<summary>点击查看答案。</summary>

**答案：** 对 H100，我们有峰值 990e12 fp16 FLOPs 和 3.35e12 字节/s 的带宽。所以临界 intensity 是 `990e12 / 3.35e12 = 295`，与 TPU 的 240 相当接近。对 B200 是 `2250e12 / 8e12 = 281`，非常相似。这意味着，与 TPU 类似，我们需要约 280 的批量大小才能在矩乘中是计算密集型的。

对 H100 和 B200 我们恰好有 2x fp8 FLOPs，所以峰值 intensity 也加倍到 590 和 562，虽然某种意义上若考虑权重也会以 fp8 加载，则保持不变。
</details>

**问题 4 [矩乘运行时]：** 用问题 3 的答案，单个 B200 上 `fp16[64, 4096] * fp16[4096, 8192]` 矩乘要多久？`fp16[512, 4096] * fp16[4096, 8192]` 呢？

<details>
<summary>点击查看答案。</summary>

由上可知，批量大小低于 281 token 时是通信密集型的。所以第一个纯粹是带宽密集型。我们读写 $2BD + 2DF + 2BF$ 字节（`2*64*4096 + 2*4096*8192 + 2*64*8192=69e6`），带宽 `8e12` 字节/s，所以约需 `69e6 / 8e12 = 8.6us`。实际上我们可能只得到部分总带宽，所以可能接近 10-12us。增加批量大小后，我们完全计算密集，所以期望 `T=2*512*4096*8192/2.3e15=15us`。我们再次只期望部分总 FLOPs，所以可能看到接近 20us。
</details>

**问题 5 [L1 缓存容量]：** H100 的总 L1/SMEM 容量是多少？寄存器内存呢？这与 TPU VMEM 容量相比如何？

<details>
<summary>点击查看答案。</summary>

**答案：** 我们每个 SM 有 256kB SMEM 和 256kB 寄存器内存，所以各约 33MB（`132 * 256kB`）。合起来，给我们共约 66MB。这大约是现代 TPU 120MB VMEM 的一半，虽然 TPU 总共只有 256kB 寄存器内存！TPU VMEM 延迟低于 SMEM 延迟，这是 TPU 上寄存器内存不那么关键的一个原因（spill 和 fill 到 VMEM 是廉价的）。
</details>

**问题 6 [计算 B200 时钟频率]：** NVIDIA [报告](https://resources.nvidia.com/en-us-blackwell-architecture)B200 可以执行 80TFLOPs/s 的向量 fp32 计算。鉴于每个 CUDA core 在 FMA（融合乘加）op 中每周期可做 2 FLOPs，估计峰值时钟频率。

<details>
<summary>点击查看答案。</summary>

**答案：** 我们有 148 * 4 * 32 = 18944 个 CUDA cores，所以可做 `18944 * 2 = 37888 FLOPs/周期`。因此 `80e12 / 37888 = 2.1GHz`，一个高但合理的峰值时钟速度。B200 通常液冷，所以更高的时钟周期更合理。
</details>

**问题 7 [估算 H100 add 运行时]：** 用上述数字，计算单个 H100 上将两个 `fp32[N]` 向量相加应耗时多少。同时计算 $T_\text{math}$ 和 $T_\text{comms}$。这个操作的算术 intensity 是多少？如果你能访问，也试着在 PyTorch 或 JAX 中以 `N = 1024` 和 `N=1024 * 1024 * 1024` 运行此操作。比较如何？

<details>
<summary>点击查看答案。</summary>

**答案：** 首先，将两个 `fp32[N]` 向量相加执行 N FLOPs，需要加载 `4 * N * 2` 字节并写回 4 * N 字节，共 `3 * 4 * N = 12N`。计算其比率，我们有 `总 FLOPs / 总字节 = N / 12N = 1 / 12`，相当糟糕。

如上所计算，我们大致可做 33.5 TFLOPs/s boost，忽略 FMA。这只在所有 CUDA cores 都被使用时。对 `N = 1024`，我们 _最多_ 只能用 1024 个 CUDA cores 或 8 个 SM，会更慢（假设我们是计算密集型，大约慢 16 倍）。我们也有 3.35e12 字节/s 的内存带宽。所以峰值硬件 intensity 是 `33.5e12 / 3.35e12 = 10`。值得注意的是这个 intensity 在最近 GPU 代际间保持不变。对 H100 是 33.5 / 3.5，对 B200 是 80 / 8。为何如此尚不清楚，但这是个有趣的观察。所以我们将严重受通信限制。因此运行时就是

$$T = \max(T_\text{comms}, T_\text{math}) = \frac{12 \cdot N}{\text{3.35e12}} = \frac{N}{\text{2.8e11}}$$

对 `N = 65,536`，这约 0.23us。实际中我们在 JAX 中看到约 1.5us 的运行时，这很正常因为我们期望严重受延迟限制。对 `N = 1024 * 1024 * 1024`，我们有约 3.84ms 的 roofline，看到 4.1ms，不错！
</details>

## 网络

网络是 GPU 和 TPU 差异最大的领域之一。如我们所见，TPU 以 2D 或 3D 圆环（torus）连接，每个 TPU 只与邻居相连。这意味着两个 TPU 之间发送消息必须经过每个中间 TPU，迫使我们在网格上只能用统一的通信模式。虽然某些方面不便，但这也意味着每个 TPU 的链路数量恒定，我们可以扩展到任意大的 TPU "pods" 而不损失带宽。

而 GPU 使用更传统的分层基于树的交换网络。8 个 GPU 的集合（GB200 上多达 72 个——node 一词被重载，可能意味着两件事：NVLink 域，即通过 NVLink 互联完全连接的 GPU 集合，或连接到单个 CPU 主机的 GPU 集合。在 B200 之前，这两者通常相同，但在 GB200 NVL72 中，我们有一个有 72 个 GPU 的 NVLink 域，但每个主机仍只连接 8 个 GPU。我们这里用 node 一词指 NVLink 域，但这是有争议的）称为 **节点（nodes）**，通过称为 NVLink 的高带宽互联在 1 跳内相连，这些节点通过附加到每个 GPU 的 NIC 用较低带宽的 InfiniBand（IB）或以太网连接成更大单元（称为 **SU** 或 Scalable Unit）。这些进而可通过更高级别交换机连接成任意大的单元。

![](/scaling-book/assets/gpu/superpod-diagram.png)

**图：** 典型 H100 网络示意图。8 个 GPU 通过 NVSwitch（也叫 NVLink switch）连接成一个节点或 NVLink 域，这些节点通过交换式 InfiniBand 网络互联。H100 在 NVLink 域中每个有约 450GB/s 的出口带宽，每个节点有 400GB/s 进入 IB 网络的出口带宽。

### 节点层级

GPU 节点是个小单元，通常 8 个 GPU（GB200 上多达 72 个），通过全互联、全带宽、低延迟的 NVLink 互联连接。NVLink 被向我描述为类似加强版 PCIe 连接，低延迟和协议开销，但不是为可扩展性/容错设计的，而 InfiniBand 更像以太网，为更大的有损网络设计。每个节点包含若干高带宽 NVSwitch，在所有本地 GPU 之间交换数据包。实际节点级拓扑随时间变化很大，包括每节点交换机数量，但对 H100，我们每节点有 4 个 NVSwitch，GPU 以 `5 + 4 + 4 + 5` 链路模式连接，如图所示：

![](/scaling-book/assets/gpu/nvlink-nodes.png)

**图：** Pascal（P100）以来的节点即 NVLink 域示意图。自 Volta（V100）以来，我们用一组交换机在节点内实现全互联。H100 节点有 4 个 NVSwitch，用 25GB/s 链路连接所有 8 个 GPU。

对 Hopper 代际（NVLink 4.0），每个 NVLink 链路有 25GB/s 全双工（这里全双工意味着每个方向 25GB/s，两个方向相互独立。你可以在链路上总共发送 50GB/s，但每个方向最多 25GB/s）带宽（B200 是 50GB/s），给我们从每个 GPU 到网络的 `18 * 25=450GB/s` 全双工带宽。庞大的 NVSwitch 有多达 64 个 NVLink 端口，意味着带 4 个交换机的 8xH100 节点可处理多达 `64 * 25e9 * 4=6.4TB/s` 的带宽。下面是这些数字随 GPU 代际的变化概览：

| NVLink 代 | NVSwitch 代 | GPU 代 | NVLink 带宽 (GB/s, 全双工) | NVLink 端口/GPU | 节点 GPU 间带宽 (GB/s 全双工) | 节点大小 (NVLink 域) | 每节点 NVSwitch 数 |
|-----------|-------------|---------------|--------------------------------------|-------------------|----------------------------------------------|--------------------------|---------------------|
| **3.0** | **2.0** | Ampere | 25 | 12 | 300 | 8 | 6 |
| **4.0** | **3.0** | Hopper | 25 | 18 | 450 | 8 | 4 |
| **5.0** | **4.0** | Blackwell | 50 | 18 | 900 | 8/72 | 2/18 |

Blackwell（B200）有 8 GPU 节点。GB200NVL72 支持更大的 72 GPU NVLink 域。我们展示 8 和 72 GPU 系统的细节。

### 测验 2：GPU 节点

下面是更多关于网络的 Q/A 题目。我发现这些特别有用，因为它们让你逐步分析实际通信模式。

**问题 1 [H100 节点总带宽]：** 带 4 个交换机的 8xH100 节点中我们每节点有多少总带宽？_提示：_ 同时考虑 NVLink 和 NVSwitch 带宽。

<details>
<summary>点击查看答案。</summary>

**答案：** 我们有 Gen4 4xNVSwitch，每个有 `64 * 25e9=1.6TB/s` 单向带宽。这给我们交换机层面 `4 * 1.6e12=6.4e12` 带宽。然而，注意每个 GPU 只能处理 450GB/s 单向带宽，所以那意味着我们最多有 `450e9 * 8 = 3.6TB/s` 带宽。由于这更小，峰值带宽是 3.6TB/s。
</details>

**问题 2 [对分带宽（Bisection bandwidth）]**：对分带宽定义为网络任何均等划分之间的最小可用带宽。换言之，如果我们将网络分成两等份，两份之间有多少带宽？你能计算一个 8x H100 节点的对分带宽吗？_提示：_ 对分带宽通常包括两个方向的流量。

<details>
<summary>点击查看答案。</summary>

**答案：** 任何均等划分将每半边有 4 个 GPU，每个可向另一半出口 `4 * 450GB/s`。考虑两个方向的流量，这给我们 `8 * 450GB/s` 字节穿过分区，或 3.6TB/s 的对分带宽。这就是 NVIDIA 报告的，例如[这里](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)。
</details>

**问题 3 [AllGather 成本]**：给定一个 B 字节的数组，在 8xH100 节点上（吞吐量限制的）AllGather 要多久？对 bf16[DX, F]（其中 `D=4096`，`F=65,536`）做计算。_回答前值得阅读 TPU 集合通信[章节](https://jax-ml.github.io/scaling-book/sharding/)。这里思考一下，但下一节我们会更多讨论集合通信。_

<details>
<summary>点击查看答案。</summary>

**答案：** 每个 GPU 可出口 450GB/s，每个 GPU 有 $B / N$ 字节（其中 `N=8`，节点大小）。我们可以想象每个节点依次将其字节发送到其他 $N - 1$ 节点，导致总共 (N - 1) 轮，每轮 $T_\text{comms} = (B / (N * W_\text{unidirectional}))$，或 $T_\text{comms} = (N - 1) * B / (N * W_\text{unidirectional})$。这约为 $B / (N * W_\text{uni})$ 或 $B / \text{3.6e12}$，即对分带宽。

对给定数组，我们有 `B=4096 * 65536 * 2=512MB`，所以总时间是 `536e6 * (8 - 1) / 3.6e12 = 1.04ms`。这可能受延迟限制，所以实际中可能比这更长（实际中约 1.5ms）。
</details>

### 节点之上

节点之上，GPU 网络的拓扑标准化程度较低。NVIDIA 发布了一个[参考 DGX SuperPod 架构](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/network-fabrics.html)，使用 InfiniBand 连接比单节点更大的 GPU 集合，但客户和数据中心提供商可自由根据需要定制。例如，Meta 在与此描述显著不同的数据中心网络上训练了 LLaMA-3，使用以太网、3 层交换网络和顶层超额订阅交换机。

下面是参考 1024 GPU H100 系统的示意图，底行的每个盒子是一个单独的 8xH100 节点，有 8 个 GPU、8 个 400Gbps CX7 NIC（每 GPU 一个）和 4 个 NVSwitch。

![](/scaling-book/assets/gpu/h100-superpod.png)

**图：** 参考 1024 H100 DGX SuperPod 示意图，128 个节点（有时 127 个），每个有 8 个 H100 GPU，连接到 InfiniBand scale-out 网络。32 个节点（256 GPU）的集合称为'Scalable Units' 或 SU。leaf 和 spine IB 交换机为节点间提供完整对分带宽。

**Scalable Units：** 每组 32 节点称为一个"Scalable Unit"（或 SU），处于一组 8 个 leaf InfiniBand 交换机之下。这个 SU 有 256 个 GPU，每节点 4 个 NVSwitch，8 个 Infiniband leaf 交换机。所有显示的电缆都是 InfiniBand NDR（50GB/s 全双工），有 64 端口 NDR IB 交换机（也是每端口 50GB/s）。_注意 IB 交换机带宽是 NVSwitch 的 2 倍（64 端口、400 Gbps 链路）。_

**SuperPod：** 整体 SuperPod 然后用 16 个顶级"spine" IB 交换机连接 4 个 SU，给我们 1024 个 GPU、512 个节点级 NVSwitch、32 个 leaf IB 交换机和 16 个 spine IB 交换机，共 512 + 32 + 16 = 560 个交换机。Leaf 交换机以 32 节点为一组连接到节点，所以每 256 GPU 集合有 8 个 leaf 交换机。所有 leaf 交换机连接到所有 spine 交换机。

**我们有多少带宽？** InfiniBand 网络（称为"scale out 网络"）的整体拓扑是 **fat tree**，电缆和交换机保证节点级以上的完整对分带宽（这里 400GB/s）。这意味着如果我们将节点分成两半，每个节点可同时向另一分区中的节点出口 400GB/s。更确切地说，这意味着我们应在 scale out 网络中有大致恒定的 AllReduce 带宽！虽然可能不是这样实现的，但你可以想象在 scale-out 网络中对任意多节点做环归约，因为你可以构造包括每个节点的环。

| 层级 | GPU 数 | 每单元交换机数 | 交换机类型 | 每单元带宽 (TB/s, 全双工) | GPU 间带宽 (GB/s, 全双工) | Fat Tree 带宽 (GB/s, 全双工) |
|-------|------|-------------------|-------------|----------------------------------------|------------------------------------------|---------------------------------------|
| Node | 8 | 4 | NVL | 3.6 | 450 | 450 |
| Leaf | 256 | 8 | IB | 12.8 | 50 | 400 |
| Spine | 1024 | 16 | IB | 51.2 | 50 | 400 |

相比之下，TPU v5p 每条链路有约 90GB/s 出口带宽，或沿 3D torus 所有轴 540GB/s 出口。这不是点对点，所以只能用于受限的、统一的通信模式，但它仍给我们高得多的 TPU 间带宽，可扩展到任意大的拓扑（至少多达 8960 TPU）。

GPU 交换网络理论上可通过添加额外交换机或间接层扩展到任意大小，代价是额外延迟和昂贵的网络交换机。

**要点**：H100 节点内，我们每个 GPU 有完整的 fat tree 带宽 450GB/s，而节点之外，这降到节点间 400GB/s。这对通信原语将是关键。

**GB200 NVL72：** NVIDIA 最近开始生产新的 GB200 NVL72 GPU 集群，将 72 个 GPU 组合在单个 NVLink 域中，GPU 间带宽完整 900GB/s。这些域然后可链接到更大的 SuperPod，IB fat tree 带宽按比例更高（9 倍）。下面是该拓扑示意图：

![](/scaling-book/assets/gpu/gb200-superpod.png)

**图：** 一个 576 GPU 的 GB200 DGX SuperPod 示意图。底层每个机架包含 72 个 GB200 GPU。

数算单个节点的出口带宽（上图橙线），我们有 `4 * 18 * 400 / 8 = 3.6TB/s` 到 leaf 层的带宽，比 H100 多 9 倍（正如节点包含 9 倍多 GPU）。这意味着关键的节点出口带宽 _高得多_，我们的跨节点集合通信带宽实际上可能 _低于_ 节点内。详见[附录 A](#附录-agb200-带来什么变化)。

| 节点类型 | 每节点 GPU 数 | GPU 出口带宽 | 节点出口带宽 |
|-----------|--------------|---------------------|----------------------|
| H100 | 8 | 450e9 | 400e9 |
| B200 | 8 | 900e9 | 400e9 |
| GB200 NVL72 | 72 | 900e9 | 3600e9 |

**要点**：GB200 NVL72 SuperPod 大幅增加给定节点的节点大小和出口带宽，这显著改变我们的 roofline。

### 测验 3：节点之上

**问题 1 [Fat tree 拓扑]：** 用上面的 DGX H100 示意图，在节点级计算整个 1024 GPU pod 的对分带宽。证明每条链路的带宽都被选择以确保完整对分带宽。_提示：确保同时计算链路带宽和交换机带宽。_

<details>
<summary>点击查看答案。</summary>

**答案：** 让我们逐组件分析：

- 首先，每节点有 8x400Gbps NDR IB 电缆连接到 leaf 交换机，给每节点 `8 * 400 / 8 = 400 GB/s` 到 leaf 的带宽。我们有 8 个 leaf 交换机，每个 3.2TB/s（64 个 400 GBps 链路），但只能使用 64 个端口中的 32 个从 SU 入口，所以是 32 节点的 `32 * 400 / 8 = 12.8TB/s`，再次正好 400GB/s。
- 然后在 spine 级我们有 `8 * 16 * 2` 个 400Gbps NDR IB 电缆将每个 SU 连接到 spine，给每个 SU `8 * 16 * 2 * 400 / 8 = 12.8 TB/s` 到 leaf 的带宽。再次，每节点 400GB/s。我们有 16 个 spine 交换机，每个 3.2TB/s，给我们 `16 * 3.2 = 51.2 TB/s`，128 节点再次每节点 400GB/s。

所以如果我们以任何方式对分节点，我们将每 GPU 在它们之间有 400GB/s。每个组件都有恰好所需的带宽以确保 fat tree。
</details>

**问题 2 [扩展到更大 DGX pod]：** 假设我们想在 2048 GPU 而非 1024 上训练。修改上述 DGX 拓扑以处理这一点最简单/最佳的方式是什么？4096 呢？_提示：没有单一正确答案，但尽量降低成本。注意链路容量。[这份](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf)文档可能有帮助。_

<details>
<summary>点击查看答案。</summary>

**答案：** 一个选项是保持 SU 结构不变（8 个交换机下 32 个节点），只添加更多带更多顶级交换机的 SU。我们需要 2 倍多的 spine 交换机，所以我们将有 8 个 SU 配 32 个 spine 交换机，给我们足够带宽。

一个问题是每个 leaf 交换机只有 64 端口，我们在上图中已用完所有。但相反，可以轻松改为每 spine 1x 400 Gbps NDR 电缆而非 2x，给同样总带宽但节省一些端口。

对 4096 GPU，我们实际上端口耗尽，所以需要添加另一层间接，即层级中再加一层。NVIDIA 称这些为"core 交换机"，并构建 4096 GPU 集群配 128 个 spine 交换机和 64 个 core 交换机。你可以做数学证明这给足够的带宽。
</details>

## GPU 上的集合通信如何工作？

GPU 可以执行所有与 TPU 相同的集合通信：ReduceScatter、AllGather、AllReduce 和 AllToAll。与 TPU 不同，这些工作方式根据是在节点级（通过 NVLink）还是之上（通过 InfiniBand）执行而变化。这些集合通信由 NVIDIA 在 [NVSHMEM](https://developer.nvidia.com/nvshmem) 和 [NCCL](https://developer.nvidia.com/nccl)（读"nickel"）库中实现。NCCL 在[此处](https://github.com/NVIDIA/nccl)开源。NCCL 根据延迟要求/拓扑使用各种实现（[详情](https://github.com/NVIDIA/nccl/issues/1415#issuecomment-2310650081)），从这里开始我们将讨论交换树网络上的理论最优模型。

### 节点内集合通信

**AllGather 或 ReduceScatter：** 对节点级的 AllGather 或 ReduceScatter，你可以像 TPU 一样在环周围执行它们，每跳使用完整 GPU 间带宽。任意排序 GPU 并使用完整 GPU 间带宽在环周围发送数组的一部分。你也可以认为每个 GPU 发送其大小为 $\text{bytes} / N$ 的块到其他每个 $N - 1$ 个 GPU，共通信 $(N - 1) * N * bytes / N$ 字节，给出相同答案。每跳成本是 $T_\text{hop} = \text{bytes} / (N * \text{GPU egress bandwidth})$，所以总成本是

$$T_\text{AG or RS comms} = \frac{\text{bytes} \cdot (N - 1)}{N \cdot \text{GPU egress bandwidth}} \rightarrow \frac{\text{bytes}}{\text{GPU egress bandwidth}}$$

你会注意到这与 TPU 上完全相同。对 AllReduce，你可以照常组合 RS + AG，成本两倍。

![](/scaling-book/assets/gpu/all-gather.gif)

**图：** 带宽最优 1D 环 AllGather 算法。对 B 字节，这通过顶级交换机发送 B / X 字节 X - 1 次。

如果你担心延迟（例如数组很小），可以做树归约，先在 2 对中 AllReduce，然后 4 对，然后 8 对，共 $\log(N)$ 跳而非 $N - 1$ 跳，虽然总成本仍相同。

**要点：** 在单节点内 AllGather 或 ReduceScatter 一个 B 字节数组的成本约为 $T_\text{comms} = B * (8 - 1) / (8 * W_\text{GPU egress}) \approx B / W_\text{GPU egress}$。这在 H100 上理论上约 $B / \text{450e9}$，B200 上 $B / \text{900e9}$。除非启用网内归约，AllReduce 是这成本的 2 倍。

**Pop Quiz 1 [AllGather 时间]：** 用 8xH100 节点，450 GB/s 全双工带宽，AllGather(bf16[BX, F]) 要多久？设 $B=1024$，$F=16,384$。

<details>
<summary>点击查看答案。</summary>

**答案：** 我们共有 $2 \cdot B \cdot F$ 字节，450e9 单向带宽。这大约要 $T_\text{comms} = (2 \cdot B \cdot F) / \text{450e9}$，或更准确地 $(2 \cdot B \cdot F \cdot (8 - 1)) / (8 \cdot \text{450e9})$。用所给数值，这给我们大约 $(2 \cdot 1024 \cdot 16384) / \text{450e9} = \text{75us}$，或更准确地 $\text{65us}$。
</details>

**AllToAll：** 节点内 GPU 有全互联，使 AllToAll 嗯，相当容易。每个 GPU 直接发送到目标节点。节点内，对 B 字节，每个 GPU 有 $B / N$ 字节并向 $N - 1$ 个目标节点发送 $(B / N^2)$ 字节，共

$$T_\text{AllToAll comms} = \frac{B \cdot (N - 1)}{W \cdot N^2} \approx \frac{B}{W \cdot N}$$

将此与 TPU 比较，那里成本是 $B / (4W)$。所以单节点内，我们获得理论 2X 运行时加速（$B / 4W$ 对 $B / 8W$）。

对专家混合（MoE）模型，我们经常想做 _稀疏或参差 AllToAll_，保证输出维度上 $N$ 个分片中至多 $k$ 个非零，即 $T_\text{AllToAll} \rightarrow K[B, N]$，其中每个轴上 $N$ 个条目中至多 $k$ 个非零。其成本减少 $k/N$，共约 $\min(k/N, 1) \cdot B / (W \cdot N)$。对 MoE，我们经常独立随机地选非零值，所以有些机会非零少于 $k$ 个，给我们大约 $(N-1)/N \cdot \min(k/N, 1) \cdot B / (W \cdot N)$。真实成本实际上是 $$(1 - \left(\frac{Z - 1}{Z}\right)^K) \cdot \frac{Z - 1}{Z}$$ 即 $K$ 次掷骰的预期不同结果数，但与所给近似非常接近。详见附录。

**Pop Quiz 2 [AllToAll 时间]：** 用 8xH100 节点，450 GB/s 单向带宽，AllToAllX->N(bf16[BX, N]) 要多久？如果我们知道 8 个条目中只有 4 个非零呢？

<details>
<summary>点击查看答案。</summary>

**答案：** 由上知，密集情况下成本是 $B \cdot (N-1) / (W \cdot N^2)$，或 $B / (W \cdot N)$。如果我们知道只有 $\frac{1}{2}$ 条目非填充，我们可发送 $B \cdot k/N / (W \cdot N) = B / (2 \cdot W \cdot N)$，大约总成本一半。
</details>

**要点：** 单节点内 GPU 上 $B$ 字节数组的 AllToAll 成本约为 $T_\text{comms} = (B \cdot (8 - 1)) / (8^2 \cdot W_\text{GPU egress}) \approx B / (8 \cdot W_\text{GPU egress})$。对参差（top-$k$）AllToAll，这进一步减少到 $(B \cdot k) / (64 \cdot W_\text{GPU egress})$。

**实证测量：** 下面是 8xH100 节点上 AllReduce 带宽的实证测量。Algo BW 是测量带宽（字节/运行时），Bus BW 计算为 $2 \cdot W \cdot (8 - 1) / 8$，理论上是实际链路带宽的度量。你会注意到我们确实达到接近 370GB/s，少于 450GB/s 但相当接近，虽然每设备只有约 10GB。这意味着虽然这些估计理论上正确，但需要大消息才能实现。

![](/scaling-book/assets/gpu/gpu-all-reduce-bw.png)

**图：** 8xH100 节点上禁用 SHARP 的 AllReduce 吞吐量。蓝曲线是实证链路带宽，由实证测量计算为 $2 * \text{bytes} * (N - 1) / (N * \text{runtime})$。注意我们没有特别接近声称的 450GB/s 带宽，即使带巨大的 10GB 数组。

这是一个真正的问题，因为它显著复杂化了我们能做的任何理论声称，例如即使在合理大小数组上的 AllReduce，如 LLaMA-3 70B 的 MLP（大小 `bf16[8192, 28672]`，或带 8 路模型分片，`bf16[8192, 3584] = 58MB`）也只能达到约 150GB/s，相比峰值 450GB/s。相比之下，TPU 在小得多的消息大小达到峰值带宽（见附录 B）。

**要点：** 虽然 NVIDIA 声称 H100 NVLink 上带宽约 450GB/s，实际中难以超过 370 GB/s，所以请相应调整上述估计。

**网内归约：** 自 Hopper 代际起，NVIDIA 交换机支持 ["SHARP"（Scalable Hierarchical Aggregation and Reduction Protocol）](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/)，允许"网内归约"。这意味着 _网络交换机本身_ 可以做归约操作并将结果多路复用或"MultiCast"到多个目标 GPU：

![](/scaling-book/assets/gpu/sharp-algorithm.png)

**图：** 没有 SHARP 的 AllReduce 理论成本是 2 倍，因为它必须两次通过每个 GPU。实际中加速只有约 30%（来自 NCCL 2.27.5）。

理论上，这接近将 AllReduce 成本减半，因为它意味着每个 GPU 可向顶级交换机发送数据，交换机本身执行归约并将结果广播到每个 GPU，无需两次出口每个 GPU，同时也减少网络延迟。

$$T_\text{SHARP AR comms} = \frac{\text{bytes}}{\text{GPU egress bandwidth}}$$

注意这是精确的，不差 $1/N$ 因子，因为每个 GPU 先出口 $B \cdot (N - 1) / N$，然后接收其本地分片的部分归约版本（入口 $B/N$），完成归约，再出口 $B/N$，然后入口完全归约结果（入口 $B \cdot (N - 1) / N$），共恰好 $B$ 字节入口。

然而，实际中我们看到 SHARP 启用带宽增加约 30%，相比预测的 75%。这只让我们达到约 480GB/s 有效集合带宽，远非 2 倍。

![](/scaling-book/assets/gpu/sharp-all-reduce-cost.png)

**图：** 节点内启用和未启用 NVIDIA SHARP 的 AllReduce algo 带宽实证测量。增益在峰值约 30% 吞吐量改善，虽然算法上应能达到接近 75% 增益。

**要点：** 理论上，NVIDIA SHARP（在大多数 NVIDIA 交换机上可用）应将 $B$ 字节 AllReduce 成本从约 $2 * B / W$ 降到 $B / W$。然而实际中只看到约 30% 带宽改善。由于纯 AllReduce 在 LLM 中相当少见，这不特别有用。

### 跨节点集合通信

超出节点级时，成本更微妙。对树做归约时，可以认为是从底向上归约，先在节点内，然后在 leaf 级，再在 spine 级，每级用普通算法。特别对 AllReduce，可以看到这让我们总体通信更少数据，因为节点级 AllReduce 后，我们只需向 leaf 出口 $B$ 字节而非 $B * N$。

**这有多昂贵？** 一阶近似下，因为我们有完整对分带宽，AllGather 或 ReduceScatter 的成本大致是缓冲区字节数除以节点出口带宽（H100 上 400GB/s），_无论树归约的任何细节。_

$$T_\text{AG or RS comms} = \frac{\text{bytes}}{W_\text{node egress}} \underset{H100}{=} \frac{\text{bytes}}{\text{400e9}}$$

其中 $W_\text{node}$ egress 对上述 H100 网络通常是 400GB/s（每节点出口 8x400Gbps IB 链路）。最干净的图景是想象在 _集群中每个节点_ 上做环归约。由于 fat tree 拓扑，任何两节点之间总能构造一个 $W_\text{node}$ egress 的环并做正常归约。节点级归约（几乎）永远不是瓶颈，因为它有更高的总带宽和更好延迟，虽然总体成本是

$$T_\text{total} = \max(T_\text{comms at node}, T_\text{comms in scale-out network}) = \max\left[\frac{\text{bytes}}{W_\text{GPU egress}}, \frac{\text{bytes}}{W_\text{node egress}}\right]$$

<details>
<summary>这里有更精确的推导。</summary>

我们可以更精确地注意到，我们实际上是在网络的每层做环归约，可以大部分重叠，所以我们有：

$$T_\text{AG or RS comms} = \text{bytes} \cdot max_\text{depth i}\left[\frac{D_i - 1}{D_i \cdot W_\text{link i}}\right]$$

其中 $D_i$ 是深度 $i$ 的度（深度 $i$ 的子节点数），$W_\text{link i}$ 是连接每个子节点到节点 $i$ 的链路带宽。

用此，我们可计算给定拓扑的可用 AllGather/AllReduce 带宽为 $min_\text{depth i}(D_i * W_\text{link i} / (D_i - 1))$。在上述情况下，我们有：

- **Node：** $D_\text{node}$ = 8，因为节点中有 8 GPU 配 Wlink i = 450GB/s。所以 AG 带宽是 `450e9 * 8 / (8 - 1) = 514GB/s`。
- **Leaf：** $D_\text{leaf}$ = 32，因为 SU 中有 32 节点配 Wlink i = 400GB/s（8x400Gbps IB 链路）。所以带宽是 `400e9 * 32 / (32 - 1) = 413GB/s`。
- **Spine：** $D_\text{spine}$ = 4，因为我们有 4 SU 配 $W_\text{link i}$ = 12.8TB/s（来自 `8 * 16 * 2 * 400Gbps` 链路）。带宽是 `12.8e12 * 4 / (4 - 1) = 17.1TB/s`。

因此总体 AG 或 RS 带宽是 `min(514GB/s, 413GB/s, 17.1TB/s) = 413GB/s`，在 leaf 级，所以实际中 $T_\text{AG or RS comms} = B / \text{413GB/s}$，即即使在最高级我们也有约 413GB/s AllReduce 带宽。对带 SHARP 的 AllReduce，会略低于此（约 400GB/s），因为没有 $(N - 1) / N$ 因子。仍然，450GB/s 和 400GB/s 足够接近作为近似。
</details>

**其他集合通信：** 除非启用 SHARP，AllReduce 仍是上述成本的 2 倍。NVIDIA 也卖支持 SHARP 的 IB 交换机，虽然不是所有提供商都有。AllToAll 跨节点变化很大，因为它们不像 AllReduce 那样"分层"。如果我们想从每个 GPU 向其他每个 GPU 发数据，我们不能利用节点级的完整对分带宽。这意味着如果我们有跨 $M = N / 8$ 节点的 N 路 AllToAll，成本是

$$T_\text{AllToAll comms} = \frac{B \cdot (M - 1)}{M^2 \cdot W_\text{node egress}} \approx \frac{B}{M \cdot W_\text{node egress}}$$

实际上有 50GB/s 而非 400GB/s 带宽。我们从单 H100 节点内的 $B / (8 * \text{450e9})$ 到跨 2 节点的 $B / (2 * \text{400e9})$，超过 4 倍降级。

下面是 1024-GPU DGX H100 SuperPod 架构总结：

| 层级 | GPU 数 | 度（子节点数） | 交换机带宽（全双工，TB/s） | 电缆带宽（全双工，TB/s） | 集合带宽（GB/s） |
|-------|---------------|---------------------|--------------------------------------|-------------------------------------|-----------------------------|
| Node | 8 | 8 | 6.4 | 3.6 | 450 |
| Leaf (SU) | 256 | 32 | 25.6 | 12.8 | 400 |
| Spine | 1024 | 4 | 51.2 | 51.2 | 400 |

我们用术语"集合带宽"描述我们能出口 GPU 或节点的有效带宽。它也是 $\text{对分带宽} * 2 / N$。

**要点：** 节点之上，B 字节 AllGather 或 ReduceScatter 的成本大致是 $B / W_\text{node egress}$，H100 DGX SuperPod 上是 $B / \text{400e9}$，AllReduce 成本两倍除非启用 SHARP。整体拓扑是设计为在任何两对节点间提供恒定带宽的 fat tree。

**当数组沿单独轴分片时的归约：** 考虑如下归约成本

$$\text{AllReduce}_X(A[I_Y, J]\ \{ U_X \})$$

我们在沿另一轴 $Y$ 分片的数组上做 AllReduce。在 TPU 上，此操作总成本相比未分片版本减少 $1 / Y$ 倍，因为每轴发送 $1 / Y$ 数据。在 GPU 上，成本取决于哪个轴是"内"轴（节点内 vs 节点间）以及每个分片是否跨多于单个节点。假设 $Y$ 是内轴，数组共 $\text{bytes}$ 字节，总成本有效减少 $Y$，但只在 $Y$ 跨多节点时：

$$T_\text{comms at node} = \frac{\text{bytes}}{W_\text{GPU egress}} \cdot \frac{1}{\min(Y, D_\text{node})}$$

$$T_\text{comms in scale-out network} = \frac{\text{bytes}}{W_\text{node egress}} \cdot \frac{D_\text{node}}{\max(D_\text{node}, Y)}$$

$$T_\text{total} = \max(T_\text{comms at node}, T_\text{comms in scale-out network})$$

其中 N 是 GPU 数，$D_\text{node}$ 再次是节点中 GPU 数（节点的度）。如你所见，如果 $Y < D_\text{node}$，我们在节点级获胜但通常未见总运行时减少，而如果 $Y > D_\text{node}$，我们获得与跨节点数成比例的加速。

如果想精确描述环归约，树 AllGatherX(AY { UX })（假设 Y 是内轴）的一般规则是

$$T_\text{AR or RS comms} = \text{bytes} \cdot \max_{\text{depth } i}\left[\frac{D_i - 1}{D_i \cdot \max(Y, S_{i-1}) \cdot W_{\text{link } i}}\right]$$

其中 $S_i$ 是 M * N * …，树中级 i 以下子节点的大小。这大致是说我们跨越的 GPU 或节点越多，可用带宽越大，但只在该节点内。

**Pop Quiz 3 [沿 2 轴分片]：** 假设我们想在单 SU（256 芯片）上执行 $\text{AllGather}_X(\text{bf16}[D_X, F_Y])$，$Y$ 是内轴。这作为 $D$、$F$、$Y$ 的函数要多久？

<details>
<summary>点击查看答案。</summary>

**答案：** 我们可分两种情况，Y <= 8 和 Y > 8。当 $Y <= 8$ 时，仍受 leaf 交换机限制，所以答案如往常 $T_\text{comms} = 2 * D * F * (32 - 1) / (32 * 400e9)$。当 Y > 8 时，由上大致

$$T_\text{comms} = \frac{2 \cdot D \cdot F \cdot 256}{Y \cdot \text{12.8e12}} = \frac{2DF}{Y \cdot \text{50GB/s}}$$

对 `D = 8192`，`F = 32,768`，我们有：

![](/scaling-book/assets/gpu/sharded-all-gather-cost.png)

**图：** 内轴跨更多节点时分片 AllGather 的理论成本。

注意如果我们恰好做 8 路模型并行，我们确实将节点级归约成本减少 8 倍但保持总体成本不变，所以是免费的但无助于改善总体带宽。
</details>

**要点：** 当我们有多个分片轴时，外部归约的成本减少为内轴跨越节点数的因子。

### 测验 4：集合通信

**问题 1 [SU AllGather]：** 考虑只有单 SU，M 节点每节点 N GPU。AllGather 期间节点级交换机精确入口和出口多少字节？顶级交换机呢？

<details>
<summary>点击查看答案。</summary>

**答案：** 让我们逐步分析归约组件：

1. 每 GPU 向交换机发送 $B / MN$ 字节，共入口 $NB / MN = B / M$ 字节。
2. 我们向 spine 交换机出口完整 $B / M$ 字节。
3. 我们从 spine 交换机入口 $B * (M - 1) / M$ 字节。
4. 我们出口 $B - B / MN$ 字节 $N$ 次，共 $N * (B - B / MN) = NB - B / M$。

总计是 $B$ 入口和 $BN$ 出口，所以应受出口瓶颈，总时间是 $T_\text{AllGather} = BN / W_\text{node} = B / \text{450e9}$。

对 spine 交换机，数学实际更简单。我们必须 $B / M$ 字节入口 M 次（共 $B$ 字节），然后 $B (M - 1) / M$ 出口 $M$ 次，共出口 $B * (M - 1)$。由于这显著更大，成本是 $T_\text{AllGather} = B \cdot (M - 1) / (M \cdot W_\text{node}) = B \cdot (M - 1) / (M \cdot \text{400e9})$。
</details>

**问题 2 [单节点 SHARP AR]：** 考虑单节点每节点 N GPU。用 SHARP（网内归约）AllReduce 期间交换机精确入口和出口多少字节？

<details>
<summary>点击查看答案。</summary>

**答案：** 如前所述，让我们逐步分析。

1. 每 GPU 发送 $B * (N - 1) / N$ 字节，所以入口 $N * B * (N - 1) / N = B * (N - 1)$。
2. 我们累加部分和，向每 GPU 发回 $B / N$ 字节，所以出口 $N * B / N = B$ 字节。
3. 我们在本地对残差做部分和，然后发回交换机。这共入口 $N * B / N = B$ 字节。
4. 我们捕获所有分片并多播，发送 $B * (N - 1) / N$ 到 $N$ 目的地，共出口 $B * (N - 1) / N * N = B * (N - 1)$。

所以总计是 $B * (N - 1) + B = BN$ 字节入口和出口。这支持总吞吐量精确为 $B / W_\text{egress}$。
</details>

**问题 3 [跨节点 SHARP AR]：** 考虑分片在 N GPU 单节点上的 bf16[DX, FY] 数组。AllReduce(bf16[D, FY] { UX }) 要多久？可假设我们做网内归约。解释如果我们多于单节点会有何不同？

<details>
<summary>点击查看答案。</summary>

**答案：** 我们可以试着修改上述前一题答案。基本上，我们先从每 GPU 出口 $B * (X - 1) / XY$ 字节，然后向每 GPU 发回 $B / XY$，然后将相同量发回交换机，然后将 $B * (X - 1) / XY$ 发回每 GPU。总计 $NB / Y$ 入口和出口，所以总时间 $T_\text{comms} = NB / (Y * N * W_\text{link}) = N * 2DF / (Y * N * W_\text{link}) = 2 * D * F / (Y * W_\text{link})$，所以总时间随 $Y$ 减少。

如果超出单节点，可以做大致相同的归约，但当我们出口节点级交换机时，需要发送所有 B 字节，不只是 $B / Y$。这是因为我们需要保持每个分片分开。
</details>

**问题 4 [Spine 级 AR 成本]：** 考虑相同设置，但 $Y = 256$（所以 AR 在 spine 级发生）。AllReduce 要多久？再次，可自由假设网内归约。

<details>
<summary>点击查看答案。</summary>

**答案：** 这让我们利用 spine 级相当夸张的带宽。我们在 4 节点上有 25.6TB/s 带宽，所以 AllReduce 带宽 6.4TB/s。用 SHARP，这可能少至 `2 * D * F / 6.4e12` 秒。
</details>

**问题 5 [2 路 AllGather 成本]：** 计算恰好 2 节点上 $B$ 字节 AllGather 的精确成本。_确保计算精确成本而非近似，并考虑节点内和跨节点成本。_

<details>
<summary>点击查看答案。</summary>

**答案：** 节点级我们有 $T_\text{comms} = B * 7 / (8 * \text{450e9}) = B / \text{514e9}$，而之外实际上是 $T_\text{comms} = B * (2 - 1) / (2 * \text{400e9}) = B / \text{800e9}$。所以实际上我们受节点级归约限制而非 leaf 级！这激励了例如 DeepSeek v3 做 2 路数据并行。
</details>

## GPU 上 LLM 扩展的 Roofline

现在让我们看看这一切都在为什么铺垫：理解 GPU 上 LLM 扩展的 roofline。这是为了补充[此处](../part5_training)的 TPU 训练章节。如那里所做，目标是查看不同并行策略的总 $T_\text{math}$ 和 $T_\text{comms}$，理解何时 $T_\text{comms} > T_\text{math}$。如前，我们只考虑 MLP 块，操作为

$$\text{MLP}(x) \equiv x[B, D] *_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D]$$

其中 $B$ 是 **以 token 计** 的全局批量大小（即 $B = \text{batch size} \cdot \text{sequence length}$）。

这里我们重现上面的表，显示 GPU 和节点级的有效带宽：

| 节点类型 | 每节点 GPU 数 | GPU 出口带宽 | 节点出口带宽 |
|-----------|--------------|---------------------|----------------------|
| H100 | 8 | 450e9 | 400e9 |
| B200 | 8 | 900e9 | 400e9 |
| GB200 NVL72 | 72 | 900e9 | 3600e9 |

**注意：** GPU 和节点出口带宽都决定我们 LLM 的 roofline。我们用术语 $W_\text{collective}$ 描述 GPU 或节点带宽，取决于我们是在节点级内或之上操作。

让我们像 TPU 那样查看 **数据并行、张量并行、流水线并行、专家并行** 及其组合的计算通信 roofline。本节余下我们将聚焦 H100 roofline 进行具体计算。GB200-NVL72 有相同的总体 roofline，但因为我们有更大的节点出口带宽，有时在节点级会成为瓶颈。

### 数据并行

如前所述，DP 和 ZeRO 分片在反向传播中涉及权重 AllReduce 或 ReduceScatter + AllGather。由于这些成本相同，对纯数据并行或 FSDP（_无网内归约_）要计算密集，每层在反向传播中，X 轴大小为 X：

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF}{W_\text{collective}}$$

所以对 $T_\text{math} > T_\text{comms}$，需要 $B / (XC) > 1 / W_\text{collective}$ 或

$$\frac{B}{X} > \frac{C}{W_\text{collective}}$$

其中 $W_\text{collective}$ 是 GPU 或节点级出口带宽，取决于我们是在节点内或跨节点分片。所以：

- **节点内**，我们只需每 GPU **token** 批量大小 > $\text{990e12} / \text{450e9} = 2200$。
- **SU 内或 spine 级**，BS > $\text{990e12} / \text{400e9} = 2475$。

这比 TPU 上高得多，那里所有三轴的数字是 850。例如在 16000 H100 上训练的 LLaMA-3 至少需要 40M token 的批量大小（参考，他们用了 16M）。在 2048 H800 GPU（带宽更低 300GB/s 而非 H100 的 450GB/s）上训练的 DeepSeek v3 需要 $\text{990e12} / \text{300e9} = 3300$ token/GPU，约 6.7M（实际中，他们用了 4M）。

启用网内归约用纯数据并行，理论上我们有 2 倍 AllReduce 带宽，将这些数字减半。然而实际上好处接近 30%，这只是弥补我们通常难以达到所报数字的事实。此外，因为纯数据并行很少有用，这基本不重要。

**MoE 模型：** 对专家混合（MoE）模型，有 E 个专家每 token k 个专家，这增加到

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot k \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot EDF}{W_\text{collective}}$$

将每 GPU token 批量大小膨胀 $E/k$ 倍，即

$$\frac{B}{X} > \frac{E}{k} \frac{C}{W_\text{collective}}$$

例如，新的 OpenAI OSS 模型 $k=4$、$E=128$，跨节点这增加到 `32 * 2475 = 79,200`，一种荒谬高的数字。

**X 小时会怎样？** 当我们只做例如 2 节点数据并行时，受益于 $(X - 1) / X$ 缩放，给我们

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{N * C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF \cdot (X-1)}{X \cdot W_\text{collective}}$$

其中 X 是节点数，$N = 8 \cdot X$。然后对密集模型有 $B / N > \alpha \cdot (X - 1) / X$，例如 $B / N > \text{1237}$，上述值的一半。出于此原因你会经常看到 2 路数据并行。

**要点：** 数据并行和 ZeRO 分片需要每 GPU 约 2500 token 批量大小才能在 H100 或 B200 上计算密集，假设完美重叠和 FLOPs 利用率。对 MoE 模型，这增加 $E / k$ 倍，即总参数与激活参数的比率。当做少量数据并行时，临界批量大小减小。

### 张量并行

张量并行需要对激活做 AllGather 和 ReduceScatter，需要与 MLP FLOPs 重叠。换言之，前向传播中我们有

$$T_\text{math} = \frac{2\cdot 2 \cdot BDF}{Y \cdot C}$$

$$T_\text{comms} = \frac{2\cdot 2 \cdot BD}{W_\text{collective}}$$

要计算密集给出规则

$$Y < \frac{F \cdot W_\text{collective}}{C}$$

节点内，约 $F / 2200$ 或节点之外 $F / 2475$。对 $F=\text{28000}$ 如 LLaMA-3，约 11 路 TP（或向下取整约 8 路，与节点大小相同）。如上，跨恰好 2 节点时获额外 2X 带宽，所以通常可做 16 路张量并行（$F > 2475 \cdot (Y - 8)$），理论上给我们多达 19 路模型并行。

**要点：** 张量并行轴大小 Y 及前馈维度 F 在 $Y > F / 2475$ 时变为通信密集，通常将我们限制在节点内 TP 或最多 2 节点 TP。

### 专家并行

如上所注，专家混合（MoE）模型带 E 倍多模型权重但只有 k 倍多 FLOPs，使数据并行显著更难。我们可通过沿专家维度分片权重（即 Win[EZ, D, F]）一定程度缓解。要做 MLP 块，需要引入 2x AllToAll 将激活发送到对应专家。

如上所注，AllToAllZ->k([B, D, k]) 跨多节点的成本约为 $T_\text{AllToAll} = 2 \cdot B \cdot D \cdot (Z-8)/Z \min(8 * k / Z, 1)$，所以纯专家并行需要

$$T_\text{math} = \frac{4 \cdot B \cdot k \cdot D \cdot F}{Z \cdot C}$$

$$T_\text{comms} = \frac{4 \cdot B \cdot D \cdot (Z-8)}{W \cdot Z} \cdot \min\left(\frac{8 \cdot k}{Z}, 1\right)$$

我们要么需要 $K > Z/8$ 配 $F > \alpha \cdot (Z - 8)/k$，要么 $Z \gg K$ 配 $F > 8 \cdot \alpha$，其中 $\alpha = C/W$。这给你专家并行可能的两个领域：少量专家并行（约 2 节点）小 $F$，或大 $F$ 和 $Z$ 任意大（最多 E 路专家并行）。

实际中两种情况你都会看到，要么少量专家并行（如有非常小 F 和相对小、受限跨节点专家并行的 DeepSeek v3），要么大 F 模型，可以做大量跨节点 EP 配合 TP。

**要点：** 如果 $F < 8 * C / W_\text{node}$，专家并行可跨 1-2 节点，成本类似（略低于）TP，或如果 $F > 8 * C / W_\text{node}$，可做相当多专家并行（最多 $E$ 节点）成本相对低。

### 流水线并行

流水线并行将层跨节点拆分，通信成本极低，因为我们只是每几层发送小微批激活。历史上流水线遭受"流水线气泡（pipeline bubbles）"，但用新的零气泡流水线方法，通常可以避免。

流水线总通信成本很小：$N_\text{MB}$ 微批和 $N_\text{stages}$，我们有 $T_\text{comms per hop} = 2 \cdot B \cdot D / (W \cdot N_\text{MB})$ 和 $N_\text{MB} + N_\text{stages} - 2$ 跳，所以约

$$T_\text{total PP comms} = \frac{2BD}{W \cdot N_\text{MB}} \cdot (N_\text{MB} + N_\text{stages} - 2)$$

$$T_\text{per-layer comms} \approx 1.5 \cdot \frac{2BD}{W \cdot N_\text{layers}}$$

由于除以 $N_\text{layers}$，这远小于任何其他成本。换言之，从通信角度看，流水线基本免费。那为何不就做流水线？有几个原因：

(1) **代码复杂度：** 流水线不像其他方法那样适合自动并行框架（如 XLA 的 GSPMD）。因为它引入微批以隐藏流水线气泡，改变程序结构，自定义零气泡流水线调度通过要求复杂的前向后向交错而加剧此问题。

(2) **流水线使数据并行和 FSDP 变难：** 不做流水线可能最大原因是它与 FSDP 和数据并行配合不好。ZeRO-3 分片特别工作不佳，因为它要求我们每个微批 AllGather 权重，当我们只有 $B / N_\text{microbatches}$ token 来摊销 AllGather 成本时不可行。此外，反向传播期间，_我们不能 AllReduce 或 ReduceScatter 梯度，直到最后微批通过给定阶段，意味着我们有显著未重叠通信时间。_

![](/scaling-book/assets/gpu/pipeline-bubble.png)

**图：** 一个示例 2 阶段、2 微批流水线。F 表示阶段前向传播，B 是阶段反向传播（成本 2 倍）。G 表示数据并行 AllReduce，可比单微批时间显著长。

(3) **流水线气泡和步骤不平衡：** 如上述（糟糕）流水线调度所示，朴素流水线调度期间易有显著气泡（即浪费计算）。上面，第二阶段在步骤 0 闲置，第一阶段从步骤 2 到 3 闲置，第二阶段在最后步骤再次闲置。虽然我们可通过仔细调度某种程度避免这些，但仍常有些气泡。我们也必须在关键路径上将激活从一阶段传到下一阶段，可增加开销：

![](/scaling-book/assets/gpu/pipeline-transfer.png)

**图：** 显示传输成本（红）的示例流水线。这相对移动阶段并增加流水线气泡开销。

每个问题都有变通方法，但它们倾向于实现复杂、维护困难；流水线仍是相对其他方法通信成本低的技术。

**关于延迟的警告：** 如前所述，GPU 即使消息相当大也难以达到完整 AllReduce 带宽。这意味着即使我们理论上可以例如跨多节点扩展专家并行 AllToAll，我们可能难以达到甚至 50% 总带宽。这意味着我们确实试图保持 TP 或 EP 在较少节点内以最小化延迟开销。

### 示例

**DeepSeek 做什么？** 参考，[DeepSeek V3](https://arxiv.org/abs/2412.19437) 用 2048 H800 GPU 训练，配：

- 64 路专家并行（EP）跨 8 节点
- 16 路流水线并行（PP）
- 2 路 ZeRO-1 数据并行（DP）

他们有稳态批量大小 `4096 * 15360 = 62,914,560` token，或每 GPU 30k token。可以看到这已相当大，但他们的模型也非常稀疏（k=8，E=256）所以需要相当大批量大小。可以看到带 64 路 EP 和 16 路 PP，我们最终共 1024 路模型并行，意味着 AllReduce 在 spine 级做，因为只 2 路，实际上有 $2 / (2 - 1) = 2$ 倍多带宽。这也帮助减少最终数据并行 AllReduce 与最终流水线阶段重叠的成本。

**LLaMA-3 做什么？** LLaMA-3 在 16k GPU 上以 16M token BS 训练，约每 GPU 1k token。他们做：

- 节点内 8 路张量并行（TP）
- 16 路流水线并行（PP）
- 128 路 ZeRO-1 数据并行

这也是密集模型所以这些事一般相当平凡。16 路 PP 将数据并行 AllReduce 成本减少 16 倍，帮助我们减少临界批量大小。

### GPU 上 LLM 扩展的总结

让我们退一步，对到目前为止所学做总体总结：

- **数据并行或 FSDP（ZeRO-1/3）需要每 GPU 约 2500 token 的本地批量大小**，虽然理论上网内归约 + 纯 DP 可某种程度减少。
- **张量并行最多 8 路时计算密集**，但我们缺乏带宽扩展更多而不变成通信密集。这主要将我们限制在单 NVLink 域（即单节点或需要用 GB200NVL72 多达 72 GPU）。
- **任何跨多节点的模型并行形式可进一步减少 FSDP 成本**，所以我们经常想混合 PP + EP + TP 跨多节点并减少 FSDP 成本。
- **流水线并行如果你能处理零气泡流水线代码复杂度并保持批量大小相当大以避免数据并行瓶颈则工作良好。** 流水线通常使 ZeRO-3 不可能（因为你需要每个流水线阶段 AllGather），但你可以做 ZeRO-1。

**高层次上，这给我们 GPU 上分片大模型的配方：**

- 对相对小的密集模型，如果有批量大小则激进 FSDP 工作很好，可能配少量流水线或张量并行如有需要。
- 对更大密集模型，1-2 节点 TP + 多节点 PP + 纯 DP 的某种组合工作良好。
- 对 MoE，上述规则适用，但我们也可做专家并行，通常优于 TP。如果 $F > 8 * C / W_\text{node}$，可做大量多节点专家并行，否则我们限制在约 2 节点 EP。

### 测验 5：LLM Roofline

**问题 1 [B200 roofline]：** B200 DGX SuperPod（**非 GB200 NVL72**）节点内带宽是 2 倍（900GB/s 出口），但 scale-out 网络中带宽相同（400GB/s）（[来源](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-b200/latest/network-fabrics.html)）。总 FLOPs 如上报告。这如何改变模型和数据并行 roofline？

<details>
<summary>点击查看答案。</summary>

**答案：** 我们的 bfloat16 FLOPs/s 从 990 增加到 2250 TFLOPs，2.25 倍增长。带 2 倍带宽，节点内 roofline 大致保持。例如对 TP，临界 intensity 上升到 `2250e12 / 900e9 = 2500`，所以我们有 $Y < F / 2500$ 限制，只略高（除非节点大小增加，否则这不帮我们）。

然而节点之外，缺乏额外带宽实际上使我们更难计算密集！例如对数据并行，临界批量大小增加到 `2250e12 / 400e9 = 5625`，因为我们的 GPU 用相同带宽可做显著更多 FLOPs。

带 72 GPU 节点的 GB200 SuperPod 通过添加更多出口带宽改变这点（[来源](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)）。
</details>

**问题 2 [如何分片 LLaMA-3 70B]：** 考虑 LLaMA-3 70B，bfloat16 训练，fp32 优化器状态用 Adam。

1. 至少，仅存储权重和优化器需要多少 H100？
2. 假设我们想在 4096 H100 GPU 上训练 15T token。假设达到 45% MFU（模型 FLOPs 利用率）。训练要多久？
3. LLaMA-3 70B 有 `F = 28,672`，用约 4M token 批量大小训练。在不变成通信密集的情况下，我们最多可做多少模型并行？这加纯 DP，能否在 4k 芯片上训练 LLaMA-3 同时保持计算密集？ZeRO-3 呢？8 路流水线呢？_注意：考虑通信成本和 GPU 内存使用。_

<details>
<summary>点击查看答案。</summary>

1. 我们权重需 2 字节，优化器状态需 8 字节，所以至少 700GB。配 80GB DRAM，至少需要 9 GPU，或（向上取整）至少 2 个 8xH100 节点。这训练永远不完且不能持有梯度检查点，但是下界。
2. 这将共需 `6 * 70e9 * 15e12 = 6.3e24 bf16 FLOPs`。每 GPU 可做 `990e12` FLOPs，所以 45% MFU 我们可做 1.8e18 FLOPs/s。所以整个要 3.5e6 秒，或 40 天。
3. 节点内，我们有 450GB/s 带宽，所以限制约 `F / 1995 = 28672 / 1995 = 14.372`。由于这不跨 2 节点，实际意味着我们最多 8 路模型并行。
   1. 这就要求我们做 512 路 DP。首先，需要看是否有足够内存。由于模型只 8 路分片，意味着 `700GB / 8 = 87.5GB / GPU`，装不下，所以不行！
   2. 配 ZeRO-3 和 8 路 TP，做 512 路 ZeRO-3。这无内存问题因为我们激进分片所有。每 GPU 批量大小 `4e6 / 4096 = 976`。这相当低，甚至低于纯 DP 限制，且这是该限制的两倍因为我们必须移动权重。所以不行。
   3. 配 8 路流水线，每个模型并行分片现跨 8 节点。如所见，这将 leaf 级 AllGather 成本减少 8 倍，所以总体 AllReduce/AllGather 带宽从 400GB/s 到 `8 * 400GB/s = 3200GB/s`。roofline 然后是 `990e12 / 3200e9 = 309`，所以应可以！只需高效实现流水线。
</details>

**问题 3 [Megatron-LM 超参数]：** 考虑这张来自 [Megatron-LM 仓库](https://github.com/NVIDIA/Megatron-LM)的图，突出他们的高 MFU 数字。

![](/scaling-book/assets/gpu/megatron-hparams.png)

注意他们序列长度处处是 4096。对 16B、70B 和 314B 模型，每 GPU token 批量大小是多少？假设数据并行是最外轴并假设 bfloat16 归约，确定每个理论上是计算密集还是通信密集，是否有更优配置？

<details>
<summary>点击查看答案。</summary>

**答案：** 让我们从每 GPU 批量大小开始。

- **16B**：`192 * 4096 / 192 = 4096` token/GPU
- **70B**：`384 * 4096 / 768 = 2048` token/GPU
- **314B**：`1536 * 4096 / 3072 = 2048` token/GPU

这意味着除第一个外，这些都徘徊在每批 2k token，显著接近我们为 FSDP 计算的临界阈值。我们曾基于 spine 级归约计算该界为 2,472 token / GPU，应大致在此起作用。但对 70B 和 314B，因为我们有 16 和 64 路模型（PP + TP）分片，我们在 spine 级获 2 倍和 8 倍更好吞吐量，意味着应在大约 1k 和 300 token / 步分别计算密集。
</details>

## 致谢与延伸阅读

本章重度依赖许多博学 GPU 专家的帮助，包括：

- Adam Paszke，帮助解释 GPU 上 kernel 编程的现实。
- Swapnil Patil，首次解释 GPU 网络如何工作。
- Stas Bekman，指出 GPU 实证现实经常与所声称规格不同。
- Reiner Pope，帮助澄清 GPU 和 TPU 在硬件级如何比较。
- Frédéric Bastien，对芯片级故事给出详细反馈。
- Nouamane Tazi，他在 GPU 上 LLM 训练的经验帮助改进 roofline 部分。
- Sanford Miller，帮助我理解 GPU 如何联网以及 NVIDIA 规格如何与现场常部署的相比。

关于 GPU 有大量好读物，但我最爱的几个包括：

- [SemiAnalysis 的 NVIDIA Tensor Core 历史](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)：一篇精彩文章描述 GPU 如何从电子游戏引擎转变为 ML 加速器。
- [SemiAnalysis 的 Blackwell 性能分析](https://semianalysis.com/2024/04/10/nvidia-blackwell-perf-tco-analysis/)：值得读以理解下一代 NVIDIA GPU。
- [H100 DGX SuperPod 参考](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf)：枯燥但有用，关于较大 GPU 集群如何联网。[这里](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)是关于 GB200 系统的类似文档。
- [关于 NVLink Switch 的 Hot Chips Talk](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)：关于 NVLink 和 NCCL 集合通信的有趣阅读，特别包括网内归约。
- [DeepSeek-V3 技术报告](https://arxiv.org/pdf/2412.19437)：大型半开放 LLM 训练报告的好例子，描述他们如何挑选分片设置。
- [如何优化 CUDA Matmul](https://siboehm.com/articles/22/CUDA-MMM)：精彩博客描述如何用 CUDA Cores 实现高效矩乘，注重 GPU 缓存一致性。
- [HuggingFace Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：GPU 上 LLM 并行的指南，部分启发了本章。
- [从第一性原理让深度学习 Go Brrrr](https://horace.io/brrr_intro.html)：更聚焦 GPU 和 PyTorch 的 LLM roofline 和性能工程教程。
- [Cornell GPU 架构理解站点](https://cvw.cac.cornell.edu/gpu-architecture)：与本书类似的指南，更具体地比较 GPU 和 CPU 内部。

## 附录 A：GB200 带来什么变化？

Blackwell 引入大量主要网络变化，包括 NVLink 5 总 NVLink 带宽翻倍（900GB/s）。B200 仍有 8 GPU 节点，就像 H100，但 GB200 系统（结合 B200 GPU 和 Grace CPU）引入大得多的 NVLink 域（NVL72 中 72 GPU，理论上多达 576）。这更大的 NVLink 域有效增加节点出口带宽，减少节点级以上集合通信成本。

![](/scaling-book/assets/gpu/b200-node.png)

**图：** 显示 GB200 NVL72 单元如何构造的示意图，配 18 个交换机和 72 GPU。

节点内，这增加的带宽（从 450GB/s 到 900GB/s）没什么大差别，因为我们也将每 GPU 总 FLOPs/s 翻倍。我们的 roofline 大致保持，虽然由于 NVLink 带宽好得多，专家并行变得更容易。

节点之外，事情变化更多。这是来自[此处](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)的 SuperPod 示意图。

![](/scaling-book/assets/gpu/gb200-superpod.png)

**图：** 显示 576 GPU GB200 DGX SuperPod 的示意图。

如所见，每节点出口带宽增加到 `4 * 18 * 400 / 8 = 3.6TB/s`，从 H100 的 400GB/s 上升。这将有效跨节点 roofline 改善约 4 倍，因为我们的 FLOPs/芯片也翻倍。现在我们可能开始担心是否在节点级而非 scale-out 级成为瓶颈。

**Grace Hopper：** NVIDIA 也卖 GH200 和 GB200 系统，将一些 GPU 与 Grace CPU 配对。例如 GH200 有 1 个 H200 和 1 个 Grace CPU，而 GB200 系统有 2 个 B200 和 1 个 Grace CPU。该系统的优势是 CPU 用全带宽 NVLink 连接（称为 NVLink C2C）连接到 GPU，所以 CPU 到 GPU 带宽很高，对将参数卸载到主机 RAM 有用。换言之，对任何给定 GPU，到达主机内存的带宽与到达另一 GPU 的 HBM 相同。

## 附录 B：更多网络细节

下面是 NVLink 4 交换机示意图。共有 64 个 NVLink4 端口（每个使用 2 个物理通道），和处理通道间交换的大型 crossbar。相比之下，TPU 使用带可动态重配置镜面的光交换机。

![](/scaling-book/assets/gpu/nvlink4.png)

**图：** 单个 NVLink4 Switch 的较低层次视图。

每级我们可受可用链路带宽或总交换机带宽限制。

- **节点级：** 节点级我们有 4 * 1.6TB/s = 6.4TB/s 的 NVSwitch 带宽，但 8 个 GPU 每个只能向交换机出口 450GB/s，意味着实际节点内峰值带宽 450e9 * 8 = 3.6TB/s（全双工）。
- **SU/leaf 级：** SU 级，我们有 8 个交换机以全互联方式用 1x400 Gbps Infiniband 连接 32 节点。这给我们 8 * 32 * 400 / 8 = 12.8TB/s 节点出口带宽，交换机级 8 * 1.6TB/s = 12.8TB/s，所以两者精确一致。
- **Spine 级：** spine 级，我们有 16 个交换机用 2x400 Gbps 链路连接 32 个 leaf 交换机，所以有 32 * 16 * 400 * 2 / 8 = 51.2TB/s 出口带宽。16 个交换机给我们 16 * 1.6TB/s = 25.6TB/s 带宽，所以这是该级瓶颈。

每 GPU，节点级给我们 450GB/s GPU 间带宽，SU 级 50GB/s，spine 级 25 GB/s。

**GPU 实证 AR 带宽：**

![](/scaling-book/assets/gpu/gpu-all-reduce-bw.png)

**图：** 8xH100 集群上 AllReduce 带宽（节点内，禁用 SHARP）。

TPU v5p 带宽（1 轴）：

![](/scaling-book/assets/gpu/tpu-all-reduce-bw.png)

**图：** TPU v5p 4x4x4 集群上 AllReduce 带宽（沿一轴）。

这里也是 AllGather 带宽：

![](/scaling-book/assets/gpu/gpu-all-gather-bw.png)

**图：** 8xH100 集群上 AllGather 带宽（节点内）。

![](/scaling-book/assets/gpu/tpu-all-gather-bw.png)

**图：** TPU v5e 8x16 集群上 AllGather 带宽（沿一轴）。

**更多 AllToAll 成本：**

这里我们可比较近似 $\min(K / Z) * (Z - 1) / Z$ 与真实值 $(1 - ((Z - 1) / Z) ** K) * (Z - 1) / Z$。除小 $Z$ 值外它们相似。

![](/scaling-book/assets/gpu/all-to-all-approx.png)

**图：** 分片数增加时参差 AllToAll 近似和真实成本的比较。

### 杂项

\*工作于 Google DeepMind，现于 MatX。

### 引用

为学术上下文归属，请将本工作引用为：

```
Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.
```

或作 BibTeX 条目：

```bibtex
@article{scaling-book,
  title = {How to Scale Your Model},
  author = {Austin, Jacob and Douglas, Sholto and Frostig, Roy and Levskaya, Anselm and Chen, Charlie and Vikram, Sharad
  and Lebron, Federico and Choy, Peter and Ramasesh, Vinay and Webson, Albert and Pope, Reiner},
  publisher = {Google DeepMind},
  howpublished = {Online},
  note = {Retrieved from https://jax-ml.github.io/scaling-book/},
  year = {2025}
}
```
