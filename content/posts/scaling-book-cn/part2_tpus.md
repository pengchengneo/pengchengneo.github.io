---
title: "All About TPUs（TPU 详解）"
date: 2026-04-29
draft: false
math: true
weight: 3
---

{{< katex >}}

# How to Think About TPUs（如何理解 TPU）

本节是 [How To Scale Your Model](/scaling-book) 的第 2 部分（[第 1 部分：Rooflines](../part1_roofline) | [第 3 部分：Sharding](../part3_sharding)）。

本节全面讲解 TPU 的工作原理、它们如何通过网络互联以支持多芯片训练和推理，以及这一切如何影响我们常用算法的性能。即使是 GPU 用户，也能从中收获不少干货！

**目录**

- [What Is a TPU?（什么是 TPU？）](#what-is-a-tpu)
- [TPU Networking（TPU 网络）](#tpu-networking)
- [Key Takeaways（核心要点）](#key-takeaways)
  - [TPU specs（TPU 规格）](#tpu-specs)
- [Worked Problems（习题演练）](#worked-problems)
- [Appendix（附录）](#appendix)
  - [Appendix A: More on TPU internals（附录 A：TPU 内部机制详解）](#appendix-a-more-on-tpu-internals)
  - [Appendix B: How does a systolic array work?（附录 B：脉动阵列如何工作？）](#appendix-b-how-does-a-systolic-array-work)

你可能也会喜欢阅读关于 NVIDIA GPU 的新增 [第 12 节](../part12_gpus)！

## What Is a TPU?（什么是 TPU？）

TPU 本质上是一个专门做矩阵乘法的计算核心（即 TensorCore），连接到一组高速内存（high-bandwidth memory，HBM）。下图展示了它的结构：

![](https://jax-ml.github.io/scaling-book/assets/img/tpu-chip.png)

**图：** TPU 芯片的基本组件。TensorCore 是左侧灰色方框，包含矩阵乘法单元（MXU）、向量单元（VPU）和向量内存（VMEM）。

TensorCore 可以被理解为一个高效的矩阵乘法机器，但它还有几项其他重要功能。TensorCore 包含三个关键单元：

- **MXU**（Matrix Multiply Unit，矩阵乘法单元）是 TensorCore 的核心。在大多数 TPU 代际中，它使用脉动阵列（systolic array）每 8 个周期执行一次 `bfloat16[8,128] @ bf16[128,128] -> f32[8,128]` 矩阵乘法（TPU v6e/Trillium 拥有 256x256 的 MXU，而所有更早的代际使用 128x128）。详见 [附录 B](#appendix-b-how-does-a-systolic-array-work)。
  - 在 TPU v5e 上，每个 MXU 在 1.5GHz 频率下大约有 `5e13` bf16 FLOPs/s。大多数 TensorCore 含有 2 或 4 个 MXU，因此 TPU v5e 的总 bf16 FLOPs/s 为 `2e14`。
  - TPU 还支持以更高吞吐量执行更低精度的矩阵乘法（例如每个 TPU v5e 芯片可以做 `4e14` int8 OPs/s）。
- **VPU**（Vector Processing Unit，向量处理单元）执行通用数学运算，比如 ReLU 激活、向量之间的逐元素加法或乘法。归约（求和）也在这里完成。[附录 A](#appendix-a-more-on-tpu-internals) 提供了更多细节。
- **VMEM**（Vector Memory，向量内存）是 TensorCore 内部的片上暂存区，靠近计算单元。它比 HBM 小得多（例如 TPU v5e 上为 128 MiB），但与 MXU 之间的带宽要高得多。VMEM 的作用类似 CPU 上的 L1/L2 缓存，但容量大得多且由程序员控制。HBM 中的数据需要先复制到 VMEM 中，TensorCore 才能基于其进行计算。

TPU 在矩阵乘法方面极其迅速。[TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture) 是迄今为止最强大的 TPU 之一，每个 core 可以做 `2.5e14` bf16 FLOPs/秒，每个芯片可以做 `5e14` bf16 FLOPs/秒。一个 8960 芯片的 pod 可以达到 4 exaflops/秒。TPU 及其脉动阵列之所以是强大的硬件加速器，是因为矩阵乘法在 $O(n^2)$ 字节上消耗 $O(n^3)$ 计算量，使得普通 ALU 很容易被计算瓶颈而非内存带宽所限制。

上图还包含 SMEM 和标量单元（scalar unit）等其他组件，它们用于控制流处理，将在 [附录 A](#appendix-a-more-on-tpu-internals) 中简要讨论，但对理解整体不是关键。另一方面，HBM 既重要又比较简单：

- **HBM**（High Bandwidth Memory，高带宽内存）是一大块快速内存，存储供 TensorCore 使用的张量。HBM 的容量通常在数十 GB 量级（例如 [TPU v5e 拥有 16GiB 的 HBM](https://cloud.google.com/tpu/docs/v5e#system_architecture)）。
  - 计算时，张量从 HBM 通过 VMEM 流入 MXU，结果再从 VMEM 写回 HBM。
  - HBM 与 TensorCore 之间（经由 VMEM）的带宽称为 "HBM 带宽"（通常约 1-2TB/秒），它限制了内存受限工作负载的计算速度。

通常，所有 TPU 操作都是流水线化并互相重叠的。要执行矩阵乘法 $X \cdot A \to Y$，TPU 首先需要将矩阵 $A$ 和 $X$ 的分块从 HBM 复制到 VMEM，然后加载到 MXU 中执行 8x128（用于 $X$）和 128x128（用于 $A$）分块的乘法，再将结果分块写回 HBM。为了高效执行，矩阵乘法是流水线化的，进出 VMEM 的复制与 MXU 的计算重叠进行。这样 MXU 可以持续工作而无需等待内存传输，从而保持矩阵乘法是计算受限而非内存受限。

下面是一个从 HBM 执行逐元素乘法的示例：

![](https://jax-ml.github.io/scaling-book/assets/img/pointwise-product.gif)

**图：** 一段动画展示在 TPU 上执行逐元素乘法的过程，字节从 HBM 加载。注意字节是分块从内存中流出，部分结果以流水线方式回写，无需等待整个数组完全实例化。

矩阵乘法看起来几乎一样，只是数据加载到 MXU 而非 VPU/向量单元，并且加载和存储的顺序不同，因为同一权重块用于多个激活块。你可以看到数据块流入 VMEM、再流入 VREG（向量寄存器）、再到向量单元、然后回到 VMEM 和 HBM。如果从 HBM 到 VMEM 的加载比向量单元（或 MXU）的 FLOPs 慢，我们就会变成"带宽受限"，因为 VPU 或 MXU 处于"喂不饱"的状态。

**核心要点：** TPU 非常简单。它们将权重从 HBM 加载到 VMEM，再从 VMEM 加载到脉动阵列，后者每秒可执行约 200 万亿次乘加运算。HBM $\leftrightarrow$ VMEM 和 VMEM $\leftrightarrow$ 脉动阵列之间的带宽决定了 TPU 能高效完成的计算的根本上限。

**VMEM 与算术强度（arithmetic intensity）：** VMEM 比 HBM 小得多，但与 MXU 之间的带宽要高得多。如同我们在 [第 1 节](../part1_roofline) 中看到的那样，这意味着如果一个算法可以将其所有输入/输出都放进 VMEM，它就不太可能撞上通信瓶颈。这在计算的算术强度较低时尤其有用：VMEM 带宽约为 HBM 带宽的 22 倍，这意味着 MXU 操作从/向 VMEM 读写时只需要 10-20 的算术强度即可达到峰值 FLOPs 利用率。也就是说，如果我们能把权重放入 VMEM 而不是 HBM，矩阵乘法可以在更小的批量大小（batch size）下变成 FLOPs 受限。这也意味着算术强度本身较低的算法仍可高效执行。问题在于 VMEM 实在很小，这通常是个挑战。我们有时会谈到 VMEM 预取（prefetching），即提前将权重加载到 VMEM 中，以便掩藏矩阵乘法的加载开销。例如在普通 Transformer 中，我们有时可以在 attention 期间将大型前馈层（feed-forward）权重加载到 VMEM 中，这样在内存带宽受限的情况下可以隐藏权重加载的开销。这要求权重足够小或被充分分片（sharded），以便单层能放入 VMEM 并仍有余量。

![](https://jax-ml.github.io/scaling-book/assets/img/tpu-bandwidth.png)

一个 TPU 芯片通常（但并非总是）由两个 TPU core 组成，它们共享内存，可以被视为一个具有两倍 FLOPs 的大型加速器（称为 "megacore" 配置）。从 TPU v4 开始一直如此。更早的 TPU 芯片各 core 拥有独立内存，被视为两个独立的加速器（TPU v3 及更早）。像 TPU v5e 这种推理优化型芯片每个芯片只有一个 TPU core。

![](https://jax-ml.github.io/scaling-book/assets/img/cores.png)

**芯片** 以 **每 4 个一组（"tray"）** 排列，通过 **PCIe 网络连接到 CPU host**。这是大多数读者熟悉的形式：4 个芯片（8 个 core，但通常视为 4 个逻辑 megacore），通过 Colab 或单个 TPU-VM 暴露给用户。对于像 TPU v5e 这样的推理芯片，每个 host 有 2 个 tray 而非 1 个，但每芯片只有 1 个 core，所以是 8 个芯片 = 8 个 core。在 Cloud TPU VM 上，每个 tray 作为单独 VM 的一部分暴露，所以再次只能看到 4 个 core。

![](https://jax-ml.github.io/scaling-book/assets/img/pcie.png)

**PCIe 带宽有限：** 与 HBM $\leftrightarrow$ VMEM 链路类似，CPU $\leftrightarrow$ HBM 之间的 PCIe 连接也有特定的带宽，它限制了 host 内存与 HBM 之间互相加载的速度。比如 TPU v4 的 PCIe 带宽为每方向 16GB/秒，比 HBM 慢近 100 倍。我们 _可以_ 把数据加载到 host（CPU）RAM 或从中卸载，但速度不快。

## TPU Networking（TPU 网络）

在一个 Pod 中，芯片通过 ICI 网络互相连接。在更早的代际（TPU v2 和 TPU v3）、推理芯片（如 TPU v5e）和 Trillium（TPU v6e）中，ICI（"inter-chip interconnects"，芯片间互联）连接最近的 4 个邻居（带边缘链路构成 2D torus）。TPU v4 和 TPU v5p 连接最近的 6 个邻居（构成 3D torus）。注意这些连接 **不** 经过 host，而是芯片之间的直接链路。

![](https://jax-ml.github.io/scaling-book/assets/img/ici-wraparound.png)

torus 拓扑将任意两个节点之间的最大距离从 $N$ 降到 $N / 2$，使通信快得多。TPU 还有一种 "twisted torus"（扭曲环面）配置，将 torus 卷成 Mobius 带样式的拓扑，进一步降低节点之间的平均距离。

TPU pod（通过 ICI 连接）可以非常大：最大 pod 尺寸（称为 **superpod**）TPU v4 是 `16x16x16`，TPU v5p 是 `16x20x28`。这些大型 pod 由可重新配置的 `4x4x4` 芯片立方体组成，立方体之间通过 [光学环绕链路](https://arxiv.org/pdf/2208.10041) 连接（光学交换机其实就是一个可重新配置的连接，带宽与 ICI 相同，可以在保留环绕链路的同时连接立方体），我们可以重新配置以连接非常大的拓扑。

![](https://jax-ml.github.io/scaling-book/assets/img/tpu-rack.png)

也可以请求更小的拓扑（如 `2x2x1`、`2x2x2`），但没有环绕链路。这是一个重要的注意点，因为通常会让大多数通信耗时翻倍。任何完整立方体的倍数（如 `4x4x4` 或 `4x4x8`）会有由光学交换机提供的环绕链路。注意 `2x2x4` 不会有任何环绕链路，因为它们由仅在完整立方体上才提供的光学交换机供给。然而 TPU v5e 8x16 _确实_ 在较长的轴上有环绕链路，因为它不使用可重新配置的光学网络。

![](https://jax-ml.github.io/scaling-book/assets/img/subslices.png)

TPU v5e 和 Trillium 的 pod 由单个 `16x16` 2D torus 构成，沿任何尺寸为 16 的轴都有环绕链路（也就是说，`8x16` 在长轴上有环绕链路）。TPU v5e 和 v6e（Trillium）不能扩展到超过 16x16 的 torus，但 pod 之间仍可以通过标准数据中心网络（DCN）通信，DCN 连接 TPU host。同样，可以请求更小的拓扑，但维度 $<16$ 的轴上没有环绕。

![](https://jax-ml.github.io/scaling-book/assets/img/more-subslices.png)

这种最近邻连接是 TPU 与 GPU 的关键差异之一。GPU 通过分层交换机连接，近似实现每个 GPU 之间的点对点连接，而不是像 TPU 那样使用本地连接。通常一个节点内的 GPU（H100 是 8 个，B200 NVL72 多达 72 个）直接连接，更大的拓扑则需要每对 GPU 之间 O(log(N)) 跳。一方面，这意味着 GPU 可以在少量跳数内发送任意数据。另一方面，TPU 大幅便宜（因为 NVLink 交换机昂贵）、连线更简单，并且因为每设备的链路数和带宽都恒定，可以扩展到更大的拓扑。详见 [这里](../part12_gpus#networking)。

ICI 相对于 DCN 非常快，但仍比 HBM 带宽慢。例如，[TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture) 拥有：

- 每芯片 `2.8e12` 字节/秒（2.8 TB/s）的 HBM 带宽。
- 每芯片每轴 `9e10` 字节/秒（90 GB/s）的 ICI 带宽，每芯片有 3 个轴。（上面那个页面列出 100 GB/s 的带宽，与此处略有不同。TPU ICI 链路根据所执行操作不同会有略微不同的带宽。一般可以放心使用本文中的数字。）
- 每 TPU `6.25e9` 字节/秒（6.25 GB/s）的 DCN（出口）带宽（通过每 host 上的 1-2 个 NIC）。TPU v6e 是 12.5e9 字节/秒，v5e 是 3.125e9 字节/秒。

这意味着当我们将模型分割到多个芯片上时，需要小心避免被较慢的跨设备通信阻塞 MXU。

**多 slice 训练（Multi-slice training）：** 一组 ICI 连接的 TPU 称为一个 **slice**。不同 slice 之间可以通过 DCN 连接，例如连接不同 pod 上的 slice。由于 DCN 比 ICI 慢得多，我们应该尽量避免计算等待来自 DCN 的数据。DCN 是 host 到 host 的，所以要通过 DCN 把缓冲区从 TPU 传输到 TPU，我们需要先经 PCIe 传到 host，然后从网络出口出去，再到目标 host 网络入口，再经 PCIe 传到 HBM。

## Key Takeaways（核心要点）

- TPU 很简单，大多数情况下可以视为一个矩阵乘法单元，连接到内存（极快）、其他芯片（通过 ICI，较快）以及数据中心其他部分（通过 DCN，一般快）。

- 通信受限于各种网络带宽，按速度顺序如下：
  - HBM 带宽：在 TensorCore 与其关联的 HBM 之间。
  - ICI 带宽：在一个 TPU 芯片与其最近的 4 或 6 个邻居之间。
  - PCIe 带宽：在 CPU host 与其关联的芯片 tray 之间。
  - DCN 带宽：在多个 CPU host 之间，通常是未通过 ICI 连接的 host。

- **在 slice 内部，TPU 仅通过 ICI 与最近的邻居相连。** 这意味着 slice 内远距离芯片之间的 ICI 通信需要先跳过中间的芯片。

- **权重矩阵在两个维度上至少需要填充到 128**（TPU v6e 上是 256），以填满 MXU（实际上小轴会被填充到 128）。

- **更低精度的矩阵乘法通常更快。** 在支持的代际上，TPU 可以以约 2 倍/4 倍于 bfloat16 的速度执行 int8 或 int4 FLOPs。VPU 操作仍以 fp32 执行。

- 为了避免阻塞 TPU 计算单元，我们需要 **确保每个通道上的通信量与其速度成正比**。

### TPU specs（TPU 规格）

下面是我们各芯片的具体数字：

| 型号 | Pod 大小 | Host 大小 | 每芯片 HBM 容量 | 每芯片 HBM 带宽 (字节/秒) | 每芯片 FLOPs/s (bf16) | 每芯片 FLOPs/s (int8) |
|-------|----------|-----------|-------------------|----------------------|---------------------|---------------------|
| TPU v3 | 32x32 | 4x2 | 32GB | 9.0e11 | 1.4e14 | 1.4e14 |
| TPU v4p | 16x16x16 | 2x2x1 | 32GB | 1.2e12 | 2.75e14 | 2.75e14 |
| TPU v5p | 16x20x28 | 2x2x1 | 96GB | 2.8e12 | 4.59e14 | 9.18e14 |
| TPU v5e | 16x16 | 4x2 | 16GB | 8.1e11 | 1.97e14 | 3.94e14 |
| TPU v6e | 16x16 | 4x2 | 32GB | 1.6e12 | 9.20e14 | 1.84e15 |

Host 大小指的是连接到单个 host 的 TPU 拓扑（例如 TPU v5e 一个 CPU host 连接到 8 个 TPU，构成 4x2 拓扑）。下面是互联数据：

| 型号 | 每链路 ICI 带宽（单向，字节/秒） | 每链路 ICI 带宽（双向，字节/秒） |
|-------|-------------------------------|---------------------------|
| **TPU v3** | 1.0e11 | 2.0e11 |
| **TPU v4p** | 4.5e10 | 9.0e10 |
| **TPU v5p** | 9.0e10 | 1.8e11 |
| **TPU v5e** | 4.5e10 | 9.0e10 |
| **TPU v6e** | 9.0e10 | 1.8e11 |

我们同时给出单向（unidirectional）带宽和双向（bidi）带宽，因为单向带宽更贴近硬件实际，但双向带宽在涉及完整环（ring）的方程中出现得更多。所谓双向（bidi）带宽指的是单条链路两个方向上可以发送的总字节数，等价地说，是单个 TPU 沿某一轴的总出口字节数（假设我们能够高效地利用两条链路）。这在我们拥有可工作的 ring 时成立，也就是当我们在某轴上有环绕连接时。这种情况发生在推理芯片完整 16 轴上，或者训练芯片（v*p）上某轴是 4 的倍数时。我们倾向于使用双向带宽，因为它在涉及双向通信的计算中频繁出现。

PCIe 带宽通常约为每 TPU `1.6e10` 字节/秒（TPU v6e 是 `3.2e10`），而 DCN 带宽通常约为每 TPU `6.25e9` 字节/秒（TPU v6e 是 `12.5e9`，TPU v5e 是 `3.125e9`）。

## Worked Problems（习题演练）

这些数字有点枯燥，但它们让你能为模型性能做基本的 roofline 估算。让我们做几道题来说明这为什么有用。第 3 部分中你会看到更多例子。

**问题 1 [LLM 延迟下界]：** 假设你想从一个 200B 参数的 bf16 模型中采样，模型分片在 32 个 TPU v4p 上。从 HBM 把所有参数加载进脉动阵列需要多少时间？*提示：使用上面的数字。*

**答案：** 我们要在 32 个芯片上加载 `sizeof(bf16) * 200e9 = 400e9` 字节，每芯片 12.5e9 字节，每芯片 HBM 带宽为 1.23e12。所以加载大约需要 10ms。

这其实很有意思，因为这是采样延迟的合理下界。每一步采样都需要从 HBM 加载所有参数，所以耗时不可能少于 10 ms。实际上，在小批量（small batch size）下，这个数字接近能达到。

**问题 2 [TPU 细节]：** 考虑一个完整的 TPU v5e pod。总共有多少个 CPU host？多少个 TPU TensorCore？整个 pod 的总 FLOPs/s 是多少？总 HBM 是多少？对 TPU v5p pod 也做一遍。

**答案：** 对于 TPU v5e，每个 pod 是 `16x16`，每个 host 是 4x2 slice，所以我们有 `16*16 / 8 = 32` 个 host。对于 TPU v5e，每个 TPU 只有一个 core，所以我们有 256 个 TensorCore。总 FLOPs/s 在 bfloat16 下是 `16*16*2e14 = 5.1e16`。每芯片 16GB HBM，所以一共 `256 * 16 = 4TB` 内存。

对于完整的 TPU v5p pod，我们有 `16x20x28` 个芯片，每个 host 是 2x2x1，所以我们有 `(16*20*28) / (2*2) = 2,240` 个 host。对于 TPU v5p，每个 TPU 有两个 TensorCore，所以我们有 `8960 * 2 = 17,920` 个 core。总 FLOPs/s 在 bfloat16 下是 `8960 * 4.5e14 = 4e18`。每芯片 96GB HBM，所以一共 `8960 * 96 = 860TB` 内存。

**问题 3 [PCIe 算术强度]：** 假设我们被迫将一个大型权重矩阵 $A$（类型 $\text{bfloat16}[D, F]$）和一批激活 $x$（类型 $\text{bfloat16}[B, D]$）存储在 host DRAM 中，并希望在它们之间执行矩阵乘法。这运行在单个 host 上，我们使用一个连接到该 host 的 TPU v6e 芯片。可以假设 $B \ll D$，且 $F = 4D$（在以后章节中我们会看到为什么这些假设合理）。要在 PCIe 上保持 FLOPs 受限，$B$ 的最小值是多少？假设 PCIe 带宽为 1.5e10 字节/秒。

**答案：** 我们要执行 $2BDF$ 次浮点运算，每芯片每秒可执行 `9.2e14` 次浮点运算。所以执行需要 $2BDF / 9.2e14$ 秒。我们要从 DRAM 加载 $2DF + 2BD$ 字节，并写回 $2BF$ 字节。我们被 PCIe 传输速率瓶颈，所以将数据传给 TPU 并传回需要 $2 \cdot (BD + DF + BF) / 1.5e10$ 秒。由于我们希望计算耗时长于权重加载（假设权重加载可以与计算完全重叠），我们要 $2BDF / 9.2e14 > 2 \cdot (BD + DF + BF) / 1.5e10$。利用 $B \ll D$ 和 $F = 4D$ 的假设可化简为

$$\frac{8BD^2}{9.2 \times 10^{14}} > \frac{8D^2}{1.5 \times 10^{10}}$$

或

$$B > \frac{9.2 \times 10^{14}}{1.5 \times 10^{10}} \simeq 61{,}000$$

**问题 4 [一般矩阵乘法延迟]：** 假设我们要将一个 int8[16384, 4096] 的权重矩阵乘以一个尺寸为 int8[B, 4096] 的激活矩阵，B 是某个未知 batch size。先在 1 个 TPU v5e 上进行。

1. 这次乘法作为 B 的函数耗时多久？*提示：可能有助于先计算从 HBM 加载数组所需时间，以及实际乘法所需时间。哪个是瓶颈？*
2. 如果想在 VMEM 中执行此操作呢？作为 B 的函数耗时多久？

**答案：** (1) 我们需要执行的浮点运算次数是 $2 \cdot 4096 \cdot 16384 \cdot B = 1.3 \times 10^{8} \cdot B$。所以 $T_{\text{math}} = (1.3 \times 10^{8} \cdot B) / 3.94 \times 10^{14}$ 秒。我们需要从 HBM 加载 $16384 \cdot 4096 + 4096 \cdot B$ 字节到 VMEM，再从 VMEM 写回 $16384 \cdot B$ 字节到 HBM。这意味着 $T_{\text{comms}} = (6.7 \times 10^{7} + 2 \times 10^{4} \cdot B) / 8.1 \times 10^{11}$ 秒。假设通信和计算尽可能重叠，整个乘法大约耗时

$$\max\{T_{\text{math}}, T_{\text{comms}}\} = \max\left\{ \frac{6.7 \times 10^{7} + 2 \times 10^{4} \cdot B}{8.1 \times 10^{11}}, \frac{1.3 \times 10^{8} \cdot B}{3.94 \times 10^{14}} \right\}$$

当 $\frac{6.7 \times 10^{7} + 2 \times 10^{4} \cdot B}{8.1 \times 10^{11}} < \frac{1.3 \times 10^{8} \cdot B}{3.94 \times 10^{14}}$ 时我们是 FLOPs 受限的，等价于 $B > 271$。这比我们在 [第 1 节](../part1_roofline) 推导出的 240 略大，因为我们把 $D$ 和 $F$ 的完整影响纳入了考虑。

(2) 如果改为从 VMEM 加载，假设 VMEM 到 MXU 的带宽是 HBM $\leftrightarrow$ VMEM 带宽的 22 倍。这把数据加载分母从 8.1e11 提升到 1.78e13，得到 $B > 11$。注意实际中我们不能把 VMEM 全部带宽都用于加载 $W$，所以实际中接近 20。

**问题 5 [ICI 带宽]：** 假设我们有一个 TPU v5e `4x4` slice。我们想把一个类型为 `bfloat16[8, 128, 8192]` 的数组从 `TPU{0,0}` 发送到 `TPU{3, 3}`。假设 TPU v5e 每跳延迟为 $1\mu s$。

1. 第一个字节多快到达目的地？
2. 整个传输需要多长时间？

**答案：** TPU v5e 是 2D 连接。因为我们只有 `4x4` slice（没有尺寸为 16 的轴），所以没有环绕连接。因此目标芯片只有两个端口可以接收数据，源芯片同样只有两个端口可以发送。需要传输的数据量是 `2 * 8 * 128 * 8192 = 1.7e7` 字节。我们可以同时使用两个端口（即一半向右一半向下发送），所以每秒传输 `2 * 4.5e10 = 9e10` 字节，意味着传输整个数组大约需要 `1.7e7 / 9e10 = 188us`（假设是带宽受限）。在 `4x4` slice 中芯片 $(0, 0)$ 与 $(3, 3)$ 之间有六跳，因为少于 16 个芯片的轴没有环绕链路。每跳延迟约 $1\mu s$，所以第一字节大约 `6us` 到达，整个传输耗时 `188us`。

**问题 6 [综合应用，难]：** 假设你有一个大矩阵 **A**：`int8[128 * 1024, 128 * 1024]` 在 TPU v5e 4x4 slice 上均匀分片，但在每个芯片上被卸载到 host DRAM。假设你想将整个数组复制到 TPU{0, 0} 并与一个向量 `bf16[8, 128 * 1024]` 相乘。这需要多长时间？*提示：使用上面的数字。*

**答案：** 让我们先列出需要执行的操作。我们的数组大约 16GB。从上表看，TPU v5e host 是 4x2 拓扑，所以 4x4 有 2 个 host。因此，由于数组均匀分片，每个 host 实际上包含数组的 1/2（即 8GB）。我们需要将这些块全部复制到 TPU{0,0}，有两个选项：

1. 我们可以通过 DCN 复制，然后通过 PCIe 把整个未分片的数组加载到 HBM。
2. 我们可以将分片数组加载到对应的 TPU 上，然后通过 ICI 进行 gather，再在 TPU{0,0} 上执行矩阵乘法。

显然选项 (2) 更好。DCN 比 ICI 慢，并且我们更愿意通过多个 PCIe 链路加载大数组，而不是只用少数几个（host 0 上的 8 个）。下图展示了系统的一部分。如上所述，TPU 之间通过 ICI 连接（即使跨 host 也是如此），所有 TPU 都连接到其 host CPU（通过 PCIe），host 之间通过 DCN 连接。

![](https://jax-ml.github.io/scaling-book/assets/img/challenge-problem.png)

每个芯片实际上都有自己的 PCIe 链路连接到 host，但为清晰起见图中只展示了一条。

现在我们逐步分析每部分耗时：

1. **PCIe 加载**：我们通过 16 个 PCIe 链路加载 16GB 的块，每个链路带宽为 `1.5e10` 字节/秒。所以这大约需要 66ms。

2. **ICI 复制：** 现在每个 TPU 有 16GB / 16 = 1GB 的数组。我们的 ICI 带宽是每链路 9e10 字节/秒 *双向*，从上图可以看到 TPU{0,0} 在此拓扑下 TPU v5e 的 4 个 ICI 链路只有 2 个被用到。由于 TPU{0,0} 需要沿 2 个轴接收共 15GB 数据（每轴每链路 `4.5e10` 字节/秒），我们可以将时间下界估为 `15e9 / (4.5e10 * 2) = 167ms`。实际中可能达不到，因为负载非常不均，但应该不会差超过 2 倍。如第 3 节所述，执行完整的 AllGather 大约也是 `16e9 / (4.5e10 * 2)`，所以这接近最优。

3. **HBM $\rightarrow$ MXU 加载：** 要执行最后的矩阵乘法，需要将这 16e9 字节加上 bf16[8, 128 * 1024] 数组（再 2MB，可忽略）通过 HBM 带宽加载到 MXU，需要 `16e9 / 8.1e11 = 19ms`。

4. **FLOPs：** 我们一共执行 $2 \cdot 8 \cdot 128 \cdot 1024 \cdot 128 \cdot 1024 = 2.7 \times 10^{11}$ FLOPs，因为我们能做 `1.97e14` bf16 FLOPs/s，所以耗时 1.3ms。

总耗时的上界是这些时间之和，但因为 TPU 通常可以重叠这些操作，可以把它视为一个流水线问题，受制于最慢的部分。如果是这样，那么答案至少是 167ms，由于重叠不完美，可能更接近 200ms。

**第 2 部分到此结束！第 3 部分讲分区（partitioning）和跨 TPU 通信，[点击此处](../part3_sharding) 阅读。**

## Appendix（附录）

### Appendix A: More on TPU internals（附录 A：TPU 内部机制详解）

这里我们更深入地讨论 TPU 的内部操作。除非另有说明，我们提供 TPU v5p 的规格。

### VPU

VPU 是 TPU 的向量算术核心。VPU 包含一个二维 SIMD 向量机（即 **VPU**），执行 vadd（向量加）或 vmax（逐元素最大）等逐元素算术运算，以及一组称为 **VREGs** 的向量寄存器，为 VPU 和 MXU 保存数据。

**VREGs：** 每个 TPU v5p core 有 64 个 32 位 VREG（TPU v4 是 32 个），每个 core 总共约 `64 * 8 * 128 * 4 = 256kB` 的 VREG 内存（整个芯片是这个的 2 倍，因为有两个 core）。TPU v5p 每周期可以从 VMEM 加载 3 个寄存器、向 VMEM 写入 1 个寄存器。

**VPU：** VPU 是形状为 `(8, 128)` 的 2D 向量算术单元，128 维度称为 lane 轴，8 维度称为 sublane 轴。在 v5 上每个 (lane, sublane) 对包含 4 个标准浮点 ALU，它们彼此独立。VPU 在每个 ALU 上以单周期执行大多数算术指令（如 vadd 即向量加），延迟为 2 周期，所以例如在 v5 上你可以每周期把 4 对 f32 值从 VREG 中相加。一条典型的 VPU 指令可能像 `{v2 = vadd.8x128.f32 v0, v1}`，其中 v0 和 v1 是输入 VREG，v2 是输出 VREG。

所有 lane 和 sublane 在每个周期执行相同的程序，呈纯 SIMD 方式，但每个 ALU 可以执行不同的运算。所以我们可以在单周期内例如做 1 个 vadd 和 1 个 vsub，每个都对两个完整 VREG 操作并把结果写到第三个。

**小测验 [计算 VPU 吞吐]：** 利用上述信息，计算 TPU v5p 每秒可执行多少向量 FLOPs。TPU v5p 时钟频率约 1.75GHz。

*答案*：每周期每 core 可以在 `8 * 128` 个 ALU 上执行 4 条向量指令。整个芯片每周期有 `8 * 128 * 4 * 2` FLOPs，即 `8 * 128 * 4 * 2 * 1.75e9 = 1.4e13 FLOPs/s`。注意这比 MXU 的约 `2e14` FLOPs/s 小了多少（约 10 倍）。

**归约（Reductions）：** 通常，跨 sublane 维度的通信或归约比跨 lane 维度更容易。例如 VPU 支持一种 lane 内 shuffle 操作，可在大约一个周期内沿尺寸 8 的轴滚动。这可用于对 sublane 维度执行高效归约（只需要 shuffle 4、2 和 1，并做 3 对逐元素求和）。

跨 lane 归约困难得多，需要一个独立的硬件单元，称为 XLU 或 "cross lane unit"（跨 lane 单元），它较慢且代价较高。

**与 GPU 的对比：** 对熟悉 NVIDIA GPU 的人来说，VPU 中的每个 ALU 类似于一个 CUDA core，单个 VPU lane 类似于一个 "Warp Scheduler"，即通常 32 个执行 SIMD 算术的 CUDA core 集合。lane 内归约相当容易，但若需要跨 lane，至少要经过 VMEM/XLU/SMEM，慢得多。详见 [GPU 章节](../part12_gpus)。

### Scalar Core（标量核心）

标量核心是 TPU 的控制单元。它取指并分发所有指令，执行从 HBM 到 VMEM 的传输，并可以编程做标量元数据工作。由于标量核心是单线程的，一个副作用是 TPU 的每个 core 每周期只能创建一个 DMA 请求。

举例来说，单个标量核心控制一个 VPU（包含 4096 个 ALU）、4 个 MXU、2 个 XLU 和多个 DMA 引擎。这种每单位计算控制权高度倾斜的特性是硬件效率的一个来源，但也限制了以任何有意思的方式做数据相关向量化的能力。

### Appendix B: How does a systolic array work?（附录 B：脉动阵列如何工作？）

TPU MXU 的核心是一个 `128x128` 的脉动阵列（systolic array）（TPU v6e 是 `256x256`）。完全饱和时，脉动阵列可以每 8 个时钟周期执行一次 `bfloat16[8,128] @ bf16[128,128] -> f32[8,128]`（将一个 bfloat16 元素的 `8x128` 矩阵乘以一个 bfloat16 元素的 `128x128` 矩阵，并将结果存入 float32 元素的 `8x128` 矩阵）。

- 脉动阵列的核心是 2D `128x128`（`=16,384`）的 ALU 网格，每个 ALU 都能执行乘加运算。
- 权重（**W**，`128x128` 输入）从上方传入（称为 RHS），输入（**X**，`8x128` 输入）从左侧传入（称为 LHS）。

下面是一段简化动画，展示一组权重（蓝色）与一组激活（绿色）相乘。你会注意到权重（RHS）首先沿对角线部分加载，然后激活也沿对角线送入。在下面每一帧中，我们将所有重叠的绿色和蓝色单元相乘，与从上方传入的残差求和，然后再将结果向下传一格。

![](https://jax-ml.github.io/scaling-book/assets/img/systolic-array.gif)

下面是这段动画的更通用版本，展示输出从计算中流出：

![](https://jax-ml.github.io/scaling-book/assets/img/systolic-array2.gif)

下图展示如何在多个 RHS 和 LHS 数组上做流水线：

![](https://jax-ml.github.io/scaling-book/assets/img/systolic-array-pipelining.png)

在权重（RHS）和激活（LHS）加载时存在初始的流水线气泡（pipeline bubble）。在初始气泡之后，新的输入和权重可以无额外气泡地继续加载。

下面是一段不太好的 bf16[2, 3] x bf16[3, 3] 矩阵乘法动画，可以想象成一个 2x3 权重矩阵与 batch 1、size 3 的输入激活相乘。它相对前面的幻灯片做了旋转，输入向右流出而非向下流出，但你大致可以看到结构。

![](https://jax-ml.github.io/scaling-book/assets/img/systolic-array-bad.gif)

我们可以高效地流水线化以做大矩阵乘法而无需太大的流水线气泡。话虽如此，重要的是矩阵的形状要大于 MXU 的边长，通常是 128x128。一些 TPU（自 TPU v3 起）有多个 MXU，TPU v3 是 2 个，TPU v4/5 是 4 个，所以我们要确保 tiling 维度大于 128 * MXU 数量。[这里](https://www.youtube.com/watch?v=sJltBQ4MOHA) 是一段不错的动画。

Trillium（TPU v6e）有 `256x256` 脉动阵列，意味着每周期可执行 4 倍多的 FLOPs。这也意味着张量的维度需要再大一倍才能完全利用 MXU。

[这篇博客](https://fleetwood.dev/posts/domain-specific-architectures#google-tpu) 有另一段很棒的脉动阵列乘法动画，针对固定权重矩阵。

### Miscellaneous（杂项）

*工作完成于 Google DeepMind，作者现就职于 MatX。

### Citation（引用）

学术语境下的引用，请将本作品引用为：

```
Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.
```

或 BibTeX 条目：

```
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
