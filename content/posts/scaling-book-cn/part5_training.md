---
title: "Training（训练）"
date: 2026-04-29
draft: false
math: true
weight: 6
---

{{< katex >}}

# 如何为训练并行化 Transformer（How to Parallelize a Transformer for Training）

《How To Scale Your Model》第 5 部分（[第 4 部分：Transformers](../part4_transformers) | [第 6 部分：Training LLaMA](../part6_applied_training)）

本节我们讨论 LLM 训练中四种主要的并行方案：数据并行（data parallelism）、完全分片数据并行（fully-sharded data parallelism, FSDP）、张量并行（tensor parallelism）和流水线并行（pipeline parallelism）。对每一种方案，我们都会计算在何时会受到通信瓶颈的限制。

**目录**

[什么是 Scaling？](#什么是-scaling)

- [数据并行（Data Parallelism）](#数据并行data-parallelism)
- [完全分片数据并行（FSDP）](#完全分片数据并行fsdp)
- [张量并行（Tensor Parallelism）](#张量并行tensor-parallelism)
- [组合 FSDP 和张量并行](#组合-fsdp-和张量并行)
- [流水线（Pipelining）](#流水线pipelining)
- [跨 Pod 扩展](#跨-pod-扩展)

[在 TPU 上训练 LLM 的要点](#在-tpu-上训练-llm-的要点)

[一些练习题](#一些练习题)

[附录](#附录)

- [附录 A：推导反向传播通信](#附录-a推导反向传播通信)

## 什么是 Scaling？

"模型 scaling"的目标是：在增加用于训练或推理的芯片数量时，能够获得成比例的、线性的吞吐量提升（我们称之为 _强扩展（strong scaling）_）。单芯片性能取决于内存带宽和 FLOPs 之间的权衡，而集群级别的性能则取决于能否通过将芯片间通信与有效 FLOPs 重叠来隐藏通信开销。这并非易事，因为增加芯片数量会增加通信负载，同时减少每个设备上可用于隐藏通信的计算量。正如我们在[第 3 节](../part3_sharding)所看到的，分片矩阵乘法通常需要昂贵的 AllGather 或 ReduceScatter，这些操作会阻塞 TPU 执行有用的工作。本节的目标就是找出这些操作在何时变得 _过于昂贵_。

本节将讨论四种常见的并行方案：（纯）**数据并行**、**完全分片数据并行**（FSDP / ZeRO 分片）、**张量并行**（也称为模型并行），以及（简要介绍）**流水线并行**。对于每一种方案，我们将展示其通信成本，以及该成本在何时开始成为计算成本的瓶颈。我们将关注通信瓶颈——因为虽然内存容量约束很重要，但当使用重计算（rematerialization，即 activation checkpointing）以及预训练时的大量芯片时，它通常不会成为瓶颈。我们也不讨论 MoE 中的专家并行（expert parallelism）——它会大幅扩展设计空间，本节只关注稠密 Transformer 的基础情况。在本节中，你可以只关注芯片间的通信成本，因为只要单芯片批量大小（batch size）足够大，从 HBM 到 MXU 的数据传输就已经能够与计算重叠。

我们将使用以下符号简化本节中的计算。

| 符号 | 含义（模型参数） |
|----------|---------------------------|
| D | **d**model（隐藏维度/残差流维度） |
| F | **d**ff（前馈维度） |
| B | 批量维度（批中的 token 总数；总数，非每设备数） |
| T | 序列长度 |
| L | 模型层数 |

| 符号 | 含义（硬件特性） |
|----------|----------------------------------|
| C | 每芯片的 FLOPS/s |
| W | 网络带宽（双向，常带下标如 $W_{\text{ici}}$ 或 $W_{\text{dcn}}$） |
| X | 沿 mesh 轴 X 的芯片数 |
| Y | 沿另一 mesh 轴 Y 的芯片数 |
| Z | 沿第三个 mesh 轴 Z 的芯片数 |

为简化起见，**我们将 Transformer 近似为 MLP 块的堆叠**——正如我们在[第 4 节](../part4_transformers)所看到的，对于较大的模型，注意力（attention）所占的 FLOPs 比例相对较小。我们也会忽略门控（gating）矩阵乘法，从而每一层就有如下简单结构：

![一个简化的 Transformer 层](https://jax-ml.github.io/scaling-book/assets/img/transformer-layer.png)

**图：** 一个简化的 Transformer 层。我们将每个 FFW 块视为两个矩阵的堆叠：**Win**: `bf16[D, F]`（上投影）和 **Wout**: `bf16[F, D]`（下投影），输入为 **In**: `bf16[B, D]`。

下面是我们这个无并行小型 Transformer 的完整算法。

**前向传播：** 需要计算 Loss[B]

1. Tmp[B, F] = In[B, D] \*D Win[D, F]
2. Out[B, D] = Tmp[B, F] \*F Wout[F, D]
3. Loss[B] = …

**反向传播：** 需要计算 dWout[F, D]、dWin[D, F]

1. dOut[B, D] = …
2. dWout[F, D] = Tmp[B, F] \*B dOut[B, D]
3. dTmp[B, F] = dOut[B, D] \*D Wout[F, D]
4. dWin[D, F] = In[B, D] \*B dTmp[B, F]
5. dIn[B, D] = dTmp[B, F] \*F Win[D, F]（*前面层需要*）

我们提供这个算法以便与加入通信的算法进行对比。

下面是我们将讨论的 4 种并行方案。每种方案都可以由上图中 **In**、**Win**、**Wout** 和 **Out** 的分片方式唯一定义。

**1. 数据并行：** *激活按 batch 分片，参数和优化器状态在每个设备上复制。通信只在反向传播时发生。*

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

**2. 完全分片数据并行（FSDP 或 ZeRO-3）：** *激活按 batch 分片（与纯数据并行一样），参数沿同一 mesh 轴分片，并在前向传播中即用即 AllGather。优化器状态也按 batch 分片。减少了重复的内存。*

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

**3. 张量并行（也称为 Megatron 分片或模型并行）：** *激活沿 D（$d_\text{model}$）分片，参数沿 F（$d_{ff}$）分片。在每个块前后 AllGather 和 ReduceScatter 激活。与 FSDP 兼容。*

$$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$

**4. 流水线并行：** *权重沿层（layer）维度分片，激活进行微批（microbatch）化并沿层维度滚动。流水线阶段间通信极少（仅在单跳间移动激活）。借用一下符号：*

$$\text{In}[L_Z, B, D][i] \cdot_D W_\text{in}[L_Z, D, F][i] \cdot_F W_\text{out}[L_Z, F, D][i] \rightarrow \text{Out}[L_Z, B, D][i]$$

### 数据并行（Data Parallelism）

**语法：** $\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$

当你的模型即使以很小的 batch size（>240 tokens，以保持计算受限）也能放入单芯片时，**你就总应该使用简单的数据并行**。纯数据并行将我们的激活划分到任意数量的 TPU 上，只要 TPU 数量小于 batch size 即可。前向传播不涉及通信，但每个步骤结束时，**每个 TPU 都要对其本地梯度执行 AllReduce 来同步它们，然后再更新参数**。

![纯数据并行示意图](https://jax-ml.github.io/scaling-book/assets/img/data-parallelism.png)

**图：** 纯数据并行示意图（前向传播）。我们的激活（左侧）沿 batch 维度完全分片，权重则完全复制，所以每个 TPU 都有一份相同的权重副本。这意味着权重的总内存增加了 N 倍，但前向传播不需要通信。

下面是前向和反向传播的完整算法。为简洁起见，我们用 dOut 表示 dL/dOut（这是符号上的滥用）。

**纯数据并行算法：**

**前向传播：** 需要计算 Loss[BX]

1. Tmp[BX, F] = In[BX, D] \*D Win[D, F]
2. Out[BX, D] = Tmp[BX, F] \*F Wout[F, D]
3. Loss[BX] = …

**反向传播：** 需要计算 dWout[F, D]、dWin[D, F]

1. dOut[BX, D] = …
2. dWout[F, D] {UX} = Tmp[BX, F] \*B dOut[BX, D]
3. dWout[F, D] = **AllReduce**(dWout[F, D] {UX})（*不在关键路径上，可以异步执行*）
4. dTmp[BX, F] = dOut[BX, D] \*D Wout[F, D]
5. dWin[D, F] {UX} = In[BX, D] \*B dTmp[BX, F]
6. dWin[D, F] = **AllReduce**(dWin[D, F] {UX})（*不在关键路径上，可以异步执行*）
7. dIn[BX, D] = dTmp[BX, F] \*F Win[D, F]（*前面层需要*）

我们忽略了损失函数的细节，并将 $\text{Tmp} = W_\text{in} \cdot \text{In}$ 简写。注意虽然最终 loss 是 **AllReduce**(Loss[BX]) 的平均值，但我们只在反向传播时需要计算 AllReduce，用于平均权重梯度。

注意前向传播没有通信——**通信全部发生在反向传播**！反向传播还有一个很棒的性质：AllReduce 不在"关键路径"上，意味着每次 AllReduce 可以在方便的时候执行，不会阻塞后续操作。如果总通信成本超过总计算成本，**仍然可能成为瓶颈**，但从实现角度看要宽容得多。我们会看到模型/张量并行就没有这个性质。

**为什么这样做？** 纯数据并行通过沿 batch 维度划分激活来减轻激活的内存压力，让我们几乎可以任意增加 batch size，只要有足够多的芯片来划分 batch 维度。尤其在训练期间激活常常主导内存使用时，这非常有帮助。

**为什么不这样做？** 纯数据并行无法减轻模型参数和优化器状态的内存压力，这意味着对于参数 + 优化器状态无法放入单个 TPU 的大规模有趣模型，纯数据并行很少有用。给个直观感受：如果我们用 bf16 训练参数、用 fp32 加 Adam 训练优化器状态（Adam 存储参数、一阶和二阶累加器。由于参数是 bfloat16 而优化器状态是 float32，每个参数共需 `2 + 8 = 10` 字节），那么我们能容纳的最大模型有 $\text{TPU memory} / 10$ 个参数，所以例如在有 96GB HBM 的 TPUv5p 芯片上用纯数据并行，约为 9B 参数。

**要点：** 用 Adam 和纯数据并行能训练的最大模型是 $\text{num\_params} = \text{HBM per device} / 10$。在 TPU v5p 上大约 9B 参数。注意这没有计入梯度检查点（gradient checkpoints），所以这其实并不实用。这是 batch 为 1 个 token 时的绝对下限。

*要让它对实际训练中的真实模型有用，我们至少需要部分分片模型参数或优化器。*

**何时被通信瓶颈限制？** 如上所示，每层有两个 AllReduce，每个大小为 $2DF$（bf16 权重）。何时数据并行会让我们受通信限制？

如上表所示，设 $C$ 为每芯片 FLOPs，$W_{\text{ici}}$ 为**双向**网络带宽，$X$ 为 batch 划分到的分片数（我们假设此划分在 ICI mesh 上完成，所以相关网络带宽是 $W_\text{ici}$）。让我们计算执行相关矩阵乘法所需的时间 $T_\text{math}$ 和所需的通信时间 $T_\text{comms}$。由于此并行方案前向传播不需要通信，我们只需为反向传播计算这些量。

*通信时间：* 由前一节我们知道，1D mesh 中执行 AllReduce 所需的时间只取决于被 AllReduce 数组的总字节数和 ICI 带宽 $W_\text{ici}$；具体而言 AllReduce 时间是 $2 \cdot \text{total bytes} / W_\text{ici}$。由于我们需要为 $W_\text{in}$ 和 $W_\text{out}$ 都执行 AllReduce，每层有 2 次 AllReduce。每个 AllReduce 针对一个权重矩阵，即 $DF$ 个参数的数组，或 $2DF$ 字节。综上，单层 AllReduce 的总时间为

$$\begin{align}
T_\text{comms} &= \frac{2 \cdot 2 \cdot 2 \cdot D \cdot F}{W_\text{ici}}. \\
\end{align}$$

*矩阵乘法时间：* 每层包含前向 2 次矩阵乘法，反向 4 次矩阵乘法，每次需要 $2(B/X)DF$ FLOPs。所以单层反向传播有

$$\begin{align}
T_\text{math} &= \frac{2 \cdot 2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
\end{align}$$

由于我们重叠它们，每层的总时间是这两个量的最大值：

$$\begin{aligned}
T &\approx \max(\frac{8 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{8 \cdot D \cdot F}{W_\text{ici}}) \\
T &\approx 8 \cdot D \cdot F \cdot \max(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}})
\end{aligned}$$

当 $T_\text{math}/T_\text{comms} > 1$ 时我们计算受限，即

$$\begin{align}
\frac{B}{X} > \frac{C}{W_\text{ici}}.
\end{align}$$

结论是，要在数据并行下保持计算受限，我们需要每设备 batch size $B / X$ 超过 ICI 操作强度（operational intensity）$C / W_\text{ici}$。这归根结底是因为计算时间随每设备 batch size 缩放，而通信时间与该量无关（因为我们传输的是模型权重）。注意 $B/X > C/W_\text{ici}$ 条件与单设备计算受限规则 $B > 240$ 的相似性；在那种情况下规则也来自计算时间随 batch size 缩放，而数据传输大小（在 $B \ll F, D$ 范围内）与 batch size 无关。

让我们带入一些真实数字来感受规模。对于 TPUv5p，`C=4.6e14`、`W=2 * 9e10`（1D ICI 数据并行），所以**每芯片 batch size 必须至少为 2,550 才能避免受通信限制**。由于我们可以沿多个轴做数据并行，如果将 TPUv5p pod 的全部三个轴都用于纯数据并行，我们会将带宽 $W_\text{ici}$ 提高 3 倍，可以将每 TPU 的 BS 缩小到 850，即每个 pod（8960 芯片）每批 7.6M tokens！**这告诉我们纯数据并行很难成为瓶颈！**

**注意 [上下文并行]：** 本节中 $B$ 始终指**以 token 计**的总 batch size。当然，我们的 batch 由许多不同序列组成，那么这是怎么工作的呢？对 MLP 而言，**token 就是 token**！它们属于同一个序列还是两个不同序列都无所谓。所以我们大致上可以自由地在 batch 和 sequence 维度上做数据并行：我们称之为上下文并行（context parallelism）或序列并行（sequence parallelism），但你可以把它看作另一种数据并行。注意力比 MLP 更棘手，因为我们做了一些跨序列计算，但可以通过在 attention 期间收集 KV 或 Q 并仔细重叠 FLOPs 与通信来处理（通常使用所谓的 "ring attention"）。本节中我们将完全忽略序列维度，假设有某种形式的 batch 或 sequence 并行。

**关于多 mesh 轴的注意事项：** 我们应该简要说明多个轴如何影响可用带宽。当我们在给定并行策略上使用多个 mesh 轴时，会获得更多带宽。

- **定义：** $M_X$（$M_Y$、$M_Z$ 等）是给定并行策略所跨越的硬件 mesh 轴数量。
- **效果（带宽受限）：** 使用 $M$ 个轴提供（约 $M$ 倍）的总链路带宽，因此集合操作时间按 $\propto 1/M_X$ 缩放。

### 完全分片数据并行（FSDP）

**语法：** $\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$

完全分片数据并行（常称 FSDP 或 ZeRO 分片）将模型优化器状态和权重划分到数据并行分片上，并按需高效地 gather 和 scatter 它们。**与纯数据并行相比，FSDP 大幅减少了每设备的内存使用并节省了反向传播 FLOPs，开销极小。**

![FSDP 示意图](https://jax-ml.github.io/scaling-book/assets/img/fsdp.png)

**图：** FSDP 将 Win 的收缩维度和 Wout 的输出维度沿数据维度分片。这减少了内存，但（来自第 3 节）要求我们在做矩阵乘法之前先 gather W 的权重。注意激活（左侧）*没有沿收缩维度分片*，这就是为什么我们必须 gather。**注意我们的权重优化器状态同样沿收缩维度分片。**

你可能记得（来自[第 3 节](../part3_sharding)），AllReduce 可以分解为 AllGather 和 ReduceScatter。这意味着，相比标准数据并行的完整梯度 AllReduce，我们可以将权重和优化器状态划分到芯片上，在前向传播每层时 AllGather 它们，并在反向传播时 ReduceScatter 权重，无需额外开销。

下面是 FSDP 的完整算法。

**完全分片数据并行（FSDP）：**

**前向传播：** 需要计算 Loss[BX]

1. Win[D, F] = **AllGather**(Win[DX, F])（*不在关键路径上，可以在前一层时完成*）
2. Tmp[BX, F] = In[BX, D] \*D Win[D, F]（*现在可以丢弃 Win[D, F]*）
3. Wout[F, D] = **AllGather**(Wout[F, DX])（*不在关键路径上，可以在前一层时完成*）
4. Out[BX, D] = Tmp[BX, F] \*F Wout[F, D]
5. Loss[BX] = …

**反向传播：** 需要计算 dWout[F, DX]、dWin[DX, F]

1. dOut[BX, D] = …
2. dWout[F, D] {UX} = Tmp[BX, F] \*B dOut[BX, D]
3. dWout[F, DX] = **ReduceScatter**(dWout[F, D] {UX})（*不在关键路径上，可异步执行*）
4. Wout[F, D] = **AllGather**(Wout[F, DX])（*可以提前完成*）
5. dTmp[BX, F] = dOut[BX, D] \*D Wout[F, D] *（这里可以丢弃 Wout[F, D]）*
6. dWin[D,F] {UX} = dTmp[BX, F] \*B In[BX, D]
7. dWin[DX, F] = **ReduceScatter**(dWin[D, F] {UX}) *（不在关键路径上，可异步执行）*
8. Win[D, F] = **AllGather**(Win[DX, F])（*可以提前完成*）
9. dIn[BX, D] = dTmp[BX, F] \*F Win[D, F]（*前面层需要）（这里可以丢弃 Win[D, F]*）

这也叫 "ZeRO 分片"，来自 "Zero Redundancy Optimizer"（零冗余优化器），因为我们不执行任何不必要的计算或存储任何不必要的状态。ZeRO-{1,2,3} 分别用于指代以这种方式分片优化器状态、梯度和权重。由于它们都有相同的通信成本（严格来说，FSDP 在前向传播中增加了纯 DP 没有的通信，但比例与反向传播相同，因此对通信 roofline 没有影响。关键在于 ZeRO-3 将反向传播 AllReduce 转换为 AllGather 和 ReduceScatter，它们具有相同的总通信量），我们基本上总是可以做 ZeRO-3 分片，将参数、梯度和优化器状态分片到一组设备上。

**为什么要这样做？** 标准数据并行涉及大量重复工作。每个 TPU 都对完整梯度做 AllReduce，然后更新完整优化器状态（所有 TPU 上做相同工作），然后更新参数（再次完全重复）。对 ZeRO 分片（分片梯度/优化器状态）而言，你可以 ReduceScatter 梯度，只更新自己那一片优化器状态，更新一片参数，然后在前向传播需要时 AllGather 参数。

**何时被通信瓶颈限制？** 我们的相对 FLOPs 和通信成本与纯数据并行完全相同，因为反向传播的每个 AllReduce 已变成 AllGather + ReduceScatter。回想一下 AllReduce 由 AllGather 和 ReduceScatter 实现，每个成本是其一半。这里我们对前向传播建模，因为它的 FLOPs 与通信比与反向传播相同：

$$\begin{aligned}
T_\text{math} &= \frac{2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
T_\text{comms} &= \frac{2 \cdot 2 \cdot D \cdot F}{W_\text{ici}} \\
T &\approx \max\left(\frac{4 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{4 \cdot D \cdot F}{W_\text{ici}}\right) \\
T &\approx 4 \cdot D \cdot F \cdot \max\left(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}}\right)
\end{aligned}$$

因此，与纯数据并行一样，我们在 $B / X > C / W_\text{ici}$ 时计算受限，即每设备 batch size $B/X$ 超过 "ICI 操作强度" $C/W_\text{ici}$（对 v5p 是 `4.59e14 / 1.8e11 = 2550`）时。这对我们很好，因为这意味着如果我们的每设备 batch size 大到足以让纯数据并行计算受限，我们就可以——不必担心离开计算受限范围——直接升级到 FSDP，节省大量参数和优化器状态内存！虽然我们确实在前向传播中增加了通信，但这个成本无关紧要，因为它只是与前向传播 FLOPs 重叠。

**要点：** 在 TPUv5 上，FSDP 和纯数据并行在每设备 batch size 小于 $2550 / M_X$（其中 $M_X$ 是 mesh 轴数）时变得带宽受限。

例如，DeepSeek-V2（少数几个最近发布训练 batch size 信息的强模型之一）使用了约 40M tokens 的 batch size。**这让我们可以扩展到约 47,000 个芯片，或大约 5 个 TPUv5 pod，然后才会触及带宽限制。**

对于 LLaMA-3 70B，它训练了约 `6.3e24 (15e12 * 70e9 * 6)` FLOPs，我们可以将 16M tokens 的 batch 划分到约 `16e6 / (2550 / 3) = 18,823` 个芯片（约 2 个 8960 芯片的 pod）上，每个芯片以 `4.59e14` FLOPs、50% 的峰值 FLOPs 利用率（常称 MFU）运行，**约 17 天就能训完**。不错！但让我们看看如何做得更好。

**关于临界 batch size 的注意：** 有点反直觉的是，我们在总 batch size 减少（芯片数固定）时变得更受通信瓶颈限制。数据并行和 FSDP 让我们可以扩展到任意多的芯片，只要我们能继续增加 batch size！然而实际上，随着 batch size 增加，我们倾向于看到训练收益递减，因为我们的梯度变得几乎无噪声。我们有时也会看到训练不稳定。因此，在"无限计算范围"中寻找最佳分片方案的游戏，通常从一个由 scaling laws 决定的固定 batch size 和已知（大量）芯片数开始，然后旨在找到一个能将这个小 batch size 装入这么多芯片上的划分方案。

### 张量并行（Tensor Parallelism）

**语法：** $\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$（我们用 $Y$ 是为了之后与 FSDP 组合）

在完全分片数据并行 AllReduce 中我们跨芯片移动权重。我们也可以分片模型的前馈维度并在层中移动激活——这称为"1D 模型并行"或 Megatron 分片。这可以解锁每 pod 更小的有效 batch size。下图展示了一个以这种方式分片的单矩阵示例：

![基本张量并行示例](https://jax-ml.github.io/scaling-book/assets/img/model-parallelism.png)

**图：** 基本张量并行示例。由于我们只在 Y 上分片激活（不像 FSDP 中我们在 X 上分片），所以在 X 上复制激活。用我们的标准语法：**A**[B, DY] \* **B**[D, FY] -> **C**[B, FY]。因为我们只在一个收缩维度上分片，通常需要在矩阵乘法之前 AllGather 激活 **A**。

如所述，**In[B, DY] \*D Win[D, FY] \*F Wout[FY, D] -> Out[B, DY] 意味着我们必须在第一个矩阵乘法之前 gather 激活。当激活小于权重时，这比 ZeRO 分片更便宜。** 这通常只在加入一定 ZeRO 分片（这会减小 gather 的大小）时才成立。这是我们倾向于混合 ZeRO 分片和张量并行的原因之一。

下面是张量并行的算法！

**张量并行：**

**前向传播：** 需要计算 Loss[B]

1. In[B, D] = **AllGather**(In[B, DY]) *（在关键路径上）*
2. Tmp[B, FY] = In[B, D] \*D Win[D, FY] *（未沿收缩维度分片，无需通信）*
3. Out[B, D] {UY} = Tmp[B, FY] \*F Wout[FY, D]
4. Out[B, DY] = **ReduceScatter**(Out[B, D] {UY}) *（在关键路径上）*
5. Loss[B] = …

**反向传播：** 需要计算 dWout[FY, D]、dWin[D, FY]

1. dOut[B, DY] = …
2. dOut[B, D] = **AllGather**(dOut[B, DY]) *（在关键路径上）*
3. dWout[FY, D] = Tmp[B, FY] \*B dOut[B, D]
4. dTmp[B, FY] = dOut[B, D] \*D Wout[FY, D] *（这里可以丢弃 dOut[B, D]）*
5. In[B, D] = **AllGather**(In[B, DY]) *（可以通过共享前向传播中的 (1) 来跳过）*
6. dWin[D, FY] = dTmp[B, FY] \*B In[B, D]
7. dIn[B, D] {UY} = dTmp[B, FY] \*F Win[D, FY] *（前面层需要）*
8. dIn[B, DY] = **ReduceScatter**(dIn[B, D] {UY}) *（在关键路径上）*

张量并行的一个不错的性质是它与 Transformer 前向传播中的两个矩阵能很好地交互。直观上，我们会在两个矩阵之后各做一次 AllReduce。但这里我们先做 **In[B, DY] \* Win[D, FY] -> Tmp[B, FY]**，然后做 **Tmp[B, FY] \* Wout[FY, D] -> Out[B, DY]**。这意味着我们在开头 AllGather **In**，在结尾 ReduceScatter **Out**，而不是做 AllReduce。

**这有多昂贵？** 我们只对前向传播建模——反向传播只是这里每个操作的转置。在 1D 张量并行中，我们在第一个矩阵乘法之前 AllGather 激活，在第二个之后 ReduceScatter，每次发送两个字节（bf16）。让我们算出何时被通信瓶颈限制。

$$\begin{align}
T_\text{math} & = \frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} \\
T_\text{comms} & = \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\\
\textnormal{T} & \approx \max \left(\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C}, \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\right)
\end{align}$$

注意我们想让计算成本大于通信成本，得到：

$$\begin{align}
\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} > \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}
\end{align}$$

$$\begin{align}
\frac{F}{Y \cdot C} > \frac{1}{W_\text{ici}}
\end{align}$$

$$\begin{align}
F > Y \cdot \frac{C}{W_\text{ici}}
\end{align}$$

因此例如对于 TPUv5p，bf16 下 $C / W_{ici} = 2550$，所以我们只能做 $Y < F / 2550$ 的张量并行。当我们有多个 ICI 轴时，$T_\text{comms}$ 会减小 $M_Y$ 倍，所以我们得到 $Y < M_Y \cdot F / 2550$。

**要点：** 张量并行在 $Y > M_Y \cdot F / 2550$ 时变得通信受限。对大多数模型这在 8 到 16 路张量并行之间。

**注意这不依赖于计算的精度**，因为例如对 int8，在 TPUv5p 上 $C_\text{int8} / W_{ici}$ 是 $5100$ 而非 $2550$，但通信量也减半了，所以两个因子相互抵消。

**让我们看一些示例：**

- 在 TPUv5p 上对 LLaMA 3-70B（$D = 8192,$ $F \approx 30,000$），我们可以舒适地做 8 路张量并行，但 16 路张量并行会通信受限。8 路模型分片所需的 F 是 20k。

- 对 Gemma 7B，$F \approx 50k$，所以我们在 19 路张量并行时通信受限。这意味着我们大概可以做 16 路并仍获得良好性能。

### 组合 FSDP 和张量并行

**语法：** $\text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y]$

FSDP 和张量并行的好处在于它们可以组合。通过沿两个轴分片 **Win** 和 **Wout**，我们既节省内存又节省计算。因为我们沿 X 分片 B，我们减小了模型并行 AllGather 的大小；因为我们沿 Y 分片 F，我们减少了 FSDP 的通信开销。这意味着两者的组合可以让我们达到比上面更低的有效 batch size。

![组合 FSDP 和张量并行的示意图](https://jax-ml.github.io/scaling-book/assets/img/mixed-fsdp-model-parallelism.png)

**图：** 组合 FSDP 和张量并行的示意图。与其他情况不同，这里没有模型参数的重复。

下面是混合 FSDP + 张量并行的完整算法。虽然我们有大量通信，但所有的 AllGather 和 ReduceScatter 都更小，因为我们已经按 batch 分片了激活并按张量分片了更多权重！

**前向传播：** 需要计算 Loss[B]

1. In[BX, D] = **AllGather**Y(In[BX, DY]) *（在关键路径上）*
2. Win[D, FY] = **AllGather**X(Win[DX, FY]) *（可以提前完成）*
3. Tmp[BX, FY] = In[BX, D] \*D Win[D, FY]
4. Wout[FY, D] = **AllGather**X(Wout[FY, DX]) *（可以提前完成）*
5. Out[BX, D] {UY} = Tmp[BX, FY] \*F Wout[FY, D]
6. Out[BX, DY] = **ReduceScatter**Y(Out[BX, D] {UY}) *（在关键路径上）*
7. Loss[BX] = …

**反向传播：** 需要计算 dWout[FY, DX]、dWin[DX, FY]

1. dOut[BX, DY] = …
2. dOut[BX, D] = **AllGather**Y(dOut[BX, DY]) *（在关键路径上）*
3. dWout[FY, D] {UX} = Tmp[BX, FY] \*B dOut[BX, D]
4. dWout[FY, DX] = **ReduceScatter**X(dWout[FY, D] {UX})
5. Wout[FY, D] = **AllGather**X(Wout[FY, DX]) *（可以提前完成）*
6. dTmp[BX, FY] = dOut[BX, D] \*D Wout[FY, D] *（这里可以丢弃 dOut[B, D]）*
7. In[BX, D] = **AllGather**Y(In[BX, DY]) *（不在关键路径上 + 可与上一层中的 (2) 共享）*
8. dWin[D, FY] {UX} = dTmp[BX, FY] \*B In[BX, D]
9. dWin[DX, FY] = **ReduceScatter**X(dWin[D, FY] {UX})
10. Win[D, FY] = **AllGather**X(Win[DX, FY]) *（可以提前完成）*
11. dIn[BX, D] {UY} = dTmp[BX, FY] \*F Win[D, FY] *（前面层需要）*
12. dIn[BX, DY] = **ReduceScatter**Y(dIn[BX, D] {UY}) *（在关键路径上）*

**FSDP 和 TP 的正确组合是什么？** 一条简单但关键的格言是 FSDP 移动权重，张量并行移动激活。这意味着随着 batch size 缩小（特别是当我们做更多数据并行时），张量并行变得更便宜，因为我们每分片的激活更小。

- 张量并行执行 $\mathbf{AllGather}_Y([B_X, D_Y])$，随 $X$ 增加而缩小。
- FSDP 执行 $\mathbf{AllGather}_X([D_X, F_Y])$，随 $Y$ 增加而缩小。

因此通过组合两者，我们可以将每副本的最小 batch size 推得更低。我们可以用与上面相同的方式计算最优的 FSDP 和 TP 量：

设 $X$ 为用于 FSDP 的芯片数，$Y$ 为用于张量并行的芯片数。设 $N$ 为切片中的总芯片数，$N=XY$。设 $M_X$ 和 $M_Y$ 分别为 FSDP 和 TP 所跨越的 mesh 轴数（它们大致应加起来等于 3）。我们只对前向传播建模，因为它每 FLOP 的通信量最大。则将上面算法中的通信加总，我们有

$$T_\text{FSDP comms}(B, X, Y) = \frac{2\cdot 2\cdot D \cdot F}{Y \cdot W_\text{ici} \cdot M_X}$$

$$T_\text{TP comms}(B, X, Y) = \frac{2 \cdot 2 \cdot B \cdot D}{X \cdot W_\text{ici} \cdot M_Y}$$

同样我们的总 FLOPs 时间为

$$T_\text{math} = \frac{2\cdot 2 \cdot B \cdot D \cdot F}{N \cdot C}.$$

为简化分析，我们做两个假设：第一，允许 $X$ 和 $Y$ 取非整数值（只要它们为正且满足 $XY=N$）；第二，假设我们可以完全将 $X$ 和 $Y$ 轴上的通信彼此重叠。在第二个假设下，总通信时间为

$$T_\text{comms} = \max\left(T_\text{FSDP comms}, T_\text{TP comms}\right)$$

在我们询问什么条件下计算受限之前，先找出最小化总通信的 $X$ 和 $Y$ 最优值。由于 FLOPs 与 $X$ 和 $Y$ 无关，最优设置就是简单地最小化通信。为此，让我们将上面的 $T_\text{comms}$ 用 $X$ 和 $N$（保持固定，因为它是系统中的芯片数）而非 $X$ 和 $Y$ 来表示：

$$T_\text{comms} (X) = \frac{4D}{W_\text{ici}} \max\left(\frac{F \cdot X}{N \cdot M_X}, \frac{B}{X \cdot M_Y}\right)$$

由于 $T_\text{FSDP comms}$ 在 $X$ 上单调递增，$T_\text{TP comms}$ 在 $X$ 上单调递减，最大值必须在 $T_\text{FSDP comms} = T_\text{TP comms}$ 时被最小化，即

$$\begin{align*}
\frac{FX_{opt}}{M_X} = \frac{BN}{X_{opt} M_Y} \rightarrow \\
X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}
\end{align*}$$

这超有用！这告诉我们，对给定的 $B$、$F$ 和 $N$，最优的 FSDP 量是多少。让我们感受一下规模。代入实际值，即 $N = 64$（对应 4x4x4 芯片阵列），$B=48,000$，$F=32768$，得到大约 $X\approx 13.9$。所以我们会选 $X$ 为 16 而 $Y$ 为 4，接近我们计算的最优值。

**要点：** 一般而言，训练时最优的 FSDP 量为 $X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}$。

现在让我们回到一直问的问题：**在什么条件下我们计算受限？** 由于我们可以重叠 FLOPs 和通信，我们在以下情况下计算受限

$$\max\left(T_\text{FSDP comms}, T_\text{TP comms}\right) < T_\text{math}$$

设 $\alpha \equiv C / W_\text{ici}$ 为 ICI 算术强度，我们可以简化：

$$\max\left(\frac{F}{Y \cdot M_X}, \frac{B}{X \cdot M_Y}\right) < \frac{B \cdot F}{N \cdot \alpha}$$

由于我们计算 $X_{opt}$ 使 LHS 最大值相等，我们可以将其代入任一侧（注意 $Y_{opt} = N/X_{opt}$），即

$$\frac{F}{N \cdot W_\text{ici} \cdot M_X} \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N} < \frac{B \cdot F}{N \cdot C}$$

进一步简化，我们发现

$$\sqrt{\frac{B\cdot F}{M_X \cdot M_Y \cdot N}} < \frac{B \cdot F}{N \cdot \alpha},$$

其中左侧与通信时间成比例，右侧与计算时间成比例。注意虽然计算时间随 batch size 线性缩放（无论何种并行都如此），通信时间按 batch size 的平方根缩放。计算时间与通信时间的比也按 batch size 的平方根缩放：

$$\frac{T_\text{math}}{T_\text{comms}} = \frac{\sqrt{BF}\sqrt{M_X M_Y}}{\alpha \sqrt{N}}.$$

为确保该比大于 1 以使我们计算受限，我们要求

$$\frac{B}{N} > \frac{\alpha^2}{M_X M_Y F}$$

为得到近似数字，再次代入 $F=32,768$、$\alpha=2550$、$M_X M_Y=2$（3D mesh 必须如此）。这给出大约 $B/N > 99$。这相比纯数据并行（或 FSDP）情况赢得了约 8 倍因子，那种情况下假设 3D mesh 我们计算 $B/N$ 必须超过约 $850$ 才能计算受限。

**要点：** 结合张量并行与 FSDP 让我们可以将 $B/N$ 降到 $2550^2 / 2F$。这让我们能处理低至每芯片 100 的 batch，比仅 FSDP 能达到的小约 8 倍。

下面我们绘制混合 FSDP + TP 的 FLOPs 与通信时间比，将其与仅张量并行（TP）和仅数据并行（FSDP）在代表性 4x4x4 芯片阵列上比较。虽然在非常大的 batch size 下纯 FSDP 并行占优，但在 batch size 与芯片数的比在大约 100 到 850 之间的范围内，需要混合 FSDP + TP 策略才能计算受限。

![最优混合 FSDP/TP 的 FLOPs 与通信时间比](https://jax-ml.github.io/scaling-book/assets/img/mixed-fsdp-comms-2.png)

**图：** TPUv5p 4x4x4 切片上 F=30k 时最优混合 FSDP/TP 的 FLOPs 与通信时间比。如预期，张量并行随 batch size 有固定比；理想混合 FSDP + TP 按 $\sqrt{B}$ 缩放，FSDP 按 $B$ 缩放。然而在中间 batch size 范围内，只有 FSDP + TP 达到大于 1 的比值。

下面是另一个 TPU v5p 16x16x16 的示例，展示不同分片方案下 FLOPs 和通信时间随 batch size 的变化。

![不同并行方案下的通信耗时](https://jax-ml.github.io/scaling-book/assets/img/math-comms-time.png)

**图：** 不同并行方案下的通信耗时。黑色虚线是矩阵乘法 FLOPs 的耗时，因此该线之上的任何曲线都是通信受限。我们注意到所有策略在 batch size 低于 6e5 时变得通信受限，这与我们预期的 4096 \* 2550^2 / (2 \* 8192 \* 4) = 4e5 一致。

黑色曲线是花在模型 FLOPs 上的时间，意味着任何 batch size 在该曲线低于所有通信成本时都严格通信受限。你会注意到黑色曲线在大约 `4e5` 处与绿色曲线相交，正如预测的那样。

下面是一个交互式动画可以玩玩，展示不同 batch size 下的总计算时间和通信时间：

你会注意到这与上面大体一致（最小值约在 FSDP=256，TP=16），加减一些波动因子，因为每种方法的轴数有些差异。

### 流水线（Pipelining）

你可能注意到我们在前几节完全避免谈论流水线。流水线是 GPU 并行的主导策略，但在 TPU 上不那么必要。简而言之，流水线训练涉及将模型的层划分到多个设备上，并在前向和反向传播期间在流水线阶段间传递激活。算法大致如下：

1. 在 TPU 0 上初始化数据，权重沿层维度分片（对带 FSDP 和张量并行的流水线为 $W_\text{in}[L_Z, D_X, F_Y]$）。
2. 在 TPU 0 上执行第一层，然后将得到的激活复制到 TPU 1，重复直到到达最后一个 TPU。
3. 计算损失函数及其导数 $\partial L / \partial x_L$。
4. 对最后一个流水线阶段，计算导数 $\partial L / \partial W_L$ 和 $\partial L / \partial x_{L-1}$，然后将 $\partial L / \partial x_{L-1}$ 复制到上一个流水线阶段，重复直到到达 TPU 0。

下面是一些（可工作的）Python 伪代码

这段伪代码应该能在 Cloud TPU VM 上运行。虽然它不太高效或现实，但能让你感受数据如何在设备间传播。

```python
batch_size = 32
d_model = 128
d_ff = 4 * d_model

num_layers = len(jax.devices())

key = jax.random.PRNGKey(0)

# Pretend each layer is just a single matmul.
x = jax.random.normal(key, (batch_size, d_model))
weights = jax.random.normal(key, (num_layers, d_model, d_model))

def layer_fn(x, weight):
  return x @ weight

# Assume we have num_layers == num_pipeline_stages
intermediates = [x]
for i in range(num_layers):
  x = layer_fn(x, weights[i])
  intermediates.append(x)

  if i != num_layers - 1:
    x = jax.device_put(x, jax.devices()[i+1])

def loss_fn(batch):
  return jnp.mean(batch ** 2)  # make up some fake loss function

loss, dx = jax.value_and_grad(loss_fn)(x)

for i in range(num_layers - 1, -1, -1):
  _, f_vjp = jax.vjp(layer_fn, intermediates[i], weights[i])
  dx, dw = f_vjp(dx)  # compute the jvp dx @ J(L)(x[i], W[i])
  weights[i] = weights[i] - 0.01 * dw  # update our weights

  if i != 0:
    dx = jax.device_put(dx, jax.devices()[i-1])
```

**为什么这是个好主意？** 流水线很棒有很多原因：它在流水线阶段间通信成本低，意味着即使在低带宽互连下你也能训练非常大的模型。这在 GPU 上常常很有用，因为 GPU 不像 TPU 那样通过 ICI 密集互连。

**为什么这困难/烦人？** 你可能注意到上面的伪代码中 TPU 0 几乎一直空闲！它只在流水线的第一步和最后一步工作。这段空闲期称为流水线气泡（pipeline bubble），处理起来非常烦人。我们通常先用微批化（microbatching）来缓解，它通过流水线发送多个小批次，让 TPU 0 在总步骤时间的较大比例内保持活跃。

第二种方法是仔细重叠前向矩阵乘法 $W_i @ x_i$、反向 $dx$ 矩阵乘法 $W_i @ \partial L / \partial x_{i+1}$ 和 $dW$ 矩阵乘法 $\partial L / \partial x_{i+1} @ x_i$。由于每个都需要一些 FLOPs，我们可以重叠它们以完全隐藏气泡。下面是来自最近 DeepSeek v3 论文的图，显示了他们的"无气泡"流水线调度：

![DeepSeek v3 流水线调度](https://jax-ml.github.io/scaling-book/assets/img/deepseek-pipeline.png)

**图：** DeepSeek v3 流水线调度（来自他们的[最近论文](https://arxiv.org/pdf/2412.19437)）。橙色是前向矩阵乘法，绿色是 dL/dx 矩阵乘法，蓝色是 dL/dW 矩阵乘法。通过优先处理反向 dL/dx 乘法，我们可以避免"搁浅"FLOPs。

由于流水线对 TPU（具有更大互连 pod）来说不那么关键，我们不会深入探讨，但理解流水线的关键瓶颈是个好练习。

### 跨 Pod 扩展

最大的 TPU 切片是含 8960 芯片（和 2240 个主机）的 TPU v5p SuperPod。当我们想超出这个规模扩展时，需要跨越数据中心网络（Data-Center Networking, DCN）边界。每个 TPU 主机配备一张或多张 NIC（Network Interface Cards），通过以太网将主机连接到其他 TPU v5p pod。如 [TPU 节](../part2_tpus)所述，每个主机大约有 200Gbps（25GB/s）的全双工 DCN 带宽，即每 TPU 约 6.25GB/s 全双工（出口）带宽。

通常，当扩展超出单个 pod 时，我们在 ICI 域内做某种模型并行或 FSDP，然后跨多个 pod 做纯数据并行。设 $N$ 为我们想扩展到的 TPU 数，$M$ 为每个 ICI 连接切片中的 TPU 数。要在 DCN 上做 AllReduce，我们可以在 pod 集合上做环形规约（ring-reduction），得到（在反向传播中）：

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{N \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF}{M \cdot W_\text{dcn}}$$

通信带宽随 $M$ 缩放，因为不像 ICI，总带宽随我们扩大 ICI 域并获得更多 NIC 而增长。简化后我们发现 $T_\text{math} > T_\text{comms}$ 当

$$\frac{B}{\text{slice}} > \frac{C}{W_\text{dcn}}$$

对 TPU v5p，$\frac{C}{W_\text{dcn}}$ 约为 `4.46e14 / 6.25e9 = 71,360`。这告诉我们，为了在 DCN 上高效扩展，每个 ICI 域需要一个最小 batch size 来出口每个节点。

**这有多大问题？** 举个具体例子，假设我们想在 TPU v5p 上以 BS=2M tokens 训练 LLaMA-3 70B。LLaMA-3 70B 有 $F\approx 30,000$。从上面几节我们知道：

- 我们可以做张量并行直到 $Y = M_Y \cdot F / 2550 \approx 11 \cdot M_Y$。
- 我们可以做 FSDP 只要 $B / N > 2550 / M_X$。这意味着如果我们想以 BS=2M 和 3 轴数据并行训练，最多能用约 $2400$ 芯片，约 TPU v5p pod 的四分之一。
- 当我们组合 FSDP + 张量并行，在 $B / N < 2550^2 / (2 \cdot 30000) = 108$ 时变成通信受限，所以这让我们扩展到约 18k 芯片！但 TPU v5p pod 的最大尺寸是 8k 芯片，所以超出后我们必须使用 DCN。

总结是我们有一个不错的 BS=1M 训练配方，使用约 X (FSDP) = 1024 和 Y (TP) = 8，但 BS=2M 时我们需要使用 DCN。如上所述，我们的 DCN 算术强度是 $\text{71,360}$，所以我们只需确保每 ICI 域 batch size 大于此。这对我们来说轻而易举，因为用 2 个 pod 我们每 pod BS 为 1M，每 TPU batch size 为 111，这很好（也许有点紧，但理论上可行）。

**要点：** 跨多个 TPU pod 的扩展用纯数据并行相当直接，只要每 pod batch size 至少 71k tokens。

## 在 TPU 上训练 LLM 的要点

- 增加并行度或减少 batch size 都倾向于让我们更受通信限制，因为它们减少了每芯片执行的计算量。

- 在合理的上下文长度内（~32k），我们可以将 Transformer 建模为 MLP 块的堆叠，并通过它们如何分片每层的两/三个主要矩阵乘法来定义几种并行方案。

- 训练期间我们考虑 4 种主要并行方案，每种都有自己的带宽和计算需求（数据并行、FSDP、张量并行和混合 FSDP + 张量并行）。

| **策略** | **描述** |
|---|---|
| **数据并行** | 激活按 batch 分片，其他完全复制，反向传播时 all-reduce 梯度。 |
| **FSDP** | 激活、权重和优化器都按 batch 分片，权重在使用前 gather，梯度被 reduce-scatter。 |
| **张量并行（也称 Megatron、模型并行）** | 激活沿 $d_\text{model}$ 分片，权重沿 $d_{ff}$ 分片，激活在 Win 之前 gather，结果在 Wout 之后 reduce-scatter。 |
| **混合 FSDP + 张量并行** | 同时使用上述两者，FSDP gather 模型分片的权重。 |

下面是每种方法的"公式"：

$$\small \begin{array}{cc}
\text{策略} & \text{公式}\\
\hline
\text{DP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D] \\
\text{FSDP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D] \\
\text{TP} & \text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y] \\
\text{TP + FSDP} & \text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y] \\
\hline
\end{array}$$

- 每种策略都有一个变成网络/通信受限的极限，基于其每设备计算和通信。下面是每层的计算和通信，假设 $X$ 是 FSDP，$Y$ 是张量并行。

$$\small \begin{array}{ccc}
\text{策略} & \text{每层计算} & \text{每层通信} \\
& \text{（忽略门控 einsum）} & \text{（字节，前向 + 反向）}\\
\hline
\text{DP} & 4BDF/X + 8BDF/X & 0 + 8DF \\
\text{FSDP} & 4BDF/X + 8BDF/X & 4DF + 8DF \\
\text{TP} & 4BDF/Y + 8BDF/Y & 4BD + 4BD \\
\text{FSDP + TP} & 4BDF/(XY) + 8BDF/(XY) & (4BD/X + 4DF/Y) + (8BD/X + 8DF/Y) \\
\hline
\end{array}$$

- 纯数据并行很少有用，因为模型及其优化器状态使用 字节 = 10x 参数数。这意味着我们很少能在内存中放下超过几十亿参数。

- 数据并行和 FSDP 在每分片 batch size 小于 $C / W$（网络的算术强度）时通信受限。对 ICI 这是 2,550，对 DCN 约为 71,000。这可以通过更多并行轴来增加。

- 张量并行在 $\lvert Y\rvert > F / 2550$ 时通信受限。**对大多数模型这约为 8-16 路。** 这与 batch size 无关。

- 混合 FSDP + 张量并行让我们将 batch size 降到低至 $2550^2 / 2F \approx 100$。这非常低。

- 跨 pod 数据并行在变成 DCN 受限前需要每 pod 最小 batch size 约 71,000。

- 基本上，如果你的 batch size 很大或模型很小，事情很简单。你可以做数据并行，或在 DCN 上做 FSDP + 数据并行。中间部分才是有趣之处。

## 一些练习题

让我们用 LLaMA-2 13B 作为本节的基础模型。这是模型详情：

| 超参数 | 值 |
|---|---|
| L | 40 |
| D | 5,120 |
| F | 13824 |
| N | 40 |
| K | 40 |
| H | 128 |
| V | 32,000 |

LLaMA-2 有独立的嵌入和输出矩阵以及门控 MLP 块。

**问题 1：** LLaMA-2 13B 有多少参数（我知道这很傻，但做一下数学）？*注意，与 [Transformer Math](../part4_transformers) 一样，LLaMA-3 有 3 个大 FFW 矩阵，两个上投影和一个下投影。我们在本节忽略了两个"门控"einsum 矩阵，但它们的行为与本节的 Win 相同。*

<details>
<summary>点击查看答案。</summary>

- FFW 参数：$3LDF$ = `8.5e9`
- 注意力参数：$4DNHL$ = `4.2e9`
- 词汇参数：$2VD$ = `0.33e9`
- 总计：`8.5e9 + 4.2e9 + 0.33e9 = 13.0e9`，如预期！

</details>

**问题 2：** 假设我们以 BS=16M tokens 用 Adam 训练。暂时忽略并行，模型的参数、优化器状态和激活总共使用多少内存？*假设我们以 bf16 存储参数，以 fp32 存储优化器状态，并每层 checkpoint 激活三次（在三个大矩阵乘法之后）。*

<details>
<summary>点击查看答案。</summary>

参数（bf16）和两个优化器状态（fp32，一阶和二阶累加器）使用的总内存为 `(2 + 4 + 4) * 13e9 ~ 130GB`。前两个矩阵乘法之后的激活形状为 $BF$，最后一个之后为 $BD$（按上面 Transformer 图），所以 bf16 总内存为 $2 \cdot L \cdot (BD + 2 * BF) = 2LB \cdot (D + 2F)$ 或 `2 * 40 * 16e6 * 5,120 * (1 + 2 * 2.7) ~ 4.2e13 = 42TB`，因为 `B=16e6`。所有其他激活或多或少可以忽略。

</details>

**问题 3：** 假设我们想在 TPUv5p 16x16x16 切片上以 32k 序列长度和 3M tokens 总 batch size 训练。假设我们想用 bfloat16 权重和 float32 优化器，如上所述。

1. 我们能用纯数据并行吗？为什么？
2. 我们能用纯 FSDP 吗？为什么？用纯 FSDP 每设备会用多少内存（假设我们只在 3 个大 FFW 矩阵后做梯度 checkpoint）。
3. 我们能用混合 FSDP + 张量并行吗？为什么？如果可以，$X$ 和 $Y$ 应是多少？每设备会存多少内存？仅用 roofline FLOPs 估计并忽略注意力，每个训练步在 40% MFU 下需多长时间？

<details>
<summary>点击查看答案。</summary>

首先，让我们写下一些数字。32k 序列长度和 3M batch size，我们的序列 batch size 为 96。在 TPU v5p 16x16x16 切片上，我们有 `393TB` HBM。

1. 我们不能用纯数据并行，因为它在每个芯片上复制参数和优化器状态，已经约 130GB（来自 Q2），超过了我们每芯片有的 HBM（96GB）。

2. 让我们先单纯看内存。将 Q2 中的 BS=16M 替换为 3M，我们得到 `~7.86e12` 总 checkpoint 激活，加上 1.3e11 优化器状态使我们几乎正好达到 8e12 = 8TB。TPUv5p 切片总共有 `393TB` HBM，所以我们安全地在 HBM 限制下。接下来看是否会通信受限或计算受限。在 4096 芯片和 3 轴并行下，我们能做的最小 batch size 为 `850 * 4096 = 3.48M` tokens。这略高于我们的 3M batch size。所以我们实际上是通信受限，这很糟。所以一般答案是**不，我们不能单独做 FSDP**。

3. 现在我们知道主要担忧是通信受限，所以让我们代入一些数字。首先，我们从上面知道用混合 FSDP + 张量并行时每芯片 batch size 需要在 $2550^2 / 2F = 235$ 之上。这意味着理论上我们可以做！让我们算出每种多少。

我们有规则 $X_{opt} = \sqrt{(B / F) \cdot (M_X / M_Y) \cdot N}$，所以这里我们有 `sqrt(3e6 * 2 * 4096 / 13824) = 1333`，意味着我们做约 1024 路 DP 和 4 路 TP。每 TPU 内存如 (2)，步骤时间为 `6 * 3e6 * 13e9 / (4096 * 4.6e14 * 0.4) = 300ms`。

</details>

**第 5 部分到此结束！要将本内容应用到真实 LLaMA 模型的第 6 部分，[请点击这里](../part6_applied_training)！**

## 附录

### 附录 A：推导反向传播通信

上面，我们将 Transformer 层前向传播简化为 Out[B, D] = In[B, D] \*D Win[D, F] \*F Wout[F, D]。如何推导反向传播所需的通信？

这相当自然地由前一节中单矩阵乘法 **Y = X \* A** 的规则得出：

$$\frac{dL}{dA} = \frac{dL}{dY}\frac{dY}{dA} = X^T \left(\frac{dL}{dY}\right)$$

$$\frac{dL}{dX} = \frac{dL}{dY}\frac{dY}{dX} = \left(\frac{dL}{dY}\right) A^T$$

使用这个，我们得到以下公式（设 Tmp[B, F] 代表 In[B, D] \* Win[D, F]）：

1. dWout[F, D] = Tmp[B, F] \*B dOut[B, D]
2. dTmp[B, F] = dOut[B, D] \*D Wout[F, D]
3. dWin[D, F] = In[B, D] \*B dTmp[B, F]
4. dIn[B, D] = dTmp[B, F] \*F Win[D, F]

注意这些公式是数学陈述，没有提到分片。反向传播的工作就是计算这四个量。所以要算出所需的通信，我们只需取上面四个方程中要矩阵乘法的所有量（Tmp、dOut、Wout、Win）的分片（由我们的并行方案指定），并用分片矩阵乘法的规则算出我们必须做什么通信。注意 dOut 的分片方式与 Out 相同。

### 杂项

\*工作在 Google DeepMind 完成，现就职于 MatX。

### 引用

学术语境中的引用，请将本工作引为：

```
Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.
```

或作为 BibTeX 条目：

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
