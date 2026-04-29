---
title: "Transformers"
date: 2026-04-29
draft: false
math: true
weight: 5
---

{{< katex >}}

# 你需要了解的所有 Transformer 数学（All the Transformer Math You Need to Know）

《如何扩展你的模型》（How To Scale Your Model）第 4 部分（[第 3 部分：Sharding](../part3_sharding) | [第 5 部分：Training](../part5_training)）

这里我们将快速回顾 Transformer 架构，特别是如何计算 FLOPs、字节数和其他感兴趣的量。

**目录**

- [数点积（Counting Dots）](#counting-dots)
  - [前向与反向 FLOPs（Forward and reverse FLOPs）](#forward-and-reverse-flops)
- [Transformer 计算（Transformer Accounting）](#transformer-accounting)
- [全局 FLOPs 与参数计算（Global FLOPs and Params Calculation）](#global-flops-and-params-calculation)
- [杂项数学（Miscellaneous Math）](#miscellaneous-math)
  - [稀疏性与混合专家（Sparsity and Mixture-of-Experts）](#sparsity-and-mixture-of-experts)
  - [梯度检查点（Gradient checkpointing）](#gradient-checkpointing)
  - [键值缓存（Key-Value (KV) caching）](#key-value-kv-caching)
- [本节要点（What Should You Take Away from this Section?）](#what-should-you-take-away-from-this-section)
- [一些练习题（A Few Problems to Work）](#a-few-problems-to-work)
- [附录（Appendix）](#appendix)
  - [附录 A：Flash Attention 如何工作？（Appendix A: How does Flash Attention work?）](#appendix-a-how-does-flash-attention-work)

## 数点积（Counting Dots） {#counting-dots}

让我们从形状如下的向量 $x$、$y$ 和矩阵 $A$、$B$ 开始：

$$\def \red#1{\textcolor{red}{#1}} \def \green#1{\textcolor{green}{#1}} \def \blue#1{\textcolor{blue}{#1}} \def \purple#1{\textcolor{purple}{#1}} \def \orange#1{\textcolor{orange}{#1}} \def \gray#1{\textcolor{gray}{#1}} \begin{array}{cc} \textrm{array} & \textrm{shape} \\ \hline x & \textrm{[P]} \\ y & \textrm{[P]} \\ A & \textrm{[N P]} \\ B & \textrm{[P M]} \\ \hline \end{array}$$

- $x \cdot y$ 的点积（dot product）需要 $P$ 次*加法*和*乘法*，总共 $2P$ 次浮点运算。
- 矩阵-向量乘积 $Ax$ 沿 $A$ 的行做 $N$ 次点积，共 $2NP$ FLOPs。
- 矩阵-矩阵乘积 $AB$ 对 $B$ 的 $M$ 列各做一次矩阵-向量乘积，共 $2NPM$ FLOPs。
- 一般来说，如果我们有两个高维数组 $C$ 和 $D$，其中一些维度是 CONTRACTING（收缩）的，一些是 BATCHING（批处理）的（例如 $C[\blue{GH}IJ\red{KL}], D[\blue{GH}MN\red{KL}]$），那么这次收缩的 FLOPs 成本是 $C$ 和 $D$ 所有维度乘积的 2 倍，其中批处理和收缩维度只计算一次（例如 $2\blue{GH}IJMN\red{KL}$）。注意，只有当一个维度同时出现在两个被乘数中时，它才是批处理维度。（还要注意，如果没有收缩维度而仅仅是逐元素乘积，则不会有 2 这个因子。）

> **收缩**（Contracting）维度是在操作中被求和的轴（它们出现在两个输入中但不在输出中），如矩阵乘法中的内部维度。**批处理**（Batching）维度是出现在两个输入中并原样传递到输出的共享轴；它们对独立的子问题进行索引，并且不会在 FLOP 计数中相乘。用 einsum 术语来说：同时出现在两个输入和输出中的标签是批处理；同时出现在两个输入中但不在输出中的标签是收缩。

$$\begin{array}{ccc} \textrm{Operation} & \textrm{FLOPs} & \textrm{Data} \\ \hline x \cdot y & 2P & 2P \\ A x & 2NP & NP + P \\ AB & 2NPM & NP + PM \\ [c_0,...,c_N] \cdot [d_0,...,d_N] & 2 \prod c_i \times \prod_{\substack{d_j \notin \blue{BATCH} \\ d_j \notin \red{CONTRACT}}} d_j & \prod c_i + \prod d_j \\ \hline \end{array}$$

请注意，对于矩阵-矩阵乘法，*计算*以三次方 $O(N^3)$ 增长，而数据传输仅以二次方 $O(N^2)$ 增长——这意味着随着我们扩大 matmul 的规模，达到计算饱和极限变得*更容易*。这是非常不寻常的，并在很大程度上解释了为什么我们使用以矩阵乘法为主导的架构——它们易于扩展！

![](/scaling-book/assets/img/matmul-flops.gif)

### 前向与反向 FLOPs（Forward and reverse FLOPs） {#forward-and-reverse-flops}

在训练期间，我们并不特别关心给定矩阵乘法的结果；我们真正关心的是它的导数。这意味着我们在反向传播（backpropagation）期间会做明显更多的 FLOPs。

如果我们设想 **B** 只是一个更大网络中的一个矩阵，**A** 是我们的输入激活值（input activations），且 **C = A B**，那么损失（loss）**L** 关于 **B** 的导数由链式法则给出：

$$\frac{\partial L}{\partial B} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial B} = A^T \left(\frac{\partial L}{\partial C}\right)$$

它需要 $2NPM$ FLOPs 来计算（因为它在 $N$ 维度上收缩）。同样地，损失关于 **A** 的导数

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial A} = \left(\frac{\partial L}{\partial C}\right) B^T$$

也是 $2NPM$ FLOPs，因为 **dL/dC** 是大小为 $[N, M]$ 的矩阵。虽然这个量不是关于参数的导数，但它用于计算网络前面层的导数（例如，正如 dL/dC 在上面用于计算 dL/dB 一样）。

将这些相加，我们看到**在训练期间，我们总共有 6NPM FLOPs**，相比之下，推理时只有 2NPM：前向传递 2NPM，反向传递 4NPM。由于 PM 是矩阵中的参数数量，这是著名的 Transformer 训练 FLOPs 的 $6 * \text{num parameters} * \text{num tokens}$ 近似的最简形式：每个 token 需要 $6 * \text{num parameters}$ FLOPs。我们将在下面展示更准确的推导。

## Transformer 计算（Transformer Accounting） {#transformer-accounting}

Transformer 是未来。嗯，至少是现在。也许几年前，它们只是众多架构之一。但今天，了解架构的几乎每一个细节都是值得的。我们不会重新介绍这个架构，但[这篇博客](https://jalammar.github.io/illustrated-transformer/)和[原始 Transformer 论文](https://arxiv.org/abs/1706.03762)可能是有用的参考。

这是 Transformer 解码器（decoder）架构的基本图示：

![](/scaling-book/assets/img/transformer-diagram.png)

**图：**该图显示了一个标准 Transformer 的一层，从上到下流动。我们使用单字母约定来描述 Transformer 中数组的形状和布局，再次用红色显示收缩维度，蓝色显示批处理维度。在给定操作中，输入形状显示在左上角，参数形状显示在右上角，结果形状在下方，例如 BTD 是门控 einsum 的输入形状，DF 是权重形状。

**注 [门控 einsum]**：上面的图使用了"[门控 einsums](https://arxiv.org/abs/2002.05202)"，我们将上投影矩阵分成两个矩阵（上面的 $W_\text{In1}$ 和 $W_\text{In2}$），它们的输出按元素相乘作为一种"门控函数"。并非所有 LLM 都使用这种方式，所以你有时会看到单个 $W_\text{In}$ 矩阵和总 MLP 参数数为 2DF 而不是 3DF。通常在这种情况下，D 和 F 会被放大以保持参数数量与 3 矩阵的情况相同。话虽如此，某种形式的门控 einsum 被 LLaMA、DeepSeek 和许多其他模型使用。

**注 2 [MHA 注意力]**：对于自注意力（self-attention），T 和 S 是相同的，但对于交叉注意力（cross-attention）它们可能不同。对于普通的多头注意力（Multi-Head Attention，MHA），N 和 K 相同，而对于[多查询注意力（Multi-Query Attention）](https://arxiv.org/abs/1911.02150)（MQA），K=1；对于[分组 MQA（Grouped MQA）](https://arxiv.org/abs/2305.13245)（GMQA），K 只需要能整除 N。

**注 3 [pre-norm 与 post-norm]：**上面的图显示了所谓的"post-norm" Transformer，其中 layernorm 发生在残差连接（residual connection）之后，即 `norm(x + attn(x))`。这与原始 Transformer 论文一致，但今天大多数现代 Transformer 使用"pre-norm"架构，其中 norm 发生在残差连接之前，通常为 `x + attn(norm(x))`。像 LLaMA-3 这样的模型今天使用这种方式。

## 全局 FLOPs 与参数计算（Global FLOPs and Params Calculation） {#global-flops-and-params-calculation}

下面我们将计算每层 FLOPs，避免到处写 **L** 因子。

### MLPs

Transformer 的 MLP 通常包括 2 个按元素组合的输入 matmul 和一个输出 matmul：

$$\begin{array}{ccc} \textrm{operation} & \textrm{train FLOPs} & \textrm{params} \\ \hline \\ A[B,T,\red{D}] \cdot W_{in1}[\red{D}, F] & 6BTDF & DF \\\\[10pt] A[B,T,\red{D}] \cdot W_{in2}[\red{D}, F] & 6BTDF & DF \\\\[10pt] \sigma\left(A_{in1}\right)[B,T, F] * A_{in2}[B,T, F] & \gray{O(BTF)} \\\\[10pt] A[B,T,\red{F}] \cdot W_{out}[\red{F}, D] & 6BTDF & DF \\\\[10pt] \hline \\ & \approx 18BTDF & 3DF \end{array}$$

### 注意力（Attention）

对于具有不同 **Q** 和 **KV** 头数的通用分组查询注意力（grouped-query attention）情况，让我们假设 **Q**、**K**、**V** 投影具有相等的头维度 H，并估计 **QKVO** matmul 的成本：

$$\begin{array}{ccc} \textrm{operation} & \textrm{train FLOPs} & \textrm{params} \\ \hline \\ A[B,T,\red{D}] \cdot W_{Q}[\red{D}, N, H] & 6BTDNH & DNH \\\\[10pt] A[B,T,\red{D}] \cdot W_{K}[\red{D}, K, H] & 6BTDKH & DKH \\\\[10pt] A[B,T,\red{D}] \cdot W_{V}[\red{D}, K, H] & 6BTDKH & DKH \\\\[10pt] A[B,T,\red{N}, \red{H}] \cdot W_{O}[\red{N}, \red{H}, D] & 6BTDNH & DNH \\\\[10pt] \hline \\ & 12BTD(N+K)H & 2D(N+K)H \end{array}$$

点积注意力操作更微妙，本质上是在 $B$、$K$ 维度上批处理的 $TH \cdot HS$ matmul，加上一个 softmax，再加上在 $B$、$K$ 维度上批处理的 $TS \cdot SH$ matmul。我们用蓝色突出显示批处理维度：

$$\begin{array}{cc} \textrm{operation} & \textrm{train FLOPs} \\ \hline \\[3pt] Q[\blue{B}, T, \blue{K}, G, \red{H}] \cdot K[\blue{B}, S, \blue{K}, \red{H}] & 6BTSKGH = 6BTSNH \\[3pt] \textrm{softmax}_S \;\; L[B, T, S, K, G] & \gray{O(BTSKG) = O(BTSN)} \\[3pt] S[\blue{B}, T, \red{S}, \blue{K}, G] \cdot V[\blue{B}, \red{S}, \blue{K}, H] & 6BTSKGH = 6BTSNH \\[3pt] \hline \\ & \approx 12BTSNH = 12BT^2NH \\ \end{array}$$

**注 [因果掩码（causal masking）]**：大多数最近的 Transformer 使用因果掩码，而不是全双向注意力。在这种情况下，点积操作的有用 FLOPs 减少了 1/2。要在实践中实现这种减少，我们需要使用注意力内核（attention kernel），而不是简单的 einsum。

### 其他操作

Transformer 中还发生了其他几个操作。Layernorm 相对便宜，可以在一阶成本估计中忽略。还有最后巨大的（虽然不是按层计算的）反嵌入（unembedding）矩阵乘法。

$$\begin{array}{ccc} \textsf{operation} & \textsf{train FLOPs} & \textsf{params} \\ \hline \\ \textrm{layernorm}_D \;\; A[B,T,\red{D}] & \gray{O\left(BTD\right)} & \gray{D} \\\\[10pt] A[B,T,\red{D}] \cdot W_{unembed}[\red{D}, V] & 6BTDV & DV \\ \end{array}$$

### Transformer FLOPs 的一般经验法则

如果我们忽略短上下文训练中点积注意力的成本，那么所有层的总 FLOPs 是

$$\begin{align*} (18BTDF + 12BTD(N+K)H)L = 6 *BT * (3DF + 2D(N+K)H)L \\ = 6 * \textrm{num tokens} * \textrm{parameter count} \end{align*}$$

得出了一个估计稠密 Transformer FLOP 计数的著名经验法则，忽略了注意力 FLOPs。（反嵌入是另一个简单的 matmul，具有 $6BTDV$ FLOPs 和 $DV$ 参数，并遵循相同的经验法则。）

### 注意力随上下文长度的相对成本

如果我们考虑上面的点积注意力并假设 $F=4D$、$D=NH$（这是典型情况）和 $N=K$：

$$\small{\frac{\textrm{attention FLOPs}}{\textrm{matmul FLOPs}} = \frac{12BT^2NH}{18BTDF + 24BTDNH} = \frac{12BT^2D}{4*18 BTD^2 + 24 BTD^2} = \frac{12BT^2D}{96 BTD^2} = \frac{T}{8D}}$$

所以要点是**点积注意力的 FLOPs 在训练期间只有当 T>8D 时才占主导地位**。对于 D ~ 8k，这大约是 ~64K tokens。这有一定道理，因为它意味着随着 MLP 大小的增加，注意力 FLOPs 变得不那么关键。对于大型模型，注意力的二次成本实际上不是长上下文训练的巨大障碍。然而，对于较小的模型，即使是 Gemma-27B，D=4608，这意味着注意力在 32k 序列长度左右开始占主导。Flash Attention 也有助于缓解长上下文的成本，我们在[附录 A](#appendix-a-how-does-flash-attention-work) 中简要讨论。

## 杂项数学（Miscellaneous Math） {#miscellaneous-math}

### 稀疏性与混合专家（Sparsity and Mixture-of-Experts） {#sparsity-and-mixture-of-experts}

我们不能不简要讨论混合专家（Mixture of Experts，MoE）模型，它用一组可以动态路由的独立 MLP 替换标准 Transformer 中的单个稠密 MLP 块。粗略地说，**MoE 只是一个普通的稠密模型，每层有 E 个 MLP 块**，而不是只有一个。每个 token 激活其中的 $k$ 个专家，通常 $k \ll E$。比率 $E / k$ 称为稀疏性（sparsity），通常在 8 到 64 之间（例如，[DeepSeek v3](https://arxiv.org/pdf/2412.19437) 实际上 $k=8$，$E=256$）。这将参数数量增加了 $O(E)$，同时将每个 token 的总激活参数数乘以 $k$，与稠密版本相比。

![](/scaling-book/assets/img/moe.png)

**图：**一个具有 $n$ 个专家的 MoE 层示例。门控专家将每个 token 路由到其中的 $k$ 个，并且这 $k$ 个 MLP 的输出被求和。我们的参数数量是每个专家大小的 $n$ 倍，但每个 token 只使用 $k$ 个。[来源](https://deepgram.com/learn/mixture-of-experts-ml-model-guide)。

与稠密模型相比，MoE 引入了新的通信，主要是两个 AllToAll（一个在 MoE 块之前，一个在之后），将 token 路由到正确的专家并将它们带回其归属设备。从技术上讲，这只发生在我们沿与专家相同的轴进行数据或序列分片时。然而，正如我们在前一节中所看到的，每个 AllToAll 的成本仅为沿单轴上可比较的 AllGather 的 1/4（对于双向环）。

### 梯度检查点（Gradient checkpointing） {#gradient-checkpointing}

反向传播作为一种算法，是用内存换计算。**反向传递不需要 $O(n_\text{layers}^2)$ FLOPs，而是需要 $O(n_\text{layers})$ 内存**，保存前向传递期间生成的所有中间激活值。虽然这比二次方计算要好，但它在内存方面非常昂贵：一个 $B * T=4M$（每批 4M 个总 token）、L=64、D=8192 的模型，避免所有不必要的反向传递计算，必须在 bfloat16 中保存大约 $2 * 20 * B * T * D * L = 84TB$ 的激活值。20 来自（粗略地）计算上面 Transformer 图中的每个中间节点，因为例如

$$f(x) = \exp(g(x))$$
$$\frac{df}{dx} = \exp(g(x)) \cdot \frac{dg}{dx}$$

所以为了避免重新计算，我们需要从前向传递中保存 $g(x)$ 和 $\exp(g(x))$。为了避免保存这么多内存，我们可以选择只保存中间激活值的一部分。这里有一些我们使用的策略。

- **块重计算（Block remat）**：只保存每层的输入。这是我们使用的最激进的方法，每层只保存 1 个检查点，意味着我们在上面的例子中只保存 4.2TB。这迫使我们在反向传递中基本上重复所有前向传递的 FLOPs，意味着我们将 FLOPs 从 $6ND$ 增加到大约 $8ND$。
- **仅大型 matmul：**另一个简单的策略是只保存大型 matmul 的输出。这让我们避免在反向传递期间重新计算任何大型 matmul，但仍然让我们重新计算其他激活函数和注意力的部分。这将每层的 20 减少到接近每层 7。

这绝不是全面的。当使用 JAX 时，这些通常由 `jax.remat`/`jax.checkpoint` 控制（你可以在[这里](https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html)阅读更多内容）。

### 键值缓存（Key-Value (KV) caching） {#key-value-kv-caching}

正如我们将在[第 7 节](../part7_inference)中看到的，LLM 推理有两个关键部分：预填充（prefill）和生成（generation）。

- **预填充**处理一个长提示并将其注意力激活值保存在键值缓存（Key-Value Cache，KV Cache）中以供生成使用，特别是注意力块中的键值投影。
- **生成**将其中几个 KV 缓存批处理在一起，并从每个缓存中采样 token。

每个 KV 缓存实际上是一个大小为 $[2, S, L, K, H]$ 的数组，其中 2 表示键和值。这相当大！int8 中键值缓存的总大小是 $2SLKH$。对于一个中等大小的模型，8k 上下文长度、64 层、$KH = NH = D = 8192$，这是 $2 \cdot 8192 \cdot 64 \cdot 8192 = 8\text{GiB}$。你可以看到为什么我们想要使用 $K \ll N$ 的 GMQA。

## 本节要点（What Should You Take Away from this Section?） {#what-should-you-take-away-from-this-section}

- Transformer 的总参数和 FLOPs 相当容易计算，这里总结一下，假设 MHA（批大小 B、词汇大小 V、长度为 T 的序列、D=dmodel、F=dff）：

| 组件 | 每层参数 | 每层训练 FLOPs |
|-----------|-----------------|-------------------------|
| **MLP** | 3DF | 18BTDF |
| **Attention** | 4DNH | 24BTDNH + 12BT²NH |
| **Other** | D | BTD |
| **Vocab** | DV (总数，非每层) | 12BTDV |

- MLP 块的参数数量主导了总参数数量，并且只要序列长度 $T < 8D$，MLP 块也主导了 FLOPs 预算。
- 对于合理的上下文长度，训练期间的总 FLOPs 预算可以很好地近似为 $6 \cdot \text{num\_params} \cdot \text{num\_tokens}$。
- 在推理期间，我们的 KV 缓存大约是每个缓存 $2 \cdot S \cdot L \cdot K \cdot H$（其中 K 是 KV 头的数量），尽管架构修改通常可以减少这个值。

## 一些练习题（A Few Problems to Work） {#a-few-problems-to-work}

**问题 1：**一个 $D=4096$、$F=4 \cdot D$、$V=32,000$、$L=64$ 的模型有多少参数？其中有多少比例是注意力参数？我们每个 token 的 KV 缓存有多大？*你可以假设 $N\cdot H=D$ 并使用 int8 KVs 的多头注意力。*

<details>
<summary>点击查看答案。</summary>

1. 总参数大约为 $L \cdot (3DF + 4DNH + D) + 2DV$。对于给定的数字，这是 $64 \cdot (3 \cdot 4e3 \cdot 16e3 + 4 \cdot 4e3 \cdot 4e3 + 4e3) + 2 \cdot 4e3 \cdot 32e3 = 16e9$，即 16B 参数。
2. 注意力参数与总参数的比率一般为 $4DNH / (4DNH + 3DF) = 4D^2 / (4D^2 + 12D^2) = 1/4$。这给我们大约 1/4 的参数用于注意力。
3. 每个 token，我们的 KV 缓存为 $2 \cdot L \cdot N \cdot H = 2 \cdot 64 \cdot 4096$，int8 中是 `512kB / token`。
</details>

**问题 2：**在 `{'X': 4, 'Y': 8, 'Z': 4}` 上执行 A[BX, DY] *D W[DY, F] 需要多少总 FLOPs？每个 TPU 执行多少 FLOPs？

<details>
<summary>点击查看答案。</summary>

操作的总"理论"FLOPs 是 $2 \cdot B \cdot D \cdot F$。但是，因为计算没有在 Z 维度上分片，我们实际上做了 Z 倍的额外 FLOPs，意味着总共 $2 \cdot B \cdot D \cdot F \cdot Z$ FLOPs。由于计算在其他维度上分片，每设备的总数大约为 $2 \cdot B \cdot D \cdot F / (X \cdot Y)$。
</details>

**问题 3：**执行 $A[I,J,K,L] * B[I,J,M,N,O] \rightarrow C[K,L,M,N,O]$ 涉及多少 FLOPs？

<details>
<summary>点击查看答案。</summary>

按照上面的规则，我们有 I 和 J 作为收缩维度，K、L、M、N 和 O 作为非收缩维度。我们没有"批处理维度"，所以这只是 $2 \cdot I \cdot J \cdot K \cdot L \cdot M \cdot N \cdot O$，即所有轴的乘积。如果我们有一个共享轴，它只会被计算一次。
</details>

**问题 4：**自注意力（忽略 Q/K/V/O 投影）的算术强度是多少？*将答案作为 Q 和 KV 长度 T 和 S 的函数给出。*在什么上下文长度下注意力是 FLOPs 受限的？给定我们 TPU 的 HBM 带宽，绘制注意力相对于 FFW 块的有效相对成本随上下文长度增长的图。

<details>
<summary>点击查看答案。</summary>

自注意力需要加载 $Q$、$K$ 和 $V$ 激活值，然后计算 $\text{softmax}(Q \cdot K) \cdot V$，然后将结果写回 HBM。这将使用 Flash Attention 完成，所以这个数学有一些注意事项，但基本上在 bf16 中自注意力执行

$$\text{Q[B,T,N,H]} \rightarrow_\text{reshape} \text{Q[B, T, K, G, H]} \cdot \text{K[B, S, K, H]} \rightarrow \text{O[B, T, S, K, G]}$$
$$U=\text{softmax}_S(\text{O[B, T, S, K, G]})$$
$$\text{U[B, T, S, K, G]} \cdot \text{V[B, S, K, H]} \rightarrow \text{X[B, T, K, G, H]}$$

所以我们的总字节是 $2 * \text{sizeof}(Q) + 2 * \text{sizeof(K or V)} = 4BTNH + 4BSKH = 4BHK * (TG + S)$，总 FLOPs 是 $4BTSNH + O(BTSN)$，算术强度是 $4BTSKGH / (4BHK * (TG + S))$。

所以基本上，在预填充期间我们有 $S=T$，所以算术强度为 $4BT^2KGH / 4BHKT \cdot (G+1) = TG/(G + 1) = O(T)$。在生成期间，$T=1$，所以我们有 $4BSKGH / (4BHK \cdot (G + S)) = SG / (G + S) \rightarrow G$，假设 $S$ 非常大。根据你如何解读问题，在预填充或训练期间，假设没有序列分片，自注意力在 S=240 时是计算受限的。在生成期间，我们从未达到计算受限，因为 $G$ 很小。尽管如此，你可以看到增加 $G$ 会让我们更接近计算受限。
</details>

**问题 5：**在什么序列长度下自注意力 FLOPs 等于 QKVO 投影 FLOPs？

<details>
<summary>点击查看答案。</summary>

这纯粹是一个 $24BTDNH = 12BT^2NH$ 何时成立的问题。简化得到 $2D = T$，所以例如对于 $D=4096$，这是 $8192$。这告诉我们对于大多数合理的上下文长度，matmul FLOPs 更大。
</details>

**问题 6：**假设我们在前向传递期间只保存 Transformer 层中 7 个主要 matmul 的输出（Q、K、V、O + 三个 FFW 矩阵）。我们需要在反向传递期间"重新物化"多少额外 FLOPs？

<details>
<summary>点击查看答案。</summary>

只保存七个 matmul 输出（Q、K、V、O、W₁、W₂、W₃）意味着反向传递必须重新计算两个注意力 matmul

$$QK^{\top} \quad\text{和}\quad \operatorname{softmax}(QK^{\top})V.$$

以获得 $\frac{\partial L}{\partial W_\text{O}}$。

每个都是在 $B$ 个序列和 $N$ 个头上批处理的 $T \times T$ matmul，所以额外的 FLOPs 是

$$4 \; B \, T^{2} \, N \, H.$$

其他重新计算的操作是：

1. $O(BTD)$ 用于 $\frac{\partial L}{\partial W_\text{In1}}$ 和 $\frac{\partial L}{\partial W_\text{In2}}$。
2. $O(BTF)$ 用于 $\frac{\partial L}{\partial W_\text{Out}}$。
</details>

**问题 7：**DeepSeek v3 表示它在 14.8T tokens 上训练了 2.79M H800 小时（[来源](https://arxiv.org/pdf/2412.19437v1)）。鉴于它有 37B 激活参数，他们大致达到了什么硬件利用率？*提示：注意他们使用了 FP8 FLOPs 而没有结构化稀疏性。*

<details>
<summary>点击查看答案。</summary>

从[这里](https://lenovopress.lenovo.com/lp1814.pdf)的规格表中，我们发现带稀疏性的 FP8 性能为 3,026 TFLOPs/s，或没有稀疏性时通常是这个的一半（`1.513e15` FLOPs/s）。2.79M H800 小时意味着 `2.79e6 * 1.513e15 * 60 * 60 = 1.52e25` 总 FLOPs。鉴于 37B 的激活参数数量，这次训练应该使用了大约 `6 * 37e9 * 14.8e12 = 3.3e24` FLOPs。这意味着 FLOPs 利用率约为 `3.3e24 / 1.52e25 = 21.7%`。
</details>

**问题 8：**混合专家（MoE）模型有 $E$ 个标准稠密 MLP 块的副本，每个 token 激活其中 $k$ 个专家。在 TPU v5e 上以 int8 权重运行 MoE 时，达到计算受限需要多大的批大小（按 token 数）？对于具有 256 个（路由）专家和 $k=8$ 的 DeepSeek，这个数字是多少？

<details>
<summary>点击查看答案。</summary>

因为我们有 $E$ 个每个专家的副本，在 int8 中，对于每个权重矩阵我们需要加载 $E \cdot D \cdot F$ 字节。因为每个 token 激活 $k$ 个专家，对于每个权重矩阵我们有 $2\cdot k \cdot B \cdot D \cdot F$ FLOPs。要在 bfloat16 FLOPs 下达到计算受限，我们需要算术强度超过 240，这在 $(2\cdot k \cdot BDF) / EDF > 240$ 或 $k \cdot B / E > 120$ 时发生。

因此，我们需要 $B > 120 \cdot E / k$ 才能达到计算受限。对于 DeepSeek，这给我们 $B > 120 \cdot 256 / 8 = 3840$。这在生成时是一个非常大的批大小。
</details>

**第 4 部分到此结束！要查看第 5 部分（关于扩展 Transformer 训练），[点击这里](../part5_training)！**

## 附录（Appendix） {#appendix}

### 附录 A：Flash Attention 如何工作？（Appendix A: How does Flash Attention work?） {#appendix-a-how-does-flash-attention-work}

将 Transformer 扩展到非常长的上下文的传统反对意见是，注意力 FLOPs 和内存使用随上下文长度二次方增长。虽然注意力 QK 乘积具有形状 $[B, T, S, N]$（其中 B 是批大小，S 和 T 是 Q 和 K 序列维度，N 是头数）确实是真的，但这种说法附带一些严重的注意事项：

1. 正如我们前面所指出的，即使这是二次方的，注意力 FLOPs 也只在 $S > 8 \cdot D$ 时才占主导地位，特别是在训练期间，与所有权重和激活检查点相比，单个注意力矩阵的内存很小，特别是当分片时。
2. 我们不需要物化完整的注意力矩阵来计算注意力！我们可以计算局部和与最大值，并避免物化数组中超过一小块的内容。虽然总 FLOPs 仍然是二次方的，但我们大大减少了内存压力。

第二个观察首先由 [Rabe et al. 2021](https://arxiv.org/abs/2112.05682) 提出，后来出现在 [Flash Attention 论文](https://arxiv.org/abs/2205.14135)（Dao et al. 2022）中。基本思想是分块计算 K/V 的注意力，我们计算局部 softmax 和一些辅助统计数据，然后将它们传递给下一个块，下一个块将它们与其本地块组合。具体来说，我们计算

1. **M：**$q \cdot k$ 在序列维度上的运行最大值
2. **O：**在序列维度上的运行完整注意力 softmax
3. **L：**运行分母 $\sum_i \exp(q \cdot k_i - \text{running max})$

有了这些，我们可以仅用恒定数量的内存计算新的最大值、新的运行总和和新的输出。为了粗略地描述这是如何工作的，注意力大致是这个操作：

$$\text{Attn}(Q, K, V) = \sum_i \frac{\exp(Q \cdot K_i - \max_j Q \cdot K_j) V_i}{\sum_l \exp(Q \cdot K_l - \max_j Q \cdot K_j)}$$

减去最大值是为了数值稳定性，可以在不影响结果的情况下添加，因为 $\sum_i \exp(a_i + b) = \exp(b) \sum \exp(a)$。仅查看上面的分母，如果我们想象有两个连续的键向量块 $K^1$ 和 $K^2$，并且我们计算每个的局部 softmax 和 $L^1$ 和 $L^2$

$$L^1 = \sum_i \exp(Q \cdot K_i^1 - \max_j Q \cdot K_j^1)$$
$$L^2 = \sum_i \exp(Q \cdot K_i^2 - \max_j Q \cdot K_j^2)$$

那么我们可以使用以下方式将这些组合成两个块一起的完整 softmax 和

$$L^\text{combined} = \exp(M^1 - \max(M^1, M^2)) \cdot L^1 + \exp(M^2 - \max(M^1, M^2)) \cdot L^2$$

其中

$$M^1 = \max_j Q \cdot K_j^1 \text{ 和 } M^2 = \max_j Q \cdot K_j^2$$

这也可以为完整的 softmax 完成，给我们一种累积任意大 softmax 和的方式。这是 Flash Attention 论文中的完整算法。

![](/scaling-book/assets/img/flash-algo.png)

从硬件的角度来看，这让我们将 Q 块放入 VMEM（上面算法称之为片上 SRAM），所以我们在每次迭代中只需加载 KV 块，增加了算术强度。我们还可以将运行统计数据保留在 VMEM 中。

最后一个值得强调的微妙点是注意力 softmax 属性，它用于使 Flash VJP（反向模式导数）计算对训练实用。如果我们将中间 softmax 数组定义为：

$$S_{ij} = \frac{e^{\tau q_i \cdot k_j}}{\sum_l e^{\tau q_i \cdot k_l}}$$

在注意力中，我们从反向模式 *dO* 和 *V* 数组获得 *dS*：

$$dS_{ij} = dO_{id} \cdot_d V_{jd} = \sum_d dO_{id} V_{jd}$$

在此梯度反向传播到 Q 和 K 期间

$$d(q_i \cdot k_j) = (dS_{ij} - S_{ij} \cdot_j dS_{ij}) S_{ij}$$

我们利用一个恒等式，它允许我们将沿大键**长度**维度的收缩交换为沿特征**深度**维度的局部收缩。

$$\begin{align*} S_{ij} \cdot_j dS_{ij} &= \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} \sum_d dO_{id} V_{jd} \\ &= \sum_d dO_{id} \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} V_{jd} \\ &= \sum_d dO_{id} O_{id} \\ &= dO_{id} \cdot_d O_{id} \end{align*}$$

这种替换对于能够实现 VJP 的序列块*局部*计算至关重要，并实现了进一步巧妙的分片方案，如环形注意力（ring attention）。

### 引用

对于学术上下文中的归属，请将这项工作引用为：

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
