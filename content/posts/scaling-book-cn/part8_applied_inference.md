---
title: "Serving LLaMA（部署 LLaMA）"
date: 2026-04-29
draft: false
math: true
weight: 9
---

{{< katex >}}

# 在 TPU 上部署 LLaMA 3-70B（Serving LLaMA 3-70B on TPUs）

《How To Scale Your Model》第 8 部分（[第 7 部分：Inference](../part7_inference) | [第 9 部分：Profiling](../part9_profiling)）

让我们仔细看看如何在 TPU v5e 上部署 LLaMA 3-70B 模型。在 roofline 下，部署不同模型的代价有多大？它们的 KV 缓存有多大？我们应该使用多大的 batch size？参数和激活在推理过程中是如何分片（sharded）的？让我们对生产环境中的延迟（latency）和吞吐（throughput）做一些大致的估算。

### 内容

- [LLaMA 的部署故事是什么？（What's the LLaMA Serving Story?）](#whats-the-llama-serving-story)
  - [关于吞吐的思考（Thinking about throughput）](#thinking-about-throughput)
  - [Prefill 怎么办？（What about prefill?）](#what-about-prefill)
- [可视化延迟与吞吐的权衡（Visualizing the Latency Throughput Tradeoff）](#visualizing-the-latency-throughput-tradeoff)
- [习题（Worked Problems）](#worked-problems)

*本节将探讨部署 LLaMA-3 需要付出什么，以及它能达到多高的效率。和上一节"应用"部分一样，建议你先用纸笔自己算一算答案再看下文！*

## LLaMA 的部署故事是什么？（What's the LLaMA Serving Story?）

让我们先回顾一下 LLaMA 3-70B 的架构（参考[第 6 节](../part6_applied_training)）：

| **超参数（hyperparam）** | **取值（value）** |
|---|---|
| $n_\text{layers}$ (L) | 80 |
| $d_\text{model}$ (D) | 8,192 |
| $d_{ff}$ (F) | 28,672 |
| $n_\text{heads}$ (N) | 64 |
| $n_\text{kv heads}$ (K) | 8 |
| $d_\text{qkv}$ (H) | 128 |
| $n_\text{embeddings}$ (V) | 128,256 |

让我们从一个简单的问题开始：**我们应该在哪种硬件上部署？** 答案基本上是：哪种最便宜（按 FLOPs / 美元算）就用哪种。出于这个原因，我们通常希望部署在 TPU v5e 上——这是我们当前专用的推理芯片（成本来自 [Google Cloud 定价](https://cloud.google.com/tpu/pricing)，截至 2025 年 2 月）：

| **TPU 类型** | **bfloat16 FLOPs/s** | **Google Cloud USD / 小时** | **FLOPs / $** |
|---|---|---|---|
| H100 | 9.9e14 | $10.8 | 3.3e17 |
| v5p | 4.59e14 | $4.2 | 3.9e17 |
| v5e | 1.97e14 | $1.2 | **5.8e17** |

每块 TPU v5e 有 16GB HBM，这要求我们必须相当激进地对模型分片。让我们先思考几个对我们重要的基本量：

**问题：** LLaMA 3-70B 每个 token 的 KV 缓存有多大？*假设我们用 int8 存储。这决定了在给定拓扑（topology）下我们能用多大的 batch size。*

想清楚后点击查看！

LLaMA 3-70B 有 8 个 KV head，所以每个 token 的大小是 `2 * K * H * L = 2 * 8 * 128 * 80 = 160kB`。

**注意它有多大！** 如果我们的序列长度是 32k tokens（这在生产中很常见），单个序列就要用 `160e3 * 32,768 = 5.3GB`。在 BS=240 的情况下，这就是 1.3TB！由于 TPU v5e 每块只有 16GB，我们需要约 `(70e9 + 1.3e12) / 16e9 = 86` 块 TPU v5e 芯片才能容纳这么多内存。同时注意，这相对于 70GB 的模型参数来说已经非常大了。

**问题：** 假设我们想以 batch size 32、序列长度 8192 部署 L3 70B，所有内容（参数和 KV）都用 int8。总共会用多少内存？我们能部署的最小 slice 是多大？

答案

由于我们的 KV 在 int8 下是 `160e3` 字节，我们的 KV 总内存是 `160e3 * 8192 * 32 = 41.9e9` 字节。我们的参数是 `70e9` 字节，因为每个参数 1 字节。因此总内存使用量是 `41.9e9 + 70e9 = 112GB`。

我们能用的最小 slice 需要 `112e9 / 16e9 = 7` 块 TPU，或者（向上取整为偶数尺寸）TPU v5e `4x2`。这会非常紧张，考虑到其它额外开销可能装不下，所以我们可能至少需要 `4x4`（或者降低 batch size）。

**问题：** 在 TPU v5e `4x2` 上，使用上述 batch size 和量化方式，每个 decode 步大致延迟是多少？吞吐（tokens / sec / chip）是多少？`4x4` 呢？*假设我们用 bfloat16 执行 FLOPs，并且全部完全分片。*

答案

我们可以套用上一节给出的公式：

$$\begin{align*} \tiny \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\tiny \text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\tiny \text{MLP (can be compute-bound)}} \end{align*}$$

这里我们的临界 batch size（critical batch size）大约是 120，因为参数是 int8 而 FLOPs 是 bfloat16。我们也可以手动计算右侧的 max，但那基本上是我们已经做过几次的计算了。**所以无论是 matmul 还是 FLOPs，我们都明显处于受内存带宽限制（memory-bound）的区域。**

如果只看内存带宽，我们的步时基本上是 `(KV size + param size) / (8 * HBM bandwidth) = 112e9 / (8 * 8.1e11) = 17ms`。**所以理论上每步约 17ms。** 吞吐为 `32 / .017 = 1882 tokens / sec`，或 `1882 / 8 = 235 tokens / sec / chip`。

这里有一个需要注意的地方：我们要检查 matmul 是否会受 ICI 限制。这里我们可以为它分配 2 个轴，所以理论上当 $Y > 2 * F / 2200 = 2 * 28672 / 2200 = 26$ 时才会受 ICI 限制，所以我们安全！

如果在 `4x4` 上运行，我们仍然不会受 ICI 限制，所以延迟会下降到 `17 / 2 = 8.5ms`，但每芯片的吞吐保持不变。

### 关于吞吐的思考（Thinking about throughput）

让我们花点时间纯粹思考一下吞吐。当我们优化吞吐时，希望处于受计算限制（compute bound）的状态，也就是说要尽量充分利用 TPU MXU 的容量。通常这意味着我们希望 batch size 尽可能大，以便完成尽可能多的工作。

**问题：** 在 TPU v5e 上，用 bfloat16 权重和激活，要多大的 batch size 才能在 matmul 上受计算限制？如果用 int8 权重但用 bfloat16 执行 FLOPs 呢？int8 权重 + int8 FLOPs 呢？

答案

如第 7 节所讨论，对于任何 bfloat16 matmul，当 $B \ll D, F$ 时我们有：

$$\begin{equation*} T_\text{math} > T_\text{comms} \leftrightarrow \frac{2BDF}{2DF} \geq \frac{\text{TPU bfloat16 FLOPs/s}}{\text{HBM bandwidth}} = 240 \end{equation*}$$

当权重是 int8 时，分母少了一个因子 2，所以我们有 $2BDF / DF = 2B > 240$，等价于 $B > 120$，是之前临界 batch size 的一半。这对我们非常有帮助！当我们用 int8 权重和 int8 FLOPs 时，必须使用 int8 的 TPU FLOPs/s 值，从 bfloat16 的 1.97e14 增加到 3.94e14，几乎翻倍。所以又回到了 $B > 240$ 的起点。

int8 权重 + bfloat16 FLOPs 这种情况非常常见，因为对参数做无损量化通常比做低精度算术更容易。

**问题：** 在 8k context 下，能部署 LLaMA 3-70B 的最小 TPU v5e 拓扑是多少？分别考虑 bfloat16、int8、int4（KV 和参数都用同样精度）。*这道题可以认为 KV 缓存可以忽略不计。*

答案

这很简单！如果我们能接受很小的 batch size，那么唯一的限制就是把参数放进 HBM，即 `ceil(num_params * sizeof(dtype) / HBM per TPU)`，或 `ceil(70e9 * sizeof(dtype) / 16e9)` 向上取整到最近的合理拓扑（2 的倍数）：

| dtype | 参数大小 | KV 大小 / token (bytes) | 最少 TPU v5e | 实际最小 slice | KV 缓存剩余 HBM | KV 缓存数量 @ 8k |
|---|---|---|---|---|---|---|
| bf16 | 140GB | 324kB | 8.75 | 4x4 = 16 chips | 116 | 43 |
| int8 | 70GB | 162kB | 4.38 | 4x2 = 8 chips | 58 | 43 |
| int4 | 35GB | 81kB | 2.81 | 2x2 = 4 chips | 29 | 43 |

很酷吧！它告诉我们如果想的话，可以把 LLaMA 70B 装进 TPU v5e 2x2。但你会注意到 KV 缓存数量非常少。这就是我们的 batch size！这意味着 FLOPs 利用率会很糟糕。我们会很乐意使用更大的拓扑以将 batch size 提升到 240。

**问题：** 假设我们使用这些拓扑下能容纳的最大 batch size，每个生成步预期的延迟是多少？

答案

这也很简单，因为我们选 batch size 把 HBM 装满！这就是问要把整个 TPU v5e 的字节加载到 MXU 需要多长时间。这就是 `v5e HBM / v5e HBM memory bandwidth = 16GB / 8.2e11 = 19ms`，所以是 **每步 19ms**。假设我们生成的中位长度是 512 tokens，那么每次解码大约 9 秒。注意我们用更小的 batch size 可以略微改善延迟，例如如果只看 int4 中的模型参数，最小延迟约为每步 10ms，因为 HBM 不再被装满。

**结论**：我们总是可以这样下界估算 decode 延迟——把所有模型参数从 HBM 加载到 MXU 需要多长时间。当 KV 缓存较小时，可以把每一层看作只是分块加载权重然后丢弃。除非使用大 batch size 或大量跨设备通信，否则这通常是个合理的下界（误差在 1.5x 以内）。当 batch size 较大时，我们也需要建模 KV 缓存的加载，因为它会主导参数。

类似地，在受 FLOPs 限制的情况下（例如训练或大 batch 推理），我们可以使用 $\text{Total FLOPs} / (N \cdot C) = 2 \cdot \text{param count} \cdot B / (N \cdot C)$ 这个下界，假设没有通信。

**问题：** 对于这些情形，每芯片的吞吐是多少（以 queries / chip 计）？*你可以假设解码中位长度是 512 tokens。*

答案

这是个重要问题，因为它正好与每 token 的成本相关。

按照我们对解码中位长度的假设，吞吐就是 $B / (\text{per-step latency} \cdot \text{median steps} \cdot N) \approx 43 / (0.019 * 512 * N)$。这给了我们大约 $(4.42 / N)$ QPS，所以代入 $N$ 得到：

| dtype | QPS / chip |
|---|---|
| bfloat16 | 0.27 |
| int8 | 0.55 |
| int4 | 1.11 |

注意这是相当乐观的，因为它完全忽略了前向传播的工作内存（分配给激活和 attention 的内存）。借助 Flash Attention 这并非荒谬，但也并不现实。真实数字大概是这个的一半左右。要追求绝对最大吞吐，我们大概要把芯片数量加倍以上，并显著增大 batch size。

**问题：** 如果我们把上述每个例子的拓扑加倍，峰值吞吐会如何变化？

答案

如果我们在 bfloat16 下用 4x8 slice，KV 缓存会有 372GB 剩余，这能让我们把 batch size 提升到 140。然后由于步时保持不变，吞吐会是 `14.39 / num_chips`，即：

| dtype | QPS / chip |
|---|---|
| bfloat16 (on 4x8) | 0.44 |
| int8 (on 4x4) | 0.90 |
| int4 (on 2x4) | 1.80 |

进一步增加会带来更大的收益！主要的结论是：**当我们受 KV 缓存大小限制时，最小拓扑并不总是性能最好的拓扑。**

**问题：** 现在我们深入讨论分片问题。假设我们想在 TPU v5e 4x8 上以 bfloat16 部署。我们在生成时会用什么分片方式？能避免受通信限制吗？

答案

如上一节所讨论，在生成时我们其实只有一个分片选项：模型并行（model parallelism）。我们能做多少而不变成通信受限？正如上一节所讨论的，我们的模型大致在以下条件下变得通信受限：

$$Y > \frac{F \cdot M_Y}{2200}$$

对 LLaMA 3-70B 我们有 `F = 28,672`，所以如果做 2 个轴的模型分片，这给出 $Y = 28672 \cdot 2 / 2200 = 26$，所以一般我们最多可以扩展到约 16 个芯片而不变成通信受限，这允许我们用 `4x4` 但不能用 `4x8`。一般来说，因为我们无法完美地把计算和通信重叠，所以即便这个估计也偏乐观。

**结论：在 4x8 上我们其实无法用纯模型并行部署。** 这里我们最多能做到 4x2 或*也许* 4x4。

但是，如我们讨论的，当 batch size 较小时我们经常可以做更多模型并行而不显著影响吞吐，因为模型受内存带宽限制而非 FLOPs 限制。我们之前说过这个值大约是 $Y=F / (8\cdot B)$，所以如果 batch size 为 64，理论上我们可以做到 `Y = 28,672 / (8 * 64) = 56` 路模型并行才会变成 ICI 受限。为了校验这点，我们可以看单个 matmul 的 $T_\text{ici comms}$、$T_\text{hbm comms}$ 和 $T_\text{math}$。我们清楚地有：

$$\begin{align*}T_\text{ici comms} = \frac{2BD}{W_\text{ici}} && T_\text{hbm comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{math} = \frac{2BDF}{Y \cdot C}\end{align*}$$

对于 `4x8`，这给出 $T_\text{ici comms}$ = `(2 * 64 * 8192) / 9e10 = 11us`，$T_\text{hbm comms}$ = `(2 * 8192 * 28,672) / (32 * 8.1e11) = 18us`，以及 $T_\text{math}$ = `(2 * 64 * 8192 * 28,672) / (32 * 1.97e14) = 4us`，所以理论上我们仍然受 HBM 带宽限制，这很好！*注意从 `4x4` 扩展到 `4x8` 从吞吐角度大概没什么用，但能降低延迟！*

如果看 int8 和 int4 配置，那些*确实可以*用纯模型并行做。所以我们到达了一个量化能带来显著优势的临界点（除了更快的 FLOPs 外）：它让我们可以在变成通信受限之前使用更大的 batch size。**所以这个故事的结尾是：我们无法在 4x8 上达到峰值吞吐，但对 int8 和 int4 配置可以做纯模型并行。**

**提示**：模型并行的最大有效程度取决于 $d_{ff}$ 以及模型分片的轴数。最大值通常在 8 到 32 之间，取决于模型大小。你可以超过这个限制以改善延迟，代价是部分吞吐。

### Prefill 怎么办？（What about prefill?）

我们这里基本上忽略了 prefill，因为它简单得多。让我们把几个概念合在一起，思考端到端的全局图景。

**问题：** 假设我们在 prefill 时达到 40% 的 FLOPs 利用率。在 16 个 TPU v5e 芯片上，长度 8192 的 prefill 需要多久？

答案

在 8k tokens 时，我们已经稳稳处于受计算限制的状态，所以只需要推理 FLOPs。我们知道模型有 `70e9` 参数，所以每次前向传播使用 `2 * 70e9 * B` FLOPs。假设 40% MFU（FLOPs 利用率），运行时间约为 `2 * 70e9 * 8192 / (16 * 1.97e14 * 0.4) = 0.91s`。相比之前看到的数字，这其实相当多！

**问题：** 假设我们 prefill 中位长度是 8192 tokens，decode 中位长度是 4096 tokens。假设 generate batch size 为 32。平均每步有多少序列完成解码？平均每步从 KV 缓存中淘汰多少 tokens？

答案

这相当直接。由于 decode 中位长度是 4096 tokens，一个序列大约每 1 / 4096 个 tokens 完成一次。给定 batch size 32，意味着每步淘汰 `32 / 4096` 个序列。由于 KV 缓存长度大约是 `8192 + 4096`，所以是 `32 * (8192 + 4096) / 4096 = 96` 个 tokens 每步被淘汰。一般公式是 $B * (P + G) / G$，其中 $P$ 和 $G$ 是 prefill 和 generate 长度。

**问题：** 假设我们做分离式部署（disaggregated serving），prefill 中位长度 8192，decode 中位长度 512。假设 prefill 和 generate 延迟如上所述（bfloat16）。需要 prefill 与 generate 服务器多少比例才能保持两边都满载？

答案

这是个有趣的问题。设 $P$ 为 prefill 服务器数量，$G$ 为 generate 服务器数量。一般来说，这是一个流水线问题：我们以 `P / prefill_latency` 的速率进给序列，以 `B * G / (generate_latency * median_decode_length)` 的速率消费它们。我们之前算过 `910ms` 一次 prefill 步、batch size 43（就当作 32）下 `19ms` 一次 decode 步。因此我们需要 `P / 0.91 = 32 * G / (0.019 * 512)` 或 `P = 3G`，即我们需要的 prefill 服务器是 generate 服务器的约 3 倍！

## 可视化延迟与吞吐的权衡（Visualizing the Latency Throughput Tradeoff）

继续以 LLaMA 70B 为例，让我们实际看一下生成时不同 batch size 下的延迟与吞吐。如上一节为 PaLM 模型展示的那样，这给我们一个吞吐/延迟的 Pareto 前沿。我们假设 16 路张量并行（tensor parallelism），因为这是在 MLP 块中保持受计算限制时一个合理的上界。这里我们使用 TPU v5e 4x4 拓扑。**滑块控制序列长度，可以看到更大 KV 缓存的影响。**

*   **看看成本和延迟之间的权衡有多戏剧性。** 以每 token 延迟翻倍为代价，我们能实现每 token 成本约 100x 的下降。同时，延迟可以从小 batch size 时的 5.5ms 一直到非常大 batch 时的 20ms。
*   注意在 2k context 下，吞吐基本上在大约每芯片 1 token/ms 处趋于平稳，命中 BS 120 的 roofline（这里是 120 是因为我们用 int8 权重但 bf16 FLOPs）。但是当序列长度增加时，我们再也无法把这种 batch size 装入内存，所以达不到完全饱和点。
*   注意在大 batch size 同等吞吐时延迟有多高，因为 KV 加载（而非参数加载）变成了主导。

我们可以通过把成本和延迟的来源分解为参数加载时间、KV 加载时间和 FLOPs 时间来更好地理解这一点。红色区域是我们预期在 MLP 块中受计算限制的区域。

这讲述了一个完整的故事。你可以看到，最初参数加载占据了延迟的绝大部分，直到 batch size 大到足以让 FLOPs 和 KV 加载变得更重要。值得注意的是，在所有大于 2048 的序列长度下，我们花在 KV 缓存加载上的时间都比 FLOPs 多！**所以虽然我们可以通过增加 batch size 来提高硬件利用率，但在长 context 下 KV 加载始终主导总步时。**

**结论：** 对 LLaMA 3-70B，在几乎所有这些配置中我们都强烈受 KV 缓存内存带宽限制（也即 HBM 受限），凸显了在生成吞吐中减小 KV 缓存大小的重要性。同时注意到延迟/吞吐的权衡仍然是多么戏剧性。

实现这一切的代码相当简单。

下面是计算这些 roofline 的代码：

```python
import numpy as np

num_chips = 16  # we fix 16 as the amount of total model parallelism we do
bytes_per_param = 1  # int8 means 1 byte per param
param_count = 70e9
param_size = bytes_per_param * param_count
sequence_length = 8192  # can vary this

hbm_bandwidth = 8.20E+11  # v5e
flops = 1.97E+14  # v5e

def kv_cache_size(bs):
    return 2 * bs * 128 * 8 * 80

def min_topology(bytes):
    return 2 ** np.ceil(np.log2(bytes / 16e9))

def get_max_batch_size(
    num_chips: int,
    sequence_length: int,
    param_size: float,
) -> int:
  batch_sizes = np.arange(1, 1024, 4)
  kv_sizes = kv_cache_size(sequence_length * batch_sizes)
  required_chips = min_topology(kv_sizes + param_size)
  max_idx = np.where(required_chips <= num_chips)[0][-1]
  return max_idx

max_idx = get_max_batch_size(
    num_chips=num_chips,
    sequence_length=sequence_length,
    param_size=param_size,
)  # get the largest batch size that can fit
batch_sizes = np.arange(1, 512, 1)[:max_idx]
kv_sizes = kv_cache_size(sequence_length * batch_sizes)

kv_comms_time = kv_sizes / (num_chips * hbm_bandwidth)

param_comms_time = param_size / (num_chips * hbm_bandwidth)
param_comms_time = np.asarray([param_comms_time] * batch_sizes.shape[0])

flops_time = 2 * param_size * batch_sizes / (num_chips * flops)  # roughly true in a 2ND sense

mlp_time = np.maximum(flops_time, param_comms_time)
attn_time = kv_comms_time  # always bandwidth-bound for generate

latency = 1000 * (mlp_time + attn_time)
throughput = batch_sizes / (latency * num_chips)
```

注意我们如何非常显式地把延迟拆成两个来源：KV 加载和参数加载，以及延迟如何被 FLOPs 或通信中较大的那个限制。

## 习题（Worked Problems）

下面是几道习题。其中一些在前文已涉及，但可能在教学上有用。

**问题 1：** LLaMA 3-405B 每个 token 的前向传播使用多少 FLOPs？假设我们受 FLOPs 限制，在 N 块 TPU v5e 上一次前向传播的下界是什么？如果是受通信限制呢？*忽略模型放不进单芯片的事实。*

**问题 2：** 假设我们想以 BS240 部署 LLaMA 3-8B，使用 int8 权重和 int8 KV 缓存。下列各项各占多少字节：(a) 模型参数 (b) KV 缓存 (c) 峰值工作激活（大致）？我们能运行这个的最小拓扑是什么？

**问题 3：** 你会如何在 TPU v5e 上部署 LLaMA 3-405B？假设 int8 权重和 bfloat16 FLOPs。假设我们有一个硬限制：每 token 15ms，我们能实现的最高吞吐配置是什么？理论最小步时是多少？

**第 8 部分到此结束！第 9 部分深入讲解 XLA 和 TPU profiling，请[点击这里](../part9_profiling)。**
