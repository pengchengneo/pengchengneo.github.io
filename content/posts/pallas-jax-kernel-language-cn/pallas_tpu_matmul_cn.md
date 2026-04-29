---
title: "矩阵乘法"
date: 2026-04-19
draft: false
weight: 6
---

# 矩阵乘法

在本指南中，我们将使用 Pallas 编写一个矩阵乘法例程。我们还将介绍如何思考 TPU 上的矩阵乘法性能，以及如何模板化矩阵乘法内核以融合操作。

```python
#@title 导入
import functools
from typing import Callable

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax import random
import jax.numpy as jnp
import numpy as np
```

## 背景

矩阵乘法是现代深度学习和语言建模核心的基本线性代数运算。我们希望使用 TPU 和 GPU 等专用加速器使矩阵乘法尽可能快，这些加速器都有用于快速矩阵乘法的专用单元。

为了有效利用 TPU 进行矩阵乘法，我们需要涵盖几个背景概念：分块矩阵乘法、分块（tiling）和流水线。

### 分块矩阵乘法

假设我们要实现 `matmul(x, y)`，它通用地将一个 `(m, k)` 数组与一个 `(k, n)` 数组相乘，但有一个限制：我们只能使用原语 `matmul_small`，它只能乘以小矩阵（比如 `m, k, n <= 256`）。我们该怎么做？

矩阵乘法的一个优良性质是输出的每个块都可以表示为输入的行块和列块的多个较小矩阵乘法之和。形式化地，如果我们有输入数组 $x \in \mathbb{R}^{m \times k}$ 和 $y \in \mathbb{R}^{k \times n}$，以及输出 $z \in \mathbb{R}^{m \times n}$，我们沿大小为 $b_m, b_k, b_n$ 的维度将它们分解为块。

例如，$x$ 可以分解为：

$$
\begin{bmatrix}
x_{0, 0} & \cdots & x_{0, i_k} \\
x_{1, 0} & \cdots & x_{1, i_k} \\
\vdots & \ddots & \vdots \\
x_{i_m, 0} & \cdots & x_{i_m, i_k} \\
\end{bmatrix}
$$

其中 $x_{ik} \in \mathbb{R}^{b_m \times b_k}$。（我们可以类似地分解 $y$ 和 $z$。）

对于特定的输出块 $z_{ij}$，我们可以将其计算为

$$z_{ij} = \sum_k x_{ik} y_{kj}$$

因此，每个输出块 $z_{ij}$ 是多个较小分块矩阵乘法 $x_{ik} y_{kj}$ 的和。以下是我们如何在 NumPy 中实现此算法：

```python
def matmul_small(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    m, k, n = x.shape[0], x.shape[1], y.shape[0]
    assert m <= 256
    assert k <= 256
    assert n <= 256
    return np.matmul(x, y)

def block_matmul(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bm: int = 256,
    bk: int = 256,
    bn: int = 256,
) -> np.ndarray:
    m, k = x.shape
    _, n = y.shape

    z = np.zeros((m, n), dtype=x.dtype)
    for m_i in range(m // bm):
        for n_i in range(n // bn):
            for k_i in range(k // bk):
                m_slice = slice(m_i * bm, (m_i + 1) * bm)
                k_slice = slice(k_i * bk, (k_i + 1) * bk)
                n_slice = slice(n_i * bn, (n_i + 1) * bn)
                x_block = x[m_slice, k_slice]
                y_block = y[k_slice, n_slice]
                z[m_slice, n_slice] += matmul_small(x_block, y_block)
    return z
```

我们的 `block_matmul` 函数现在应该可以处理大于 256 的输入了（尽管我们假设输入维度能被 256 整除）。

```python
m, k, n = 4096, 4096, 4096
x = np.random.uniform(size=(m, k)).astype(np.float32)
y = np.random.uniform(size=(k, n)).astype(np.float32)
np.testing.assert_allclose(x @ y, block_matmul(x, y), atol=1e-6, rtol=1e-6)
```

`block_matmul` 通过观察每个大小为 `(bm, bn)` 的输出块可以通过累加多个 `(bm, bk) x (bk, bn)` 大小的矩阵乘法来计算，从而将矩阵乘法分解为许多较小的矩阵乘法。

TPU 和 GPU 做矩阵乘法就是这样的！它们原生支持类似于 `matmul_small` 的小矩阵乘法，因此在进行更大的矩阵乘法时，我们将应用 `block_matmul` 分解来利用这种硬件。

### 分块和流水线

在 [上一篇指南](pipelining.html) 中，我们介绍了 Pallas 中分块计算和流水线的工作原理。为了确保计算单元始终在工作而不被内存传输阻塞，我们将下一次内核迭代的内存传输与当前迭代重叠。

在 Pallas 中，我们通过 `BlockSpec` 和 `grid` 来指定这一点。注意我们在分块矩阵乘法算法中已经有了嵌套的 for 循环。我们可以通过 `grid` 在 Pallas 中指定它。分块矩阵乘法中的切片也可以通过 `BlockSpec` 来指定。

## 你的第一个矩阵乘法内核

将以上所有内容结合在一起，这是一个分块矩阵乘法内核的实现，它将内存传输与计算进行流水线化。我们创建一个 3 维 grid，对应 NumPy 代码中的 3 层嵌套循环。注意虽然 MXU 只能乘以小块，但 Pallas 会自动将更大的块在 MXU 上自动分块。

grid 的最后一个维度对应矩阵乘法的收缩维度，是一个归约维度，因此我们需要确保初始化累加器。

```python
def matmul_kernel(x_ref, y_ref, z_ref):
    @pl.when(pl.program_id(2) == 0)
    def _():
        z_ref[...] = jnp.zeros_like(z_ref)

    z_ref[...] += x_ref[...] @ y_ref[...]

def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
):
    m, k = x.shape
    _, n = y.shape
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        in_specs=[pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                  pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
        grid=(m // bm, n // bn, k // bk),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
    )(x, y)
```

```python
m, k, n = 4096, 4096, 4096
k1, k2 = random.split(random.key(0), 2)
x = random.normal(k1, (m, k), dtype=jnp.float32)
y = random.normal(k2, (k, n), dtype=jnp.float32)
np.testing.assert_array_equal(x @ y, matmul(x, y))
```

## 矩阵乘法性能

让我们思考如何分析矩阵乘法性能。当我们思考矩阵乘法性能时，通常关心两件事：浮点运算总数（FLOPs）和内存带宽使用量。从 [TPU 和流水线指南](pipelining.html) 中，我们看到为了使用 TPU（以及一般的 ML 加速器）上的高效计算单元，我们需要将输入从 HBM 复制到更靠近计算单元的 VMEM 中。与 HBM 之间的复制需要时间，一个高效的内核理想情况下应该将大部分时间花在实际计算上，而不是等待这些传输。内存带宽衡量的是这种数据传输的速率。

> 快速说明：在本指南中，我们将讨论浮点运算，但要区分 FLOPs 和 FLOP/s。当我们说"FLOPs"时，我们指的是"浮点运算"，即运算的数量。当我们说"FLOP/s"时，我们指的是"每秒浮点运算"，即执行浮点运算的 _速率_。

一个 `(m, k) x (k, n)` 矩阵乘法的 FLOPs 数量（近似）为 `2 * m * k * n`。（技术上是 `n * m * (2k - 1)`，但对于足够大的 `k`，我们的近似是足够的。）

矩阵乘法的最小内存带宽使用量（假设 float32）是输入的总大小（复制到 VMEM）加上输出的大小（复制到 HBM）。因此最小带宽使用量为 `(m * k + k * n + m * n) * 4 字节/float32`。如果我们多次重读输入，内存使用量可能更大，这种情况经常发生。

一个观察是矩阵乘法的 FLOPs 相对于输入是立方增长的，而最小带宽使用量是二次增长的。直觉上，这意味着 FLOPs 增长速度比带宽使用量快，这意味着矩阵乘法越大，相对于复制我们有越多的计算。

```python
def matmul_flops(m: int, k: int, n: int):
    return 2 * m * k * n

def matmul_membw(m: int, k: int, n: int, dtype: jnp.dtype):
    return (m * k + k * n + m * n) * np.dtype(dtype).itemsize

print(matmul_flops(1024, 1024, 1024))
print(matmul_membw(1024, 1024, 1024, jnp.float32))
```

```
2147483648
12582912
```

现在我们可以计算矩阵乘法的 FLOPs 总数和（最小）内存带宽使用量了，让我们看看真正的 TPU 能处理什么。

本 notebook 在 TPU v5e 芯片上运行，所以我们使用 v5e 的数据（如果你正在运行此 notebook，你的数据可能不同）。TPU v5e 拥有 [197 TFLOP/s 的 bf16/f32 计算能力和 819 GB/s 的内存带宽](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu_v5e)。通过查看这些数字的比率（称为算术强度），我们可以得到在变为 IO 受限之前这个"FLOPs / 内存带宽使用量"比率可以有多低的边界（在 TPU v5e 上大约 240 FLOPs/字节）。

```python
v5e_flops = 197e12
v5e_membw = 819e9
v5e_op_intensity = v5e_flops / v5e_membw  # ~240.5
```

粗略地说，这些数字告诉我们矩阵乘法的 FLOPs 应该花费 `2 * m * k * n / (197 TFLOP/s)` 秒，到/从 VMEM 的复制应该花费 `(m*k + k*n + m*n) * 4 字节 / 819GB/s` 秒。

```python
def matmul_flops_intensity(m: int, k: int, n: int, dtype: jnp.dtype):
    flops = matmul_flops(m, k, n)
    membw = matmul_membw(m, k, n, dtype)
    return flops / membw
```

这个基本计算大致告诉我们能多高效地使用 MXU。如果我们的矩阵乘法运算强度低于芯片的能力，那么我们的计算将是 _内存受限_ 的，即计算单元将在等待值被传输时空闲。如果矩阵乘法强度高于芯片的能力，那么我们将是 _计算受限_ 的。

因为矩阵乘法的 FLOPs 相对于输入大小是立方增长的，而内存带宽使用量是二次增长的，我们预期随着规模越来越大将变为计算受限，但这个交叉点非常重要！假设我们正在进行 `(1024, 1024) x (1024, 1024)` 的 float32 矩阵乘法。

```python
print(f"{matmul_flops_intensity(1024, 1024, 1024, jnp.float32)} flops/byte")
```

```
170.66666666666666 flops/byte
```

我们的矩阵乘法运算强度低于芯片的能力。这不好！这种类型的矩阵乘法我们很可能是内存受限的。然而，如果我们的输入和输出更大呢？在某个时刻，当矩阵乘法变得足够大时，我们将从内存受限过渡到计算受限。例如，如果我们有一个 `m = k = n` 的矩阵乘法，我们将在（TPU v5e 上）当 `2m³ / 12m² > 240` 或 `m = k = n > 1440` 时过渡。

### `bfloat16` 矩阵乘法

为了使矩阵乘法在 TPU 上更容易成为计算受限的，我们还可以对输入和输出使用更小的数据类型。我们之前的示例使用了 `float32` 输入和输出，但 TPU v5e 也支持用于矩阵乘法的 `bfloat16` 数据类型（一种 16 位浮点格式，也称为 `bf16`）。在 TPU v5e 上，我们将拥有相同的 FLOP/s 但会 _将内存带宽使用量减半_。这使得较小矩阵更容易成为计算受限的。让我们看看 1024 x 1024 x 1024 `bf16` 矩阵乘法的强度：

```python
print(f"{matmul_flops_intensity(1024, 1024, 1024, jnp.bfloat16)} flops/byte")
```

```
341.3333333333333 flops/byte
```

我们现在有了一个计算受限的矩阵乘法！

让我们为矩阵乘法内核添加 `bf16` 支持。

原生 MXU `bf16` 矩阵乘法例程接受两个输入 `bf16` 矩阵并在 `f32` 中累加。我们将通过向 `jnp.matmul` 传递 `preferred_element_type=jnp.float32` 来触发此例程。我们还需要一个 `f32` 的累加器 `Ref`。然后我们将在写回 HBM 之前将输出向下转换为 `bf16`。这样我们不会丢失任何精度，不会做任何额外的转换，并且仍然保留 `bf16` 的内存带宽节省。

> 注意，目前分配临时空间的唯一方式是通过 `pltpu.PrefetchScalarGridSpec`。暂时不用担心它到底做什么——你现在只需知道它允许你在 VMEM 中分配临时空间。

```python
def matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps):
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(
        x_ref[...], y_ref[...], preferred_element_type=jnp.float32
    )

    @pl.when(pl.program_id(2) == nsteps - 1)
    def _():
        z_ref[...] = acc_ref[...].astype(z_ref.dtype)


@functools.partial(jax.jit, static_argnames=['bm', 'bk', 'bn'])
def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
):
    m, k = x.shape
    _, n = y.shape
    return pl.pallas_call(
        functools.partial(matmul_kernel, nsteps=k // bk),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            grid=(m // bm, n // bn, k // bk),
        ),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
    )(x, y)
```

```python
m, k, n = 4096, 4096, 4096
k1, k2 = random.split(random.key(0), 2)
x = random.normal(k1, (m, k), dtype=jnp.bfloat16)
y = random.normal(k2, (k, n), dtype=jnp.bfloat16)
np.testing.assert_array_equal(x @ y, matmul(x, y))
```

## 流水线内核的性能

我们上面关于 FLOPs 与内存使用量的分析适用于粗粒度，即当我们在看整体矩阵乘法的大小时。然而，请记住在实践中，我们是在对分块矩阵乘法进行流水线执行，这意味着我们有一个循环，在其中用更小的块进行矩阵乘法。

这意味着我们实际上关心的是每个单独内核实例的 FLOPs 与内存带宽使用量，而不是全局的 FLOPs 与内存带宽使用量。

此外，在对矩阵乘法操作进行分块时，相同的值可能从内存中被多次读取。具体来说，内核第一个操作数的内存带宽为 `(bm * bk)`，需要乘以 grid 维度，即 `(bm * bk) * m // bm * n // bn * k // bk = m * k * n // bn`。第二个操作数类似，总带宽使用量为 `(m * k * n // bn + k * n * m // bm + m * n) * element_size`。

因此，块大小 `bm`、`bk`、`bn` 对性能极其重要。即使我们拥有世界上最大的矩阵，如果我们选择非常小的 `bm`、`bk` 和 `bn`，我们将是内存受限的，因为每次调用内核时，我们的 FLOPs 太少，无法隐藏后台发生的内存传输。

因此直觉应该是：**要成为计算受限的，让块尽可能大！** 有两个主要约束：

1. **VMEM 使用量**：块越大，使用的 VMEM 越多。块足够大时，我们会用完。

2. **流水线气泡**：相对于矩阵大小，我们的块越大，流水线中的循环迭代就越少。这将使流水线开始和结束时的气泡大小相对于总流水线更大，这个开销可能不可忽略。

在 Pallas 中获得良好的矩阵乘法性能归结为选择好的块大小来平衡这个优化问题。在实践中，我们通常在大量候选块大小上进行扫描，分析内核的性能，然后选择最好的一个。

现在，让我们做一些非常简单的计时实验。我们将使用 `timeit` 来测量运行每个内核所需的时间。注意，这是内核实际运行时间的上界，因为我们使用 `timeit` 测量了 Python 调度和其他开销。我们将计算以这种方式获得的 FLOP/s 量，并计算与芯片提供的能力相比我们获得的利用率百分比，我们将使用一些合理的块大小来验证我们的直觉。

```python
import timeit

def benchmark(f, ntrials: int = 100):
    def run(*args, **kwargs):
        jax.block_until_ready(f(*args, **kwargs))
        result = timeit.timeit(lambda: jax.block_until_ready(f(*args, **kwargs)),
                               number=ntrials)
        time = result / ntrials
        return time
    return run

def analyze_matmul(m: int, k: int, n: int, dtype: np.dtype,
                   mm_func):
    x = jnp.ones((m, k), dtype=dtype)
    y = jnp.ones((k, n), dtype=dtype)
    time = benchmark(mm_func)(x, y)
    print(f"----- {m} x {k} x {n} -----")
    print("Matmul time: ", time)
    mm_flops = matmul_flops(m, k, n) / time
    print("Matmul FLOP/s: ", mm_flops)
    print(f"FLOP/s utilization: {mm_flops / v5e_flops * 100:.4f}%")
    print()

print("================bm=128, bk=128, bn=128===================")
mm = functools.partial(matmul, bm=128, bk=128, bn=128)
analyze_matmul(1024, 1024, 1024, jnp.bfloat16, mm)
analyze_matmul(4096, 4096, 4096, jnp.bfloat16, mm)
analyze_matmul(8192, 8192, 8192, jnp.bfloat16, mm)

print("================bm=512, bk=1024, bn=1024===================")
mm = functools.partial(matmul, bm=512, bk=1024, bn=1024)
analyze_matmul(1024, 1024, 1024, jnp.bfloat16, mm)
analyze_matmul(4096, 4096, 4096, jnp.bfloat16, mm)
analyze_matmul(8192, 8192, 8192, jnp.bfloat16, mm)
```

```
================bm=128, bk=128, bn=128===================
----- 1024 x 1024 x 1024 -----
Matmul time:  0.00029766598949208854
Matmul FLOP/s:  7214407167121.377
FLOP/s utilization: 3.6621%

----- 4096 x 4096 x 4096 -----
Matmul time:  0.011771515250438824
Matmul FLOP/s:  11675553278230.387
FLOP/s utilization: 5.9267%

----- 8192 x 8192 x 8192 -----
Matmul time:  0.09183577066054567
Matmul FLOP/s:  11972585626140.668
FLOP/s utilization: 6.0775%

================bm=512, bk=1024, bn=1024===================
----- 1024 x 1024 x 1024 -----
Matmul time:  0.00012708659982308746
Matmul FLOP/s:  16897797651282.135
FLOP/s utilization: 8.5776%

----- 4096 x 4096 x 4096 -----
Matmul time:  0.00088908776990138
Matmul FLOP/s:  154584235803001.88
FLOP/s utilization: 78.4692%

----- 8192 x 8192 x 8192 -----
Matmul time:  0.006099433819763363
Matmul FLOP/s:  180264539343531.62
FLOP/s utilization: 91.5048%
```

更大的块大小帮助很大！我们在较大的矩阵乘法中获得了相当好的利用率（80-90%），但最小的矩阵乘法似乎很难获得好的性能。

让我们将其与 XLA 的矩阵乘法进行比较。我们不期望 Pallas 比 XLA 更好，因为 XLA 在生成矩阵乘法方面 _非常_ 出色，但希望我们接近。通过更仔细的块大小调优（留作未来工作），我们也可以达到 XLA 的性能。

```python
print("================ XLA matmul ===================")
mm = jnp.matmul
analyze_matmul(1024, 1024, 1024, jnp.bfloat16, mm)
analyze_matmul(4096, 4096, 4096, jnp.bfloat16, mm)
analyze_matmul(8192, 8192, 8192, jnp.bfloat16, mm)
```

```
================ XLA matmul ===================
----- 1024 x 1024 x 1024 -----
Matmul time:  0.00011943008983507753
Matmul FLOP/s:  17981093801113.996
FLOP/s utilization: 9.1275%

----- 4096 x 4096 x 4096 -----
Matmul time:  0.0008272899803705514
Matmul FLOP/s:  166131533963991.34
FLOP/s utilization: 84.3307%

----- 8192 x 8192 x 8192 -----
Matmul time:  0.006047147869830951
Matmul FLOP/s:  181823175395037.44
FLOP/s utilization: 92.2960%
```

通过一些非常基本的调优，Pallas 已经相当接近 XLA 的性能数据了！通过尝试更多的块大小，我们应该可以完全弥合差距。

## 模板化矩阵乘法

现在我们有了基本的矩阵乘法内核，可以尝试将操作融合进去。

### 融合右操作数转置

一个常见的第一步是融合转置。这是什么意思？假设我们想计算 `x @ y.T` 而不是 `x @ y`。朴素地，我们可以先计算 `y.T` 然后将其传入高效的矩阵乘法内核。然而，`y.T` 操作本身并不是免费的——它涉及复制 $O(n^2)$ 数据。理想情况下，我们可以在只使用一个内核的情况下在矩阵乘法的 _同时_ 计算转置，即将其与矩阵乘法"融合"。

加速器通常支持融合右操作数转置的原生矩阵乘法例程。例如 TPU v5e，MXU 允许我们对小数组进行 `x @ y.T`。我们可以通过 `jax.lax.dot_general` 调用此例程，这比先转置再分别做矩阵乘法更高效。

```python
def matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps, transpose_rhs):
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    # dot_general 期望一个数据结构 (contraction_dims, batch_dims)，
    # 其中 contraction_dims 是 LHS 和 RHS 中将在矩阵乘法中被收缩（归约）的
    # 维度集合；另一方面，batch_dims 是被循环遍历的。其余维度将是矩阵乘法的
    # 输入和输出维度。
    if transpose_rhs:
        dims = ((1,), (1,)), ((), ())
    else:
        dims = ((1,), (0,)), ((), ())

    acc_ref[...] += jax.lax.dot_general(
        x_ref[...], y_ref[...], dims, preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == nsteps - 1)
    def _():
        z_ref[...] = acc_ref[...].astype(z_ref.dtype)


@functools.partial(jax.jit, static_argnames=['bm', 'bk', 'bn', 'transpose_rhs'])
def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
    transpose_rhs: bool = False,
):
    if transpose_rhs:
        y = y.swapaxes(0, 1)
        y_block_spec = pl.BlockSpec((bn, bk), lambda i, j, k: (j, k))
    else:
        y_block_spec = pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))
    m, k = x.shape
    _, n = y.shape
    return pl.pallas_call(
        functools.partial(matmul_kernel, nsteps=k // bk, transpose_rhs=transpose_rhs),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                y_block_spec,
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            grid=(m // bm, n // bn, k // bk),
        ),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
    )(x, y)
```

我们在 `matmul` 函数内部做了一个转置（`y = y.swapaxes(0, 1)`）。这是因为在 JIT 编译的 JAX 计算内部，维度顺序纯粹是 _逻辑_ 的，不是物理的，所以重新排列维度并不意味着物理布局的差异。然而，当我们将数组传入 `pallas_call` 时，我们确实强制执行了主到次的维度排序约束。通过在 `matmul` 函数内部转置 `y`，我们请求 `y` 处于转置布局 `(n, k)` 而不是通常的 `(k, n)`。然而，用户仍然会以（逻辑）`(k, n)` 维度传入数组。

注意：为了基准测试转置，我们实际上希望 `y` 在传入内核时处于物理转置布局，这样我们就不会测量重新布局的时间。在包装函数中，我们将（逻辑上）把它转置回 `(k, n)` 再传入 `matmul`，因为 `matmul` 期望逻辑 `(k, n)` 维度排序。

```python
def analyze_matmul(m: int, k: int, n: int, dtype: np.dtype,
                   mm_func, transpose_rhs: bool = False):
    x = jnp.ones((m, k), dtype=dtype)
    if transpose_rhs:
        y = jnp.ones((n, k), dtype=dtype)
        @jax.jit
        def _wrapper(x, y):
            y = y.swapaxes(0, 1)
            return mm_func(x, y, transpose_rhs=True)
    else:
        y = jnp.ones((k, n), dtype=dtype)
        _wrapper = mm_func
    time = benchmark(_wrapper)(x, y)
    print(f"----- {m} x {k} x {n} -----")
    print("Matmul time: ", time)
    mm_flops = matmul_flops(m, k, n) / time
    print("Matmul FLOP/s: ", mm_flops)
    print(f"FLOP/s utilization: {mm_flops / v5e_flops * 100:.4f}%")
    print()

print("================bm=128, bk=128, bn=128===================")
mm = functools.partial(matmul, bm=128, bk=128, bn=128)
analyze_matmul(1024, 1024, 1024, jnp.bfloat16, mm, transpose_rhs=True)
analyze_matmul(4096, 4096, 4096, jnp.bfloat16, mm, transpose_rhs=True)
analyze_matmul(8192, 8192, 8192, jnp.bfloat16, mm, transpose_rhs=True)

print("================bm=512, bk=1024, bn=1024===================")
mm = functools.partial(matmul, bm=512, bk=1024, bn=1024)
analyze_matmul(1024, 1024, 1024, jnp.bfloat16, mm, transpose_rhs=True)
analyze_matmul(4096, 4096, 4096, jnp.bfloat16, mm, transpose_rhs=True)
analyze_matmul(8192, 8192, 8192, jnp.bfloat16, mm, transpose_rhs=True)
```

```
================bm=128, bk=128, bn=128===================
----- 1024 x 1024 x 1024 -----
Matmul time:  0.0003029372810851783
Matmul FLOP/s:  7088872126624.065
FLOP/s utilization: 3.5984%

----- 4096 x 4096 x 4096 -----
Matmul time:  0.012017967159627005
Matmul FLOP/s:  11436123235026.848
FLOP/s utilization: 5.8051%

----- 8192 x 8192 x 8192 -----
Matmul time:  0.09500920018996112
Matmul FLOP/s:  11572685861765.383
FLOP/s utilization: 5.8745%

================bm=512, bk=1024, bn=1024===================
----- 1024 x 1024 x 1024 -----
Matmul time:  0.00012131539988331496
Matmul FLOP/s:  17701657415839.363
FLOP/s utilization: 8.9856%

----- 4096 x 4096 x 4096 -----
Matmul time:  0.0008790623804088682
Matmul FLOP/s:  156347213275211.03
FLOP/s utilization: 79.3641%

----- 8192 x 8192 x 8192 -----
Matmul time:  0.006107717020204291
Matmul FLOP/s:  180020067095253.78
FLOP/s utilization: 91.3807%
```

看到了吗，尽管有额外的转置，我们仍然获得了相同的利用率！

### 融合激活函数

融合激活函数也非常常见。这确保我们不会在一个高效、计算受限的矩阵乘法内核后面跟一个缓慢、内存受限的激活内核。

```python
def matmul_kernel(
    x_ref, y_ref, z_ref, acc_ref, *, nsteps, transpose_rhs, activation
):
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    if transpose_rhs:
        dims = ((1,), (1,)), ((), ())
    else:
        dims = ((1,), (0,)), ((), ())

    acc_ref[...] += jax.lax.dot_general(
        x_ref[...],
        y_ref[...],
        dims,
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == nsteps - 1)
    def _():
        z_ref[...] = activation(acc_ref[...]).astype(z_ref.dtype)


@functools.partial(jax.jit, static_argnames=['bm', 'bk', 'bn', 'activation'])
def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
    transpose_rhs: bool = False,
    activation: Callable[[jax.Array], jax.Array] = lambda x: x,
):
    if transpose_rhs:
        y = y.swapaxes(0, 1)
        y_block_spec = pl.BlockSpec((bn, bk), lambda i, j, k: (j, k))
    else:
        y_block_spec = pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))
    m, k = x.shape
    _, n = y.shape
    return pl.pallas_call(
        functools.partial(
            matmul_kernel,
            nsteps=k // bk,
            transpose_rhs=transpose_rhs,
            activation=activation,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                y_block_spec,
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            grid=(m // bm, n // bn, k // bk),
        ),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
    )(x, y)
```

```python
def analyze_matmul(m: int, k: int, n: int, dtype: np.dtype,
                   mm_func, transpose_rhs: bool = False,
                   activation = lambda x: x):
    x = jnp.ones((m, k), dtype=dtype)
    if transpose_rhs:
        y = jnp.ones((n, k), dtype=dtype)
        @jax.jit
        def _wrapper(x, y):
            y = y.swapaxes(0, 1)
            return mm_func(x, y, transpose_rhs=True, activation=activation)
    else:
        y = jnp.ones((k, n), dtype=dtype)
        _wrapper = functools.partial(mm_func, activation=activation)
    time = benchmark(_wrapper)(x, y)
    print(f"----- {m} x {k} x {n} -----")
    print("Matmul time: ", time)
    mm_flops = matmul_flops(m, k, n) / time
    print("Matmul FLOP/s: ", mm_flops)
    print(f"FLOP/s utilization: {mm_flops / v5e_flops * 100:.4f}%")
    print()


activation = jax.nn.relu
print("================bm=128, bk=128, bn=128===================")
mm = functools.partial(matmul, bm=128, bk=128, bn=128)
analyze_matmul(1024, 1024, 1024, jnp.bfloat16, mm, activation=activation)
analyze_matmul(4096, 4096, 4096, jnp.bfloat16, mm, activation=activation)
analyze_matmul(8192, 8192, 8192, jnp.bfloat16, mm, activation=activation)

print("================bm=512, bk=1024, bn=1024===================")
mm = functools.partial(matmul, bm=512, bk=1024, bn=1024)
analyze_matmul(1024, 1024, 1024, jnp.bfloat16, mm, activation=activation)
analyze_matmul(4096, 4096, 4096, jnp.bfloat16, mm, activation=activation)
analyze_matmul(8192, 8192, 8192, jnp.bfloat16, mm, activation=activation)
```

```
================bm=128, bk=128, bn=128===================
----- 1024 x 1024 x 1024 -----
Matmul time:  0.00030103540048003196
Matmul FLOP/s:  7133658182976.541
FLOP/s utilization: 3.6211%

----- 4096 x 4096 x 4096 -----
Matmul time:  0.011807117109419778
Matmul FLOP/s:  11640348122095.826
FLOP/s utilization: 5.9088%

----- 8192 x 8192 x 8192 -----
Matmul time:  0.09181861146935262
Matmul FLOP/s:  11974823079773.941
FLOP/s utilization: 6.0786%

================bm=512, bk=1024, bn=1024===================
----- 1024 x 1024 x 1024 -----
Matmul time:  0.00012622540001757442
Matmul FLOP/s:  17013086492108.6
FLOP/s utilization: 8.6361%

----- 4096 x 4096 x 4096 -----
Matmul time:  0.000896632740041241
Matmul FLOP/s:  153283442968721.44
FLOP/s utilization: 77.8089%

----- 8192 x 8192 x 8192 -----
Matmul time:  0.006130605939542875
Matmul FLOP/s:  179347953304919.88
FLOP/s utilization: 91.0396%
```

额外的融合激活几乎完全不影响我们的利用率！

## 结论

在本指南中，我们介绍了如何使用 Pallas 在 TPU 上编写高效的矩阵乘法。我们讨论了分块矩阵乘法和流水线，如何分析 TPU 矩阵乘法的性能，以及如何编写高效的 `bf16` 矩阵乘法。最后我们通过模板化矩阵乘法来支持融合转置和融合激活函数。

留给读者的练习：

- 添加输入融合支持。有时我们想把操作融合到矩阵乘法的输入中。尝试进一步模板化矩阵乘法来支持这一点。

- 添加 `int8` 矩阵乘法支持。TPU v5 支持原生 `int8` 矩阵乘法，FLOPs 是 `bf16` 的两倍。尝试添加支持并看看可能的利用率。

- 为 `matmul` 函数添加反向传播支持。你可以使用 `jax.custom_vjp` 来实现。
