---
title: "软件流水线"
date: 2026-04-19
draft: false
weight: 3
---

# 软件流水线

软件流水线是性能优化中的一项重要技术，它通过重叠多个异步操作来提升性能，即使这些操作之间存在数据依赖关系也是如此。在编写内核的上下文中，最常见的流水线形式涉及将通信和内存传输与计算重叠，使得硬件加速器在等待数据到达时永远不会停顿。因此，在本教程中我们将专注于通信-计算流水线问题。我们将首先从概念层面介绍该问题，然后概述用于编写流水线的 Pallas API，最后使用该 API 给出一些实际示例。

本教程仅涵盖流水线的概念基础。有关特定平台的参考，请参见 [TPU 流水线](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html) 或 [Mosaic GPU 流水线](https://docs.jax.dev/en/latest/pallas/gpu/pipelining.html)。

```python
import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
import numpy as np
```

## 内存层次结构

理解流水线概念的第一步是理解可用的不同形式的内存及其之间的权衡。大多数硬件架构（包括 CPU、GPU 和 TPU）利用多种不同的内存空间，在容量与延迟/带宽之间进行权衡。就 Pallas 而言，我们通常关注寄存器、SRAM、DRAM 以及可能的网络通信：

- **寄存器**是物理上最靠近处理器的内存，通常值必须在进行任何计算之前直接加载到寄存器中。

- **SRAM**（在 GPU 上也称为共享内存/L1 和 L2 缓存，在 TPU 上称为 VMEM）也位于处理器附近，但容量比寄存器更大。现代 ML 加速器上的 SRAM 通常在 10-100MB 范围内（TPU v5p 包含 96MB 的 VMEM，H100 GPU 包含约 30MB 的 L1 缓存和 50MB 的 L2）。可以合理预期访问 SRAM 的延迟大约是访问寄存器的 10 倍。

- **DRAM**（也称为 HBM）的容量远大于 SRAM，现代 ML 加速器通常在 10-100GB 范围内。然而，与 SRAM 相比，访问延迟大约长 10 倍。

- **网络通信**在更大的工作负载中变得至关重要，当单个设备上的 DRAM 大小不足时，或者当我们希望利用并行计算时。本教程不涉及分布式流水线，但请参见 [分布式 TPU 内核指南](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html) 以了解跨多个设备编写流水线的内容。

![内存层次结构](https://docs.jax.dev/en/latest/_images/pipelining_mem_hierarchy.svg)

为了对存储在 HBM 中的值 X 和 Y 执行计算，我们需要：

- 将值 x 和 y 复制到 SRAM 中。
- 从 SRAM 将值加载到寄存器中。
- 执行计算并将结果存储到寄存器中。
- 将输出寄存器中的值存储到 SRAM 中。
- 将 SRAM 中的输出值复制回 HBM。

让我们实现一个能做到这一点的 Pallas 函数！

```python
# 注意：这是一个 TPU 示例。

def add_matrices_kernel(x_sram_ref, y_sram_ref, z_sram_ref):
  # 从 SRAM 将 x 和 y 加载到寄存器中
  x_regs = x_sram_ref[:, :]
  y_regs = y_sram_ref[:, :]
  # 执行向量化加法
  z_regs = x_regs + y_regs
  # 将寄存器中的输出值存储回 SRAM
  z_sram_ref[:, :] = z_regs

def add_matrices(x: jax.Array, y: jax.Array) -> jax.Array:
  # pallas_call 将首先为 `x` 和 `y` 在 SRAM 中分配临时缓冲区。
  # 然后它将 `x` 和 `y` 从 HBM 复制到 SRAM。
  z = pl.pallas_call(
      add_matrices_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
  )(x, y)
  # pallas_call 也会将输出从 SRAM 复制回 HBM。
  return z

x, y = jnp.ones((512, 512)), jnp.ones((512, 512))
add_matrices(x, y)
```

```
Array([[2., 2., 2., ..., 2., 2., 2.],
       [2., 2., 2., ..., 2., 2., 2.],
       [2., 2., 2., ..., 2., 2., 2.],
       ...,
       [2., 2., 2., ..., 2., 2., 2.],
       [2., 2., 2., ..., 2., 2., 2.],
       [2., 2., 2., ..., 2., 2., 2.]], dtype=float32)
```

我们编写了两个函数：`add_matrices_kernel` 和 `add_matrices`。

`add_matrices_kernel` 使用存储在 SRAM 中的 `Ref` 进行操作。从 SRAM Ref 加载会产生一个存储在寄存器中的值。寄存器中的值的行为类似于 jax.Array，我们可以对它们使用 `jnp` 和 `jax.lax` 操作来产生新的存储在寄存器中的值。当我们产生了想要返回的值时，我们将它们存储在输出 SRAM `Ref` 中。

`add_matrices` 函数作用于 `jax.Array` 并返回一个 `jax.Array`。在其内部，我们将 `x` 和 `y` 传递给 pallas_call。`pallas_call` 负责将 `x` 和 `y` 复制到 SRAM 中，并分配内核操作所需的 SRAM 缓冲区（包括分配 `z_vmem_ref`，即输出 SRAM 缓冲区）。内核函数运行完成后，`pallas_call` 也会将 `z_vmem_ref` 中的值复制到 HBM，产生一个输出 `jax.Array`。

Pallas 暴露了对像 SRAM 这样的低级内存空间的访问，但编写高性能内核需要更加谨慎地利用各种内存空间。例如，我们需要考虑：

- **内存容量**。SRAM 很小！如果我们的数组太大，上面的内核将无法工作，因为我们无法将输入放入 SRAM。作为参考，一个 `f32[2048, 2048]` 数组是 16MiB，所以我们上面的内核无法扩展到中等大小以上的数组。

- **内存带宽**。在 HBM 和 SRAM 之间复制需要很长时间，至少与大多数计算指令相比是这样。上面的 `add_matrices` 函数可能花在 HBM 和 SRAM 之间复制的时间比实际执行加法本身的时间更多。

考虑到这两个约束，我们必须重新思考从加速器中获取性能的策略。

## 流水线基础

我们如何利用层次结构中每种类型内存的优势，既能对存储在 HBM 中的大型数组进行操作，又能利用快速 SRAM 进行计算？流水线是一种非常通用的编程模式，它恰好能让我们做到这一点，但它需要将问题转换为可以并行重叠的更小的子问题。

流水线的第一步是将问题划分为可以放入 SRAM 的更小的子问题。例如，一个逐元素操作可以通过每次对源数组的一个切片进行操作来简单地进行转换，这导致以下 3 个步骤（也称为阶段）：

- **copy_in**：将切片 `A[i]` 从 HBM 复制到 SRAM `X`。
- **compute**：将 `X` 加载到寄存器中，计算结果，并存储在 SRAM `Y` 中。
- **copy_out**：将结果 `Y` 复制回 HBM `A[i]`。

请注意步骤 1-3 之间存在数据依赖，我们无法简单地将它们重叠，因为我们需要步骤 (1) 完成后才能开始步骤 (2)，依此类推。然而，子问题的多次调用之间不存在数据依赖——也就是说，我们可以在执行块 `A[i]` 的步骤 (2) 的同时执行块 `A[i+1]` 的步骤 (1)，以及块 `A[i-1]` 的步骤 (3)。

![流水线示例](https://docs.jax.dev/en/latest/_images/pipelining_example.svg)

上图描绘了一个理想化的流水线程序如何在时间上被调度。关键洞察是，在内核的大部分时间里，复制操作与计算操作是并行执行的，这意味着我们理想情况下可以用计算来"隐藏" HBM/SRAM 之间传输的开销，并使处理器尽可能多地保持忙碌。

初始启动时间和最终收尾时间被称为"气泡"，在此期间只有部分阶段在执行，因为流水线正在被"填充"或"排空"。大部分时间花在流水线的"稳态"阶段，在该阶段中每个流水线阶段在子问题的不同迭代中并行执行。虽然在更通用的流水线方法中目标是实现 N 路并行（其中 N 是阶段数量），但在内核流水线中我们通常受到内存带宽或处理速度的瓶颈限制。因此，我们内核流水线的目标通常是实现处理器 FLOPs/s 的完全利用，这意味着在任何时间点总是有一个 `compute` 块处于活跃状态。在上图中，compute 块在 8 个时间槽中有 6 个是活跃的，假设我们在每个计算时间槽中完全利用了处理器，我们将实现 75% 的处理器利用率。

### 推导双缓冲流水线

现在让我们看看如何用伪代码实现流水线。考虑以下逐元素程序，其中我们使用 `copy_in` 指令从 HBM 加载值（`A[i]`），将结果加 1，然后使用 `copy_out` 将结果存储回 HBM：

```
for i in range(N):
  copy_in(A[i], X)
  Y = X + 1
  copy_out(Y, A[i])
```

这种方法的问题在于 `copy_in` 和 `copy_out` 通常是阻塞操作。因此我们被迫在 GPU/TPU 空闲时等待复制完成，然后在内存空闲时执行计算。我们想要做的是在执行当前循环的计算时，异步地"预取"下一次迭代所需的输入值，使得计算和内存通信同时发生。

为了对我们将要进行的代码变换进行推理，让我们展开 N=4 的循环，并将复制指令分解为单独的 `copy_start` 和 `copy_wait` 操作以表达异步性：

```
  # 迭代 1
  copy_in_start(A[0], X)
  copy_in_wait(X)
  Y = X + 1
  copy_out_start(Y, A[0])
  copy_out_wait(Y)

  # 迭代 2
  copy_in_start(A[1], X)
  copy_in_wait(X)
  Y = X + 1
  copy_out_start(Y, A[1])
  copy_out_wait(Y)

  # 迭代 3
  copy_in_start(A[2], X)
  copy_in_wait(X)
  Y = X + 1
  copy_out_start(Y, A[2])
  copy_out_wait(Y)

  # 迭代 4
  copy_in_start(A[3], X)
  copy_in_wait(X)
  Y = X + 1
  copy_out_start(Y, A[3])
  copy_out_wait(Y)
```

一旦循环被展开，流水线变换就是尽可能早地发出 `copy_start` 指令，并尽可能晚地发出 `copy_wait`（就在我们需要该值之前）。然而，在循环的当前状态中，通过 X 存在一个伪数据依赖——我们不能在对 X 进行异步复制的同时使用它进行计算，否则可能会出现竞争条件。因此，我们可以使用多重缓冲技术，为每个输入 X 和每个输出 Y 保留 2 个缓冲区。有了 2 个缓冲区，我们可以将 `copy_in_start` 提前一个迭代（有 3 个缓冲区则可以提前 2 个迭代，依此类推），并将循环重写如下：

```
  # 前序
  copy_in_start(A[0], X[0])
  
  # 迭代 1
  copy_in_start(A[1], X[1])
  copy_in_wait(X[0])
  Y[0] = X[0] + 1
  copy_out_start(Y[0], A[0])
  copy_out_wait(Y[0])

  # 迭代 2 - 稳态
  copy_in_start(A[2], X[0])
  copy_in_wait(X[1])
  Y[1] = X[1] + 1
  copy_out_start(Y[1], A[1])
  copy_out_wait(Y[1])

  # 迭代 3 - 稳态
  copy_in_start(A[3], X[1])
  copy_in_wait(X[0])
  Y[0] = X[0] + 1
  copy_out_start(Y[0], A[2])
  copy_out_wait(Y[0])

  # 迭代 4 - 无 copy-in
  copy_in_wait(X[1])
  Y[1] = X[1] + 1
  copy_out_start(Y[1], A[3])
  copy_out_wait(Y[1])
```

接下来，我们可以将 `copy_out_wait` 尽可能地推迟，就在我们需要在后续循环迭代中写入 Y 之前。

```
  # 前序
  copy_in_start(A[0], X[0])
  
  # 迭代 1
  copy_in_start(A[1], X[1])
  copy_in_wait(X[0])
  Y[0] = X[0] + 1
  copy_out_start(Y[0], A[0])

  # 迭代 2 - 稳态
  copy_in_start(A[2], X[0])
  copy_in_wait(X[1])
  Y[1] = X[1] + 1
  copy_out_start(Y[1], A[1])
  copy_out_wait(Y[0])

  # 迭代 3 - 稳态
  copy_in_start(A[3], X[1])
  copy_in_wait(X[0])
  Y[0] = X[0] + 1
  copy_out_start(Y[0], A[2])
  copy_out_wait(Y[1])

  # 迭代 4 - 无 copy-in
  copy_in_wait(X[1])
  Y[1] = X[1] + 1
  copy_out_start(Y[1], A[3])
  copy_out_wait(Y[0])

  # 后序
  copy_out_wait(Y[1])
```

最后，将我们的循环重新卷回 for 循环，我们得到以下流水线化的循环：

```python
# 前序
copy_in_start(A[0], X[0])

# 主循环
for i in range(N):
  cur_slot = i % 2
  next_slot = (i + 1) % 2

  if i+1 < N:
    copy_in_start(A[i+1], X[next_slot])
  
  copy_in_wait(X[cur_slot])
  Y[cur_slot] = X[cur_slot] + 1
  copy_out_start(Y[cur_slot], A[i])

  if i > 0:
    copy_out_wait(Y[next_slot])

# 后序
copy_out_wait(Y[1])
```

如果我们想将此循环推广以处理更广泛的计算集合，注意我们本质上需要向流水线指定 3 条信息：

- **网格（grid）**，即指定子问题数量的 for 循环的边界。在我们的示例中，我们有一个大小为 `(N,)` 的一维网格。

- **内核（kernel）**，即输入被加载到 SRAM 后执行的实际计算。在我们的示例中，我们执行了逐元素加法 `Y = X + 1`。

- **数据切片（data_slices）**，将子问题映射到 HBM 缓冲区中相应的切片。在我们的示例中，数据切片是恒等函数 `lambda i: i`。

通过允许用户指定这些信息，我们可以编写遵循此模式的各种程序：

```python
def double_buffered_pipeline(
    grid: tuple[int, ...],
    kernel: Callable,
    in_slices: Callable,
    out_slices: Callable):
  # 前序
  copy_in_start(in_hbm[in_slices(0)], in_sram[0])

  # 主循环
  grid_size = prod(grid)
  for i in range(grid_size):
    cur_slot = i % 2
    next_slot = (i + 1) % 2
    if (i + 1) < grid_size:
      copy_in_start(in_hbm[in_slices(i+1)], in_sram[next_slot])
    copy_in_wait(in_sram[cur_slot])

    kernel(in_sram[cur_slot], out_sram[cur_slot])

    copy_out_start(out_sram[cur_slot], out_hbm[out_slices(i)])
    if i > 0:
      copy_out_wait(out_sram[next_slot])

  # 后序
  last_slot = (grid_size - 1) % 2
  copy_out_wait(out_sram[last_slot])
```

现在我们已经了解了如何手动实现流水线化的循环，让我们看看如何使用 Pallas API。

## Pallas 流水线 API

Pallas 提供了一个流水线 API，它抽象掉了维护多个缓冲区和将异步通信与计算重叠的样板代码。此 API 的基础在 [Pallas 快速入门](https://docs.jax.dev/en/latest/pallas/quickstart.html) 中有介绍，因此我们在这里简要回顾该 API 以确保完整性，并讨论一些由于使用流水线而产生的尖锐边界情况。

### 网格（Grid）

程序网格是一个整数元组，将子问题的数量指定为一个数组。流水线的结构可以被解释为嵌套的 for 循环，其中每个循环的边界如下。

```python
# 对于 grid (N, M, K)
for n in range(N):
  for m in range(M):
    for k in range(K):
      kernel()
```

内核将总共被调用 `prod(grid)` 次。更多详情，请参见 [grid 和 blockspecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)。

### BlockSpecs

`BlockSpec` 指定了在每个子问题上复制到内核的数据的大小和切片。`pl.BlockSpec` 的基本构造函数涉及指定 `block_shape`（数据切片的大小）和 `index_map`（一个接受当前子问题的程序 id 并输出到源缓冲区的分块索引的函数）。分块索引指定在每次迭代中复制哪个块，假设源缓冲区已被切割成形状为 `block_shape` 的块。`memory_space` 参数指定将输入复制到哪个内存空间——默认情况下这将是 SRAM。

```python
pl.BlockSpec(
  block_shape: tuple[int, ...],
  index_map: Callable,
  memory_space: pl.MemorySpace
)
```

每个输入和每个输出到内核都应该有一个 BlockSpec。更多详情，请参见 [grid 和 blockspecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)。

### 内核（Kernel）

内核函数指定在每个子问题上执行什么计算。内核函数不应返回任何输出，相反所有输出应该写入传递给内核的输出缓冲区。默认情况下，所有输入和输出缓冲区都是 SRAM 缓冲区（除非用户通过在相应的 `BlockSpec` 上指定 `memory_space` 来覆盖此行为）。

```python
def kernel(*input_buffers, *output_buffers):
  # ... 执行计算
  # ... 将结果存储到输出缓冲区
```

当前子问题的索引可以在内核内部使用 `pl.program_id(grid_axis: int)` 查询。

### Pallas Call

`pl.pallas_call` 函数是 Pallas 的主要入口点，当提供 grid 和 BlockSpecs 时执行流水线化的执行。它具有以下签名：

```python
def pallas_call(
  kernel,
  grid: tuple[int, ...],
  in_specs: Sequence[PyTree[BlockSpec]],
  out_specs: PyTree[BlockSpec],
  out_shape: PyTree[jax.ShapeDtypeStruct],
) -> Callable:
```

`pallas_call` 将返回一个可调用函数，当使用输入值调用时，将返回与 `out_shape` 形状相同的输出。

`in_specs`、`out_specs` 和 `out_shape` 是其各自元素类型的 PyTree。`in_specs` 和提供给内核的输入缓冲区的 PyTree 应该匹配，`out_specs` 和 `out_shape` 的 PyTree 也应该匹配。

### 示例 - 重新审视逐元素内核

让我们重新审视教程开头的 `add_matrices_kernel`，这次使用流水线。我们将添加两个形状为 `f32[4096, 4096]` 的输入数组，它们存储在 HBM 中。作为子问题，我们将输入切割成 `block_shape=(512, 512)` 的块，并且每次只在内核中将两个块相加。因为加法是逐元素的，每个 `index_map` 都是相同的，在第 `i, j` 次迭代中选择第 `i, j` 个块。

```python
# 注意：这是一个 TPU 示例。

total_shape = (4096, 4096)
block_shape = (512, 512)

def add_matrices_pipelined_kernel(x_ref, y_ref, o_ref):
  o_ref[...] = x_ref[...] + y_ref[...]

def add_matrices_pipelined(x: jax.Array, y: jax.Array):
  return pl.pallas_call(
    add_matrices_pipelined_kernel,
    grid=tuple(total // block for (total, block) in zip(total_shape, block_shape)),
    in_specs=[
      pl.BlockSpec(block_shape, index_map=lambda i, j: (i, j)),
      pl.BlockSpec(block_shape, index_map=lambda i, j: (i, j))
    ],
    out_specs=pl.BlockSpec(block_shape, index_map=lambda i, j: (i, j)),
    out_shape=jax.ShapeDtypeStruct(total_shape, dtype=jnp.float32),
  )(x, y)

x = jax.random.uniform(jax.random.key(0), total_shape, dtype=jnp.float32)
y = jax.random.uniform(jax.random.key(1), total_shape, dtype=jnp.float32)
result = add_matrices_pipelined(x, y)
np.testing.assert_array_equal(
    result, x + y
)
```

事实证明，使用这个 API，编写一个流水线化的内核并不比编写我们原始的朴素加法内核多多少行代码！

### 参数化内核

在我们的内核中参数化块形状是很常见的。块大小可能是优化 Pallas 内核性能时最重要的调优参数！它们让我们控制流水线（例如，选择更小的块会增加流水线循环的迭代次数，其中每次迭代的工作量更少）。让我们编写一个实现此功能的函数：

```python
def add_matrices_pipelined_param(
    x: jax.Array, y: jax.Array, *, bm: int = 256, bn: int = 256
) -> jax.Array:
  m, n = x.shape
  block_spec = pl.BlockSpec((bm, bn), lambda i, j: (i, j))
  return pl.pallas_call(
      add_matrices_kernel,
      out_shape=x,
      in_specs=[block_spec, block_spec],
      out_specs=block_spec,
      grid=(m // bm, n // bn),
  )(x, y)

np.testing.assert_array_equal(
    add_matrices_pipelined_param(x, y, bm=256, bn=256), x + y
)
np.testing.assert_array_equal(
    add_matrices_pipelined_param(x, y, bm=128, bn=128), x + y
)
np.testing.assert_array_equal(
    add_matrices_pipelined_param(x, y, bm=512, bn=512), x + y
)
```

## 尖锐边界情况

虽然流水线提供了一种接近于简单地在循环中调用内核函数的心智模型，但由于使用中间缓冲区而产生了一些尖锐的边界情况，这些缓冲区并没有完全对用户隐藏，可能导致微妙的 bug。

### 缓冲区重访

一般来说，一个好的经验法则是传递给内核函数的输入缓冲区应该被视为只读的，输出缓冲区是只写的。

写入输入和读取输出在大多数情况下会导致不正确的结果。这是因为传递给内核的 SRAM 缓冲区仅包含底层 HBM 缓冲区中数据的副本。如果输入 SRAM 缓冲区被更新，更新的结果永远不会被写回 HBM，如果输出缓冲区被更新，其更新的值永远不会被读入 SRAM。这个问题类似于一般使用缓存时遇到的过期问题。

有两种情况缓冲区支持读写——累加（接下来讨论），以及通过向 `pallas_call` 传递 `input_output_aliases` 参数将一对输入和输出缓冲区标记为输入-输出别名。

### 归约和累加

归约/累加只应在网格的最后（最内层）维度上执行，并且缓冲区应该首先手动初始化。

归约是流水线支持对输出缓冲区同时进行读和写的少数情况之一，但其工作原理很微妙。Pallas 流水线发射器执行一个优化：如果两个连续迭代之间的数据切片相同，流水线将不会在该缓冲区上发出 `copy_in`/`copy_out`。这意味着在上一个迭代中使用的同一个 SRAM 缓冲区将在下一个迭代中再次传递给内核，因此在上一次迭代中对输出缓冲区发出的任何写入在下一次迭代中都将变得可见。一旦数据切片改变，最终累加的 SRAM 缓冲区将被写出到 HBM。这也是为什么归约必须在网格的最后维度上执行的原因——我们希望在输出缓冲区位于 SRAM 中时在最内层循环中完成所有累加，然后将其写入 HBM 并且不再触碰该输出块。

作为一个具体的例子，让我们考虑执行以下计算，将一个 `(8, 1024, 1024)` 数组沿第一个轴归约为一个 `(1024, 1024)` 数组。

```python
x = jnp.ones((8, 1024, 1024))
jnp.sum(x, axis=0)
```

```
Array([[8., 8., 8., ..., 8., 8., 8.],
       [8., 8., 8., ..., 8., 8., 8.],
       [8., 8., 8., ..., 8., 8., 8.],
       ...,
       [8., 8., 8., ..., 8., 8., 8.],
       [8., 8., 8., ..., 8., 8., 8.],
       [8., 8., 8., ..., 8., 8., 8.]], dtype=float32)
```

要使用 `pallas_call` 实现这个，我们可以使用大小为 `(8,)` 的网格，并在每次迭代 i 中将 `x[i]` 加载到 SRAM 中。然后我们可以将 `x[i]` 加到输出 SRAM 缓冲区中。让我们先朴素地实现它。

```python
# 注意：这是一个 TPU 示例。

# 警告：此实现是不正确的！
def incorrect_sum_kernel(x_ref, o_ref):
  o_ref[...] += x_ref[...]

def incorrect_sum(x: jax.Array,
              block_size: tuple[int, ...] = (256, 256)) -> jax.Array:
  reduction_size, *out_shape = x.shape
  grid = (reduction_size, *(out // blk for out, blk in zip(out_shape, block_size)))
  return pl.pallas_call(
      incorrect_sum_kernel,
      grid=grid,
      # block_shape 中的 None 表示我们选择大小为 1 并将其压缩掉
      in_specs=[pl.BlockSpec((None, *block_size), lambda i, j, k: (i, j, k))],
      out_specs=pl.BlockSpec(block_size, lambda i, j, k: (j, k)),
      out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
  )(x)

result = incorrect_sum(x)
print(result)
```

```
[[65. 65. 65. ... 66. 66. 66.]
 [65. 65. 65. ... 66. 66. 66.]
 [65. 65. 65. ... 66. 66. 66.]
 ...
 [71. 71. 71. ... 72. 72. 72.]
 [71. 71. 71. ... 72. 72. 72.]
 [71. 71. 71. ... 72. 72. 72.]]
```

这个结果完全是错误的！

这个内核中有两个错误。首先，我们在第一个网格维度而不是最后一个网格维度上进行累加。其次，`o_ref` 初始包含垃圾值，因此我们需要在开始累加之前将其初始化为零。

修复这两个问题后，我们得到以下修正后的内核。在这个新内核中，我们使用 `@pl.when` 创建一个条件，检查归约轴上的程序 ID 何时为 `0`，表示我们正在开始累加到一个新的输出块。我们还将归约维度移到了 `grid` 的最后一个轴。

```python
# 注意：这是一个 TPU 示例。

def correct_sum_kernel(x_ref, o_ref):
  @pl.when(pl.program_id(2) == 0)
  def _():
    o_ref[...] = jnp.zeros_like(o_ref)
  o_ref[...] += x_ref[...]

def correct_sum(x: jax.Array,
              block_size: tuple[int, ...] = (256, 256)) -> jax.Array:
  reduction_size, *out_shape = x.shape
  # 我们将归约移到了网格的最后一个轴。
  grid = (*(out // blk for out, blk in zip(out_shape, block_size)), reduction_size)
  return pl.pallas_call(
      correct_sum_kernel,
      grid=grid,
      # block_shape 中的 None 表示我们选择大小为 1 并将其压缩掉
      in_specs=[pl.BlockSpec((None, *block_size), lambda i, j, k: (k, i, j))],
      out_specs=pl.BlockSpec(block_size, lambda i, j, k: (i, j)),
      out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
  )(x)

result = correct_sum(x)
print(result)
```

```
[[8. 8. 8. ... 8. 8. 8.]
 [8. 8. 8. ... 8. 8. 8.]
 [8. 8. 8. ... 8. 8. 8.]
 ...
 [8. 8. 8. ... 8. 8. 8.]
 [8. 8. 8. ... 8. 8. 8.]
 [8. 8. 8. ... 8. 8. 8.]]
```

## 性能分析

流水线化内核的性能如何？这个问题根据硬件瓶颈所在而有所不同。我们通常关注 3 个量：

- **内存延迟 α**，内存传输的最小延迟。

- **内存带宽 β**，我们从 HBM 传输到 SRAM 的速率，以字节/秒为单位。

- **FLOPs/s F**，即每秒浮点运算次数，处理器每秒能执行的计算数量。

如果处理速度 FLOPs/s 是瓶颈，我们称程序为**计算受限**；如果带宽或延迟是瓶颈，则称为**内存受限**。通常，我们的目标是优化内核使其成为计算受限的，这意味着我们在利用硬件的所有可用处理能力。

假设我们正在运行一个程序，每次内核迭代需要 X 字节的内存传输，并运行 Y 次浮点运算。X 与 Y 的比率取决于计算类型——对于像加法或乘法这样的逐元素操作，它们将等比例缩放。然而，对于像矩阵乘法这样的操作，计算随问题大小呈三次方缩放，而内存呈二次方缩放。

在计算受限的情况下，运行 N 次迭代的流水线将花费 (α + X/β) + N(Y/F) 秒，其中第一项代表初始气泡的开销（如果末尾也有气泡则乘以 2），第二项代表流水线稳态的总时间。假设 N 足够大且有足够的工作来产生长流水线，运行时间的主导项是 F，即加速器的处理速度。

![计算受限的流水线](https://docs.jax.dev/en/latest/_images/pipelining_compute_bound.svg)

在内存受限的情况下，识别问题是延迟还是带宽很有用。如果带宽是瓶颈，那么总运行时间将花费 α + N(X/β) 秒。与延迟受限的情况不同，内存复制串行发生，因为带宽已经饱和。内存受限通常不是理想的，因为会有处理器空闲的时间间隔，而且在大多数硬件配置中，内存带宽 β 比处理速度 F 慢几个数量级。

![带宽受限的流水线](https://docs.jax.dev/en/latest/_images/pipelining_bandwidth_bound.svg)

如果瓶颈具体是延迟而不是带宽，可以通过插入额外的流水线阶段来解决问题，代价是需要额外的 SRAM 来存储更多缓冲区。有了足够的阶段，问题将再次变为计算受限或带宽受限，取决于在流水线稳态阶段我们首先遇到哪个瓶颈。然而，多阶段流水线的缺点是气泡的大小与阶段数成正比，因此确保流水线足够长以使气泡不会占用总运行时间的大部分很重要。

![延迟受限的多阶段流水线](https://docs.jax.dev/en/latest/_images/pipelining_latency_multistage.svg)

##### TPU 上的 Pallas 仅支持双缓冲，因为 TPU 程序可以在更大的块大小上操作，双缓冲通常足以覆盖延迟。在 GPU 上，流水线阶段数可以在 Triton（通过 `CompilerParams`）和 Mosaic GPU 后端（通过流水线发射器的参数）中指定。更多详情请参见特定平台的流水线文档。
