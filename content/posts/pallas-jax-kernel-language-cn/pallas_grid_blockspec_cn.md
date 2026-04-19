# Grid 和 BlockSpec

## `grid`，又称循环中的内核

使用 [`jax.experimental.pallas.pallas_call()`](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.pallas_call.html) 时，内核函数会在不同的输入上被执行多次，具体由 `pallas_call` 的 `grid` 参数指定。概念上：

```python
pl.pallas_call(some_kernel, grid=(n,))(...)
```

映射为

```python
for i in range(n):
  some_kernel(...)
```

Grid 可以推广为多维的，对应于嵌套循环。例如，

```python
pl.pallas_call(some_kernel, grid=(n, m))(...)
```

等价于

```python
for i in range(n):
  for j in range(m):
    some_kernel(...)
```

这可以推广到任意整数元组（长度为 `d` 的 grid 将对应 `d` 层嵌套循环）。内核被执行的次数为 `prod(grid)` 次。默认的 grid 值 `()` 导致内核被调用一次。每次调用被称为一个"程序"。要访问内核当前正在执行的是哪个程序（即 grid 中的哪个元素），我们使用 [`jax.experimental.pallas.program_id()`](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.program_id.html)。例如，对于调用 `(1, 2)`，`program_id(axis=0)` 返回 `1`，`program_id(axis=1)` 返回 `2`。你也可以使用 [`jax.experimental.pallas.num_programs()`](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.num_programs.html) 来获取给定轴的 grid 大小。

参见 [Grid 示例](https://docs.jax.dev/en/latest/pallas/quickstart.html#grids-by-example) 了解使用此 API 的简单内核。

## `BlockSpec`，又称如何切分输入

配合 `grid` 参数，我们需要向 Pallas 提供如何为每次调用切分输入的信息。具体来说，我们需要提供从 _循环迭代_ 到 _要操作的输入和输出的哪个块_ 的映射。这通过 [`jax.experimental.pallas.BlockSpec`](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.BlockSpec.html) 对象来提供。

在深入 `BlockSpec` 的细节之前，你可能想重新访问 Pallas 快速入门中的 [BlockSpec 示例](https://docs.jax.dev/en/latest/pallas/quickstart.html#pallas-block-specs-by-example)。

`BlockSpec` 通过 `in_specs` 和 `out_specs` 提供给 `pallas_call`，分别对应每个输入和每个输出各一个。

首先，我们讨论当 `indexing_mode == pl.Blocked()` 时 `BlockSpec` 的语义。

非正式地说，`BlockSpec` 的 `index_map` 接受调用索引作为参数（数量与 `grid` 元组的长度相同），并返回**块索引**（每个整体数组的轴对应一个块索引）。然后每个块索引乘以 `block_shape` 中对应的轴大小，得到对应数组轴上的实际元素索引。

> **注意**
>
> 并非所有块形状都受支持。
>
> - 在 TPU 上，仅支持秩至少为 1 的块。此外，块形状的最后两个维度必须等于整体数组的对应维度，或者分别能被 8 和 128 整除。对于秩为 1 的块，块维度必须等于数组维度，或者是 1024 的倍数，或者是 2 的幂且至少为 `128 * (32 / bitwidth(dtype))`。
>
> - 在 GPU 上，使用 Mosaic GPU 后端时，块的大小不受限制。然而，由于硬件限制，最次要数组维度的大小必须是 16 字节的倍数。例如，如果输入是 `jnp.float16`，则必须是 8 的倍数。
>
> - 在 GPU 上，使用 Triton 后端时，块本身的大小不受限制，但每个操作（包括加载或存储）必须操作大小为 2 的幂的数组。

如果块形状不能整除整体形状，则每个轴上最后一次迭代仍然会接收到 `block_shape` 大小的块引用，但越界的元素在输入时被填充，在输出时被丢弃。填充值是未指定的，你应该假设它们是垃圾值。在 `interpret=True` 模式下，我们对浮点值用 NaN 填充，以便用户有机会发现越界元素的访问，但不应依赖此行为。注意每个块中至少一个元素必须在界内。

更精确地说，对于形状为 `x_shape` 的输入 `x`，每个轴的切片按以下 `slice_for_invocation` 函数计算：

```python
>>> import jax
>>> from jax.experimental import pallas as pl
>>> def slices_for_invocation(x_shape: tuple[int, ...],
...                           x_spec: pl.BlockSpec,
...                           grid: tuple[int, ...],
...                           invocation_indices: tuple[int, ...]) -> tuple[slice, ...]:
...   assert len(invocation_indices) == len(grid)
...   assert all(0 <= i < grid_size for i, grid_size in zip(invocation_indices, grid))
...   block_indices = x_spec.index_map(*invocation_indices)
...   assert len(x_shape) == len(x_spec.block_shape) == len(block_indices)
...   elem_indices = []
...   for x_size, block_size, block_idx in zip(x_shape, x_spec.block_shape, block_indices):
...     start_idx = block_idx * block_size
...     # 块中至少一个元素必须在界内
...     assert start_idx < x_size
...     elem_indices.append(slice(start_idx, start_idx + block_size))
...   return elem_indices
```

例如：

```python
>>> slices_for_invocation(x_shape=(100, 100),
...                       x_spec = pl.BlockSpec((10, 20), lambda i, j: (i, j)),
...                       grid = (10, 5),
...                       invocation_indices = (2, 4))
[slice(20, 30, None), slice(80, 100, None)]

>>> # 相同形状的数组和块，但我们对每个块迭代 4 次
>>> slices_for_invocation(x_shape=(100, 100),
...                       x_spec = pl.BlockSpec((10, 20), lambda i, j, k: (i, j)),
...                       grid = (10, 5, 4),
...                       invocation_indices = (2, 4, 0))
[slice(20, 30, None), slice(80, 100, None)]

>>> # 块在第 2 轴上部分越界的示例。
>>> slices_for_invocation(x_shape=(100, 90),
...                       x_spec = pl.BlockSpec((10, 20), lambda i, j: (i, j)),
...                       grid = (10, 5),
...                       invocation_indices = (2, 4))
[slice(20, 30, None), slice(80, 100, None)]
```

下面定义的函数 `show_program_ids` 使用 Pallas 来显示调用索引。`iota_2D_kernel` 会用一个十进制数填充每个输出块，其中第一位数字代表第一个轴上的调用索引，第二位代表第二个轴上的调用索引：

```python
>>> def show_program_ids(x_shape, block_shape, grid,
...                      index_map=lambda i, j: (i, j)):
...   def program_ids_kernel(o_ref):  # 用 10*program_id(1) + program_id(0) 填充输出块
...     axes = 0
...     for axis in range(len(grid)):
...       axes += pl.program_id(axis) * 10**(len(grid) - 1 - axis)
...     o_ref[...] = jnp.full(o_ref.shape, axes)
...   res = pl.pallas_call(program_ids_kernel,
...                        out_shape=jax.ShapeDtypeStruct(x_shape, dtype=np.int32),
...                        grid=grid,
...                        in_specs=[],
...                        out_specs=pl.BlockSpec(block_shape, index_map),
...                        interpret=True)()
...   print(res)
```

例如：

```python
>>> show_program_ids(x_shape=(8, 6), block_shape=(2, 3), grid=(4, 2),
...                  index_map=lambda i, j: (i, j))
[[ 0  0  0  1  1  1]
 [ 0  0  0  1  1  1]
 [10 10 10 11 11 11]
 [10 10 10 11 11 11]
 [20 20 20 21 21 21]
 [20 20 20 21 21 21]
 [30 30 30 31 31 31]
 [30 30 30 31 31 31]]

>>> # 越界访问的示例
>>> show_program_ids(x_shape=(7, 5), block_shape=(2, 3), grid=(4, 2),
...                  index_map=lambda i, j: (i, j))
[[ 0  0  0  1  1]
 [ 0  0  0  1  1]
 [10 10 10 11 11]
 [10 10 10 11 11]
 [20 20 20 21 21]
 [20 20 20 21 21]
 [30 30 30 31 31]]

>>> # 形状允许小于 block_shape
>>> show_program_ids(x_shape=(1, 2), block_shape=(2, 3), grid=(1, 1),
...                  index_map=lambda i, j: (i, j))
[[0 0]]
```

当多次调用写入输出数组的相同元素时，结果取决于平台。

在下面的示例中，我们有一个 3D grid，其中最后一个 grid 维度不参与块选择（`index_map=lambda i, j, k: (i, j)`）。因此，我们对同一个输出块迭代 10 次。下面显示的输出是在 CPU 上使用 `interpret=True` 模式生成的，该模式目前按顺序执行调用。在 TPU 上，程序以并行和顺序的组合方式执行，此函数会生成所示的输出。参见 [值得注意的属性和限制](https://docs.jax.dev/en/latest/pallas/tpu/details.html#pallas-tpu-noteworthy-properties)。

```python
>>> show_program_ids(x_shape=(8, 6), block_shape=(2, 3), grid=(4, 2, 10),
...                  index_map=lambda i, j, k: (i, j))
[[  9   9   9  19  19  19]
 [  9   9   9  19  19  19]
 [109 109 109 119 119 119]
 [109 109 109 119 119 119]
 [209 209 209 219 219 219]
 [209 209 209 219 219 219]
 [309 309 309 319 319 319]
 [309 309 309 319 319 319]]
```

`block_shape` 中出现的 `None` 值作为维度值时行为类似于值 `1`，但对应的块轴会被压缩（你也可以传入 `pl.Squeezed()` 代替 `None`）。在下面的示例中，观察到当块形状被指定为 `(None, 2)` 时，`o_ref` 的形状是 `(2,)`（前导维度被压缩了）。

```python
>>> def kernel(o_ref):
...   assert o_ref.shape == (2,)
...   o_ref[...] = jnp.full((2,), 10 * pl.program_id(1) + pl.program_id(0))
>>> pl.pallas_call(kernel,
...                jax.ShapeDtypeStruct((3, 4), dtype=np.int32),
...                out_specs=pl.BlockSpec((None, 2), lambda i, j: (i, j)),
...                grid=(3, 2), interpret=True)()
Array([[ 0,  0, 10, 10],
       [ 1,  1, 11, 11],
       [ 2,  2, 12, 12]], dtype=int32)
```

当我们构造 `BlockSpec` 时，可以对 `block_shape` 参数使用值 `None`，此时整体数组的形状被用作 `block_shape`。如果对 `index_map` 参数使用值 `None`，则使用一个返回零元组的默认索引映射函数：`index_map=lambda *invocation_indices: (0,) * len(block_shape)`。

```python
>>> show_program_ids(x_shape=(4, 4), block_shape=None, grid=(2, 3),
...                  index_map=None)
[[12 12 12 12]
 [12 12 12 12]
 [12 12 12 12]
 [12 12 12 12]]

>>> show_program_ids(x_shape=(4, 4), block_shape=(4, 4), grid=(2, 3),
...                  index_map=None)
[[12 12 12 12]
 [12 12 12 12]
 [12 12 12 12]
 [12 12 12 12]]
```

### "元素"索引模式

上面记录的行为适用于默认的"分块"索引模式。当 `block_shape` 元组中使用整数时，例如 `(4, 8)`，它等价于传入 `pl.Blocked(block_size)` 对象，例如 `(pl.Blocked(4), pl.Blocked(8))`。分块索引模式意味着 `index_map` 返回的索引是 _块索引_。我们可以传入 `pl.Blocked` 以外的对象来改变 `index_map` 的语义，最值得注意的是 `pl.Element(block_size)`。当使用 `pl.Element` 索引模式时，索引映射函数返回的值直接用作数组索引，而不会先按块大小进行缩放。使用 `pl.Element` 模式时，你可以指定数组的虚拟填充为该维度的低-高填充元组：行为就像整体数组在输入时被填充了一样。在元素模式下不保证填充值，类似于当块形状不能整除整体数组形状时分块索引模式的填充值。

`Element` 模式目前仅在 TPU 上受支持。

```python
>>> # 不带填充的元素模式
>>> show_program_ids(x_shape=(8, 6), block_shape=(pl.Element(2), pl.Element(3)),
...                  grid=(4, 2),
...                  index_map=lambda i, j: (2*i, 3*j))
[[ 0  0  0  1  1  1]
 [ 0  0  0  1  1  1]
 [10 10 10 11 11 11]
 [10 10 10 11 11 11]
 [20 20 20 21 21 21]
 [20 20 20 21 21 21]
 [30 30 30 31 31 31]
 [30 30 30 31 31 31]]

>>> # 元素模式，首先用 1 行和 2 列填充数组。
>>> show_program_ids(x_shape=(7, 7),
...                  block_shape=(pl.Element(2, (1, 0)),
...                               pl.Element(3, (2, 0))),
...                  grid=(4, 3),
...                  index_map=lambda i, j: (2*i, 3*j))
[[ 0  1  1  1  2  2  2]
 [10 11 11 11 12 12 12]
 [10 11 11 11 12 12 12]
 [20 21 21 21 22 22 22]
 [20 21 21 21 22 22 22]
 [30 31 31 31 32 32 32]
 [30 31 31 31 32 32 32]]
```
