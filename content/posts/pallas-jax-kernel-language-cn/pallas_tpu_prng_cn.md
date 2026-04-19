# 伪随机数生成

Pallas TPU 实现了多种在 kernel 内部生成伪随机数的 API，它们在可移植性和效率之间有不同的权衡。如果需要最大的可移植性，可以直接使用 `jax.random` 函数。Pallas 还暴露了 TPU 上内置的硬件 PRNG，其计算速度最快，但底层实现可能因硬件代次不同而有所变化。

## 使用 `jax.random` API

Pallas 支持 `jax.random` API 中的一部分操作。这些函数保证在给定相同 key 的情况下，与在 Pallas 外部调用 JAX 中的相同函数产生逐比特相同的结果。仅支持 `threefry2x32` key。

当前支持的随机采样函数：

- `jax.random.bits()`
- `jax.random.uniform()`
- `jax.random.bernoulli()`
- `jax.random.normal()`

支持的工具函数：

- `jax.random.key()`
- `jax.random.fold_in()`
- `jax.random.wrap_key_data()`

PRNG key 可以在 kernel 内部通过 `jax.random.key()` 生成。但更常见的场景是从调用方传入 key。这种情况下，可以通过 VMEM 将 key 传入 kernel，如下所示：

```python
def body(key_ref, o_ref):
  key = key_ref[...]
  o_ref[...] = jax_random.uniform(
      key, shape=o_ref[...].shape, minval=0.0, maxval=1.0
  )

threefry_key = jax_random.key(0, impl="threefry2x32")

# 在 kernel 外部生成 threefry key，并通过 VMEM 传入。
result = pl.pallas_call(
    body,
    in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
    out_shape=jax.ShapeDtypeStruct((256, 256), jnp.float32)
)(threefry_key)
```

> **注意**
>
> 关于性能方面，在 kernel 内部生成随机数有助于减少显存带宽占用，因为传入一个 key 比传入一个大的随机数数组更廉价。然而，`threefry2x32` 是一个向量密集型算法，涉及数十个链式位运算操作。这可能成为瓶颈，导致加速器利用率低下，因为它不使用矩阵乘法单元（MXU），而大部分 FLOP/s 都集中在 MXU 上。

## 使用硬件 PRNG

TPU 在硬件中原生实现了一个顺序式（而非基于计数器的）PRNG，其计算速度远快于 `threefry2x32` 等软件实现的 PRNG。然而，JAX 的随机 API 假设使用无状态的、基于计数器的 PRNG，因此 Pallas 引入了自己的有状态 PRNG API 来提供等价功能。

> **警告**
>
> 硬件 PRNG 的底层实现因 TPU 代次不同而有所变化，因此最佳实践是不要依赖其确切行为。如需更稳定的软件实现 PRNG，建议使用 `threefry2x32` 实现。

### 有状态随机数生成

使用 Pallas PRNG 的有状态模式是生成随机数最原生、最高效的方法。首先，应使用 `pltpu.prng_seed(N)` 设置 PRNG 种子，其中 N 是一个整数种子。

之后，可以调用任意数量的有状态采样函数，它们等价于对应的 JAX 版本，只是缺少 `key` 参数：

- `pltpu.stateful_uniform`：`jax.random.uniform()` 的有状态等价物
- `pltpu.stateful_normal`：`jax.random.normal()` 的有状态等价物
- `pltpu.stateful_bernoulli`：`jax.random.bernoulli()` 的有状态等价物

生成任何随机数都会更新 PRNG 的内部状态，后续调用将生成不同的数。与 JAX 不同，无需 `split` 或 `fold_in` key 再传入采样函数。

例如，以下 kernel 生成一组 0 到 1 之间的均匀分布随机数：

```python
from jax.experimental.pallas import tpu as pltpu

def kernel_body(o_ref):
  pltpu.prng_seed(0)
  o_ref[...] = pltpu.stateful_uniform(shape=o_ref.shape, minval=0.0, maxval=1.0)

pl.pallas_call(kernel_body,
               out_shape=jax.ShapeDtypeStruct((256, 256), jnp.float32))
```

注意，在带有 grid 的 kernel 中，种子应仅在第一次迭代时设置，否则由于重置种子，每个 program instance 中生成的随机数将完全相同。

### 无状态生成

Pallas 提供了一个介于前述有状态 API 和无状态 `jax.random` API 之间的中间 API，允许以无状态方式使用硬件 PRNG。方法是通过 `pltpu.to_pallas_key(key)` 将 JAX key 转换为特殊的 Pallas 类型 key，并通过 SMEM 将此 key 传入 kernel。在 kernel 内部解引用该 key 后，可以将其传入 `jax.random` 支持的采样函数来生成随机数。与有状态 API 相比，每次调用随机数生成器时都有计算和设置种子的额外开销。

例如，以下 kernel 使用硬件 PRNG 生成均匀分布随机数：

```python
def body(key_ref, o_ref):
  o_ref[...] = jax.random.uniform(
      key_ref[...], shape=o_ref[...].shape
  )

rbg_key = jax_random.key(0, impl="threefry2x32")
key = pltpu.to_pallas_key(rbg_key)
o_shape = jax.ShapeDtypeStruct((8, 128), dtype)
result = pl.pallas_call(
    body,
    in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)],
    out_shape=o_shape,
)(key)
```

对于带有 grid 的大型 kernel，可以对 `program_id` 使用 `jax.random.fold_in()` 来为每个 program instance 生成唯一的 key。

## 块不变采样（Block-invariant sampling）

块不变采样是一种以块为单位生成随机数的方法，其结果不受块大小和迭代顺序的影响。例如，你可能希望在两个 kernel（如前向传播和反向传播）之间生成完全相同的随机数集合，但这两个 kernel 在调优后可能使用不同的块大小。

Pallas 提供了一个辅助函数 `pltpu.sample_block`，可以保证在不同的块大小和 grid 设置下生成完全相同的随机数。第一步是选择一个 `tile_size`，它必须能整除所有你希望保持不变性的块大小。例如，`tile_size=(16, 128)` 适用于块大小 `(32, 128)` 和 `(16, 256)`。tile 越大，采样过程越高效，因此所有可能块大小的最大公约数是最佳选择。

接下来，使用以下参数调用 `pltpu.sample_block`：

```python
pltpu.sample_block(
  sampler_function,  # 一个 JAX 随机函数，例如 `jax.random.uniform`。
  global_key,  # 所有块共享的全局 key。
  block_size,  # 要生成的本地块大小。
  tile_size,  # tile 大小。
  total_size,  # 所有块合在一起的生成数组总形状。
  block_index,  # 在 total_size 中的块索引。通常是当前的 program instance。
  **sampler_kwargs  # 传递给 sampler_function 的关键字参数。
)
```

例如，以下代码片段在 `(16, 128)` 块大小和 `(32, 256)` 块大小（带转置 grid 迭代顺序）下生成完全相同的随机数：

```python
def make_kernel_body(index_map):
  def body(key_ref, o_ref):
    key = key_ref[...]
    samples = pltpu.sample_block(
        jax.random.uniform,
        key,
        block_size=o_ref[...].shape,
        tile_size=(16, 128),
        total_size=(64, 512),
        block_index=index_map(pl.program_id(0), pl.program_id(1)),
        minval=0.0,
        maxval=1.0)
    o_ref[...] = samples
  return body

global_key = pltpu.to_pallas_key(jax_random.key(0))
o_shape = jnp.ones((64, 512), dtype=jnp.float32)
key_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
out_spec = pl.BlockSpec((16, 128), lambda i, j: (i, j))
result_16x128 = pl.pallas_call(
    make_kernel_body(index_map=lambda i, j: (i, j)),
    out_shape=o_shape,
    in_specs=[key_spec],
    out_specs=out_spec,
    grid=(4, 4),
)(global_key)

out_spec = pl.BlockSpec((32, 256), lambda i, j: (j, i))
result_32x256_transposed = pl.pallas_call(
    make_kernel_body(index_map=lambda i, j: (j, i)),
    in_specs=[key_spec],
    out_shape=o_shape,
    out_specs=out_spec,
    grid=(2, 2),
)(global_key)
```
