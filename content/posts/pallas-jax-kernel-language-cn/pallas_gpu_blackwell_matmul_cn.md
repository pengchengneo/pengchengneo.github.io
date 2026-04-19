# 为 Blackwell 编写高性能矩阵乘法 kernel

在本指南中，我们将逐步迭代优化一个矩阵乘法 kernel。第一版实现会非常简单，速度也比较慢。然而，只需几个简单的步骤，就可以将其改进为一个 state-of-the-art 的 kernel，达到甚至超越 cuBLAS 和 CUTLASS 等高度优化实现的性能。

> **Warning**
>
> 下表中展示的利用率可能与你在网上看到的数据不同，但差异很可能是由于输入数据分布不同造成的。我们这里的所有 benchmark 使用的是元素为独立同分布正态分布的 float16 数组，这恰好是你能选择的最慢的分布之一。你可以通过运行[我们的测试文件](https://github.com/jax-ml/jax/blob/main/tests/pallas/mgpu_examples_test.py)（将 `BENCHMARK` 变量改为 `True`）来自行复现这些数据。

**tl;dr**：如果 matmul benchmark 不指定输入数据分布，就不要轻信。

| 实现 | TensorCore 利用率 | cuBLAS 利用率占比 |
|---|---|---|
| 0. 基础 kernel | 37.62% | 59.4% |
| 1. Warp specialization | 45.47% | 71.7% |
| 2. Tiled epilogue | 55.82% | 88.1% |
| 3. Collective (2CTA) MMA | 59.41% | 93.7% |
| 4. Persistent kernel | 61.46% | 97.0% |
| 5. 专用 epilogue warpgroup | 63.38% | 100.0% |
| 6. Grid tiling | 69.44% | 109.6% |
| cuBLAS | 63.38% | 100.0% |
| CUTLASS | 69.30% | 109.3% |

cuBLAS 基线通过测量 `jax.dot` 的性能获得。CUTLASS 性能通过以下 `cutlass_profiler` 调用中的最佳结果（排除稀疏 matmul）来测量：

```bash
cutlass_profiler --dist=gaussian,mean:0,stddev:1,scale:-1 --output=results.csv --accumulator-type=f32 --m=4096 --k=4096 --n=8192 --kernels='*sm100*' --A=f16 --B=f16 --C=void --D=f16
```

在每一步中，我们将展示 kernel 的完整实现，或者与上一步代码之间的差异。完整实现可以在[我们的测试文件](https://github.com/jax-ml/jax/blob/main/tests/pallas/mgpu_examples_test.py)中找到。你也可以在 [Pallas ops 包](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py)中找到完整的独立优化 kernel 实现。

## 0. 基础 kernel

我们从一个简单的单 CTA（block）、单 warpgroup 的例子开始。为了方便，我们将 kernel 的调优参数分离到一个单独的类中：

```python
@dataclasses.dataclass(frozen=True)
class TuningConfig:
  tile_m: int
  tile_n: int
  tile_k: int
  max_concurrent_steps: int
```

`tile_m`、`tile_n` 和 `tile_k` 指定了 pipeline 每一步中执行的 matmul 的大小。一般来说，`tile_k` 理想情况下应等于 128 除以输入元素类型的字节宽度。`max_concurrent_steps` 指定了计算/内存 pipeline 中内存预取的深度，在其他实现中通常被称为 stage 数量。

kernel 实现以一些设置代码开始：

```python
def matmul0(a, b, config: TuningConfig):
  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle)
  )
  if m % tile_m != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m=}")
  if n % tile_n != 0:
    raise ValueError(f"{n=} must be divisible by {tile_n=}")
  if k % tile_k != 0:
    raise ValueError(f"{k=} must be divisible by {tile_k=}")
  m_iters = m // tile_m
  n_iters = n // tile_n
  k_iters = k // tile_k
  max_concurrent_steps = config.max_concurrent_steps
```

我们解包配置变量以便于访问，设置 tiling 和 swizzling transform 使 SMEM 数据格式匹配 [MMA 指令所期望的格式](reference.html#memory-space-a-b-mma)。

kernel 实现本身相当简短。第一部分使用 [`plgpu.emit_pipeline`](../../_autosummary/jax.experimental.pallas.mosaic_gpu.emit_pipeline.html) 搭建一个[计算/内存 pipeline](pipelining.html)。在每一步中，计算函数（`do_mma`）消费一个 `(tile_m, tile_k)` 的 LHS 切片和一个 `(tile_k, tile_n)` 的 RHS 切片。如前所述，我们指定了 `transforms`，以及 `delay_release=1`。最后这个参数确保传入 `do_mma` 的输入窗口（`a_smem`、`b_smem`）至少在下一次 `do_mma` 调用完成之前不会被覆写。这是必要的，因为我们只在下一步等待上一步 MMA 的完成，这就是为什么 `arrive_barrier_slot` 和 `wait_barrier_slot` 在每次调用时在 0 和 1 之间交替。

```python
  def kernel(a_gmem, b_gmem, out_gmem, acc_tmem, acc_smem, consumed_barriers):
    mi = lax.axis_index("m")
    ni = lax.axis_index("n")
    m_slice = pl.ds(mi * tile_m, tile_m)
    n_slice = pl.ds(ni * tile_n, tile_n)

    def do_mma(idxs, a_smem, b_smem):
      (ki,) = idxs
      arrive_barrier_slot = ki % 2
      wait_barrier_slot = 1 - arrive_barrier_slot
      plgpu.tcgen05_mma(
          acc_tmem,
          a_smem,
          b_smem,
          barrier=consumed_barriers.at[arrive_barrier_slot],
          accumulate=(ki > 0),
      )
      plgpu.barrier_wait(consumed_barriers.at[wait_barrier_slot])

    # 确保第一次迭代中的 wait 能成功。
    plgpu.barrier_arrive(consumed_barriers.at[1])
    block_kwargs = dict(transforms=transforms, delay_release=1)
    plgpu.emit_pipeline(
      do_mma,
      in_specs=[
          plgpu.BlockSpec((tile_m, tile_k), lambda ki: (mi, ki), **block_kwargs),
          plgpu.BlockSpec((tile_k, tile_n), lambda ki: (ki, ni), **block_kwargs),
      ],
      grid=(k_iters,),
      max_concurrent_steps=max_concurrent_steps,
    )(a_gmem, b_gmem)
```

kernel 以一个 epilogue 结束。在做任何操作之前，我们先等待 pipeline 发出的最后一个 MMA 完成。然后，我们从 TMEM 加载最终累加器，将其写入 SMEM（[记得调用 `plgpu.commit_smem`](reference.html#commit-smem)），并使用 TMA 将其拷贝回 GMEM。

```python
  def kernel(...):
    ...  # 如上所述的计算 pipeline
    final_barrier = 1 - (k_iters % 2)
    plgpu.barrier_wait(consumed_barriers.at[final_barrier])
    acc_smem[...] = plgpu.async_load_tmem(acc_tmem).astype(dtype)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(acc_smem, out_gmem.at[m_slice, n_slice])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
```

剩下的就是把 kernel 转变为一个可以用 JAX 数组调用的函数。我们使用 [`plgpu.kernel`](../../_autosummary/jax.experimental.pallas.mosaic_gpu.kernel.html) 来完成。grid 目前只是简单的 2D，遍历输出 tile。我们分配 kernel 使用的中间变量：

1. 用作累加器的 TMEM buffer
2. 在拷贝到 GMEM 之前暂存累加器的 SMEM buffer
3. 用于等待 MMA 操作完成的 barrier

```python
def matmul0(a, b, config):
  ... # 第一个代码片段中的设置代码
  def kernel(...):
    ... # 整个 kernel 主体

  f = plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(m_iters, n_iters),
      grid_names=("m", "n"),
      scratch_shapes=dict(
        acc_tmem=plgpu.TMEM((tile_m, tile_n), jnp.float32),
        acc_smem=plgpu.SMEM((tile_m, tile_n), dtype, transforms=transforms),
        consumed_barriers=plgpu.Barrier(
          num_arrivals=1, num_barriers=2, orders_tensor_core=True
        ),
      )
  )
  return f(a, b)
```

除去设置代码，总共只有 50 行！遗憾的是它还不是很快，但已经达到了 cuBLAS 一半的利用率！

## 1. Warp specialization

> **Note**
>
> 回忆一下，在 Blackwell 上，一个 Pallas:MGPU 执行线程对应一个 CUDA lane/thread 的 warpgroup。

上面的 kernel 使用单个 warpgroup 完成所有工作：从取数据，到发出 MMA 操作，再到将结果存回 GMEM。虽然人们可能认为 TensorCore 执行的异步性应该允许我们将异步拷贝（TMA）和控制流的开销重叠起来，但实际情况似乎并非如此。

在 Hopper 一代 GPU 上，一个常见的解决方案是使用 _warpgroup_ specialization。在 Pallas 的术语中，`plgpu.kernel` 可以用 `num_threads=2` 来调用，这意味着 grid 中的每个 program 会产生两次对 body 的调用。然后通常使用 `lax.axis_index` 查询线程索引，并用来选择多个不同角色之一，例如 _只_ 发出异步拷贝或 _只_ 运行 MMA 操作。

这个方案在 Blackwell 一代也同样适用，而且实际上更简单。由于异步拷贝（TMA）和 `tcgen05` MMA 指令[都只需要一个 CUDA lane 来发射](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-issue-granularity)，我们甚至不需要使用多个 _warpgroup_。我们只需将一个 warpgroup 拆分为 _四个 warp_ 并进行特化即可！

在 Pallas 中，这可以通过使用 `pl.core_map` 配合 `plgpu.WarpMesh` 来实现。对于每个调用了这种 `core_map` 的 Pallas 线程，body 将被恰好调用四次。`core_map` 在进入和退出时都会同步所有 warp。注意，body 中只允许标量操作。

这将是我们在整个优化序列中对这个 kernel 进行的最大改写，因此我们将再次列出完整的 kernel 源代码。

```python
def matmul1(a, b, config: TuningConfig):
  ... # 设置代码保持不变

  def kernel(a_gmem, b_gmem, out_gmem,
             a_smem, b_smem, acc_tmem, acc_smem,
             load_barriers, consumed_barriers, mma_done_barrier):
    m_index = lax.axis_index("m")
    n_index = lax.axis_index("n")
    m_slice = pl.ds(m_index * tile_m, tile_m)
    n_slice = pl.ds(n_index * tile_n, tile_n)

    @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
    def _per_warp():
      warp_id = lax.axis_index("warp")

      @pl.when(warp_id == 0)
      def _memory():
        def _loop_body(ki, _):
          slot = lax.rem(ki, max_concurrent_steps)
          @pl.when(ki >= max_concurrent_steps)
          def _():  # 确保数据已被消费后再覆写。
            plgpu.barrier_wait(consumed_barriers.at[slot])
          k_slice = pl.ds(ki * tile_k, tile_k)
          plgpu.copy_gmem_to_smem(
              a_gmem.at[m_slice, k_slice], a_smem.at[slot], load_barriers.at[slot]
          )
          plgpu.copy_gmem_to_smem(
              b_gmem.at[k_slice, n_slice], b_smem.at[slot], load_barriers.at[slot]
          )

        lax.fori_loop(0, k_iters, _loop_body, None)

      @pl.when(warp_id == 1)
      def _compute():
        def _loop_body(ki, _):
          slot = lax.rem(ki, max_concurrent_steps)
          plgpu.barrier_wait(load_barriers.at[slot])  # 等待数据到达。
          plgpu.tcgen05_mma(
              acc_tmem,
              a_smem.at[slot],
              b_smem.at[slot],
              consumed_barriers.at[slot],
              accumulate=(ki > 0),
          )
        lax.fori_loop(0, k_iters, _loop_body, None)
        plgpu.tcgen05_commit_arrive(mma_done_barrier)

    plgpu.barrier_wait(mma_done_barrier)
    acc_smem[...] = plgpu.async_load_tmem(acc_tmem).astype(dtype)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(acc_smem, out_gmem.at[m_slice, n_slice])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
```

kernel 的结构与之前完全相同：先执行计算，然后是 epilogue。epilogue 保持不变（我们只使用了不同的 barrier 来等待完成），因此不再进一步讨论。

`plgpu.emit_pipeline` 调用和 `do_mma` 函数被替换为一个 `pl.core_map` 调用。你可以看到，在进入其 body 后，每个 Pallas 线程（现在代表一个 warp！）立即查明自己是四个线程中的哪一个。然后我们使用索引为 0 的线程 _只_ 发出异步拷贝来获取 MMA 操作数，而索引为 1 的线程进入另一个循环，在其中反复调用 `plgpu.tcgen05_mma`。

这里一个有趣的方面是同步机制。我们维护一个 `load_barriers` 数组，每个 barrier 跟踪一个待完成的 GMEM->SMEM 拷贝的进度。计算线程必须等待其完成后才能将相应的操作数送入 MMA 操作。反过来，负责异步拷贝的线程必须等待消费操作数的 MMA 完成后，才能通过发出另一个异步拷贝来覆写内存。这通过 `consumed_barriers` 来跟踪。最后，当计算线程完成所有 MMA 操作的发射后，它调用 `plgpu.tcgen05_commit_arrive(mma_done_barrier)`，请求 TensorCore 在所有 MMA 操作完成后完成 `mma_done_barrier`。

现在我们可以看看 `plgpu.kernel` 的定义。与前一版本的唯一区别是，我们显式分配了两个额外的 SMEM buffer 来保存 MMA 操作数（之前它们是由 `plgpu.emit_pipeline` 隐式分配的），以及额外的 barrier。注意 `load_barriers` 的 `num_arrivals=2`，因为我们在同一个 barrier 上发出两个异步拷贝。`orders_tensor_core` 需要在用于表示 TensorCore 操作完成的 barrier 上指定。

```python
def matmul1(a, b, config: TuningConfig):
  ... # 设置代码保持不变

  def kernel(...):
    ... # 上面的 kernel 代码

  f = plgpu.kernel(
      kernel,
      ...,  # 其他参数保持不变
      scratch_shapes=dict(
        a_smem=plgpu.SMEM(
            (max_concurrent_steps, tile_m, tile_k), dtype, transforms=transforms
        ),
        b_smem=plgpu.SMEM(
            (max_concurrent_steps, tile_k, tile_n), dtype, transforms=transforms
        ),
        acc_tmem=plgpu.TMEM((tile_m, tile_n), jnp.float32),
        acc_smem=plgpu.SMEM((tile_m, tile_n), dtype, transforms=transforms),
        load_barriers=plgpu.Barrier(
            num_arrivals=2, num_barriers=max_concurrent_steps
        ),
        consumed_barriers=plgpu.Barrier(
            num_arrivals=1,
            num_barriers=max_concurrent_steps,
            orders_tensor_core=True,
        ),
        mma_done_barrier=plgpu.Barrier(
            num_arrivals=1, num_barriers=1, orders_tensor_core=True
        ),
      )
  )
  return f(a, b)
```

这个相对简单的修改已经给了我们有意义的性能提升，达到了 cuBLAS 性能的近 70%。

## 2. Tiled epilogue

这次我们将注意力从 kernel 的计算部分转向其 epilogue。我们可以通过将 TMEM 到 SMEM 的拷贝与 SMEM 到 GMEM 的传输进行 pipeline 化来提高其效率。为此，我们修改 `scratch_shapes`，分配两个更小的 buffer 而不是一个能容纳整个输出的 SMEM 窗口（这也减少了我们的 SMEM 使用量）：

```python
def matmul2(a, b, config):
  ... # 设置代码和 kernel 代码
  f = plgpu.kernel(
      ...
      scratch_shapes=dict(
        ...
        # 之前: plgpu.SMEM((tile_m, tile_n), dtype, transforms=transforms),
        acc_smem=plgpu.SMEM(
            (2, tile_m, config.epilogue_tile_n), dtype, transforms=transforms
        ),
        ...
      )
  )
```

然后，在 kernel 中，我们以 `epilogue_tile_n` 为单位循环遍历输出列，并逐步将输出发送到 GMEM：

```python
def matmul2(a, b, config):
  ... # 设置代码保持不变

  def kernel(...):
    ... # 计算部分保持不变

    plgpu.barrier_wait(mma_done_barrier)
    out_gmem_window = out_gmem.at[m_slice, n_slice]
    for ni in range(tile_n // config.epilogue_tile_n):
      acc_smem_ni = acc_smem.at[ni % 2]
      ni_slice = pl.ds(ni * config.epilogue_tile_n, config.epilogue_tile_n)
      # 确保之前的拷贝完成后再覆写。
      plgpu.wait_smem_to_gmem(1, wait_read_only=True)
      acc_smem_ni[...] = plgpu.async_load_tmem(acc_tmem.at[:, ni_slice]).astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(acc_smem_ni, out_gmem_window.at[:, ni_slice])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
```

## 3. Collective (2CTA) MMA

如果你对我们最新的 kernel 进行 benchmark，你会很快发现它无法充分利用计算单元，因为它们一直在等待内存传送 MMA 操作数。这意味着我们的 kernel 是 memory bound 的，因为它的 _算术强度_ (arithmetic intensity) 太低：每加载一个字节执行的 flop 数太少。

Blackwell 架构中一个非常有效的技巧允许我们将算术强度翻倍，那就是 [collective MMA](reference.html#collective-mma)。核心思想非常简单：我们使用两个 block 的 cluster（在两个 SM 上）来计算一个 matmul。每个 block 只加载每个操作数的一半，但 MMA 操作在运行时会交换每个 block 的 SMEM 中的数据。

我们先从 kernel 配置的变化开始：

```python
def matmul3(a, b, config):
  ...  # 设置代码
  cluster_tile_m = 2 * tile_m
  cluster_tile_n = 2 * tile_n
  m_iters = m // cluster_tile_m
  n_iters = n // cluster_tile_n
  ... # 设置代码和 kernel

  f = plgpu.kernel(
      ...
      grid=(m_iters, n_iters),
      ...
      cluster=(2,),
      cluster_names=("cluster",),
      scratch_shapes=dict(
          ...
          # 之前: plgpu.TMEM((tile_m, tile_n), jnp.float32),
          acc_tmem=plgpu.TMEM(
              (tile_m, cluster_tile_n), jnp.float32, collective=True
          ),
          ...
      )
  )
```

我们向 `plgpu.kernel` 添加 `cluster` 参数，表示我们打算让成对的 program 协作（作为 CUDA block cluster）。我们还在 TMEM 分配中添加 `collective=True`，以确保它可以被 collective MMA 使用，并将其列数翻倍（到 `cluster_tile_n`）。

另一个值得注意的变化是，我们的一对 block 最终将计算一个 4 倍大的输出 tile，因此我们相应地缩小了 grid。

我们首先更新 kernel 的入口：

```python
  def kernel(...):
    is_lead_block = lax.axis_index("cluster") == 0
    m_index = lax.axis_index("m")
    n_index = lax.axis_index("n")
    m_slice = pl.ds(m_index * cluster_tile_m, cluster_tile_m)
    n_slice = pl.ds(n_index * cluster_tile_n, cluster_tile_n)
```

这里唯一的变化是我们使用 `cluster_tile_m` 和 `cluster_tile_n` 来计算两个 block 将共同计算的输出切片，同时我们还检查当前调用是否对应 cluster 中的第一个（leader）block。这很重要，因为 _只有 leader block 应该发射 MMA 指令_：

```python
    @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
    def _per_warp():
      warp_id = lax.axis_index("warp")

      @pl.when(warp_id == 0)
      def _memory():
        def _loop_body(ki, _):
          ...  # 等待数据被消费，如前所述。
          plgpu.copy_gmem_to_smem(
              ..., collective_axes="cluster", leader_tracked=plgpu.CopyPartition.PARTITIONED(0)
          )
          plgpu.copy_gmem_to_smem(
              ..., collective_axes="cluster", leader_tracked=plgpu.CopyPartition.PARTITIONED(1)
          )
        lax.fori_loop(0, k_iters, _loop_body, None)

      @pl.when(jnp.logical_and(warp_id == 1, is_lead_block))
      def _compute():
        def _loop_body(ki, _):
          ...  # 等待数据到达，如前所述。
          plgpu.tcgen05_mma(
              ...,
              collective_axis="cluster",
          )
        lax.fori_loop(0, k_iters, _loop_body, None)
        plgpu.tcgen05_commit_arrive(mma_done_barrier, collective_axis="cluster")
```

你可以看到这里有几个修改。首先，两个 block 都必须发出异步拷贝。在两个 block 中，我们都请求拷贝整个 cluster 的完整窗口，但添加的 `collective_axes="cluster"` 表示加载由两个 block 联合执行。`leader_tracked=CopyPartition.PARTITIONED(axis)` 指定操作数的哪个轴应在 cluster 中分割。我们分割 LHS 的行和 RHS 的列。

> **Warning**
>
> 分区的 collective 拷贝只在 cluster 的 leader block 中完成传入 `copy_gmem_to_smem` 的 barrier！这就是为什么你会看到 kernel 从不在第二个 block 中等待加载。

其次，如前所述，我们额外对 `_compute` body 添加谓词判断，使得只有 leader block 运行 MMA 指令。所有 `tcgen05` 调用还额外获得一个 `collective_axis=` 参数，以表示 MMA 的完成应该完成 cluster 中两个 block 的 barrier。

最后，我们对 epilogue 做了一个小修改。尽管 cluster 中的两个 block 共同计算了一个 `(cluster_tile_m, cluster_tile_n)` 形状的结果，每个单独的 block 只持有一个 `(tile_m, cluster_tile_n)` 形状的结果。我们修改输出切片代码以切出正确的 `out_gmem_window`：

```python
def matmul3(a, b, config):
  ...
  def kernel(...):
    ... # 计算

    plgpu.barrier_wait(mma_done_barrier)
    out_m_index = m_index * 2 + lax.axis_index("cluster")
    out_m_slice = pl.ds(out_m_index * tile_m, tile_m)
    out_gmem_window = out_gmem.at[out_m_slice, n_slice]
    for ni in range(cluster_tile_n // config.epilogue_tile_n):
      ...

  ...
```

## 4. Persistent kernel

下一步是将 kernel 变为 persistent 的。这意味着我们只启动 GPU 上实际可以并发运行的 cluster 数量（SM 数量除以 2），并让每个 cluster 循环处理固定数量的输出 tile。这种技术使我们能够更好地分摊 block（反）初始化成本（因为它们只在每个 SM 上执行一次），并实现 epilogue 中 SMEM 到 GMEM 拷贝与下一个输出 tile 计算的少量重叠。

```python
def matmul4(a, b, config):
  ...

  num_sms = jax.extend.backend.get_default_device().core_count
  f = plgpu.kernel(
      ...
      grid=(num_sms // 2,),
      grid_names=("cluster_grid",),
      ...
  )
```

修改相对简单。我们利用 [`plgpu.nd_loop`](../../_autosummary/jax.experimental.pallas.mosaic_gpu.nd_loop.html) 辅助函数来指定我们的迭代空间是 `(m_iters, n_iters)`，同时我们还通过 `collective_axes=` 参数请求将其在 cluster grid 上进行分割。

```python
def matmul4(a, b, config):
  ...

  def kernel(...):
    is_lead_block = lax.axis_index("cluster") == 0

    @plgpu.nd_loop((m_iters, n_iters), collective_axes="cluster_grid")
    def _mn_loop(loop_info: plgpu.NDLoopInfo):
      m_index, n_index = loop_info.index
      m_slice = ...
      n_slice = ...

      ...  # 计算 + epilogue
```

kernel body 的计算部分中唯一有意义的修改是确保 memory warp 中前几次对 `consumed_barriers` 的等待仅在处理第一个输出 tile 时跳过（由 `loop_info.local_index == 0` 指示）。处理第二个（或后续）tile 时，SMEM buffer 已被用于计算前一个输出 tile，因此我们需要确保这些计算已完成后再覆写它们：

```python
def matmul4(a, b, config):
  ...
  def kernel(...):
    ...
    def _mn_loop(...):
      ...

      @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
      def _per_warp():
        warp_id = lax.axis_index("warp")

        @pl.when(warp_id == 0)
        def _memory():
          def _loop_body(ki, _):
            slot = lax.rem(ki, max_concurrent_steps)
            @pl.when(jnp.logical_or(ki >= max_concurrent_steps, loop_info.local_index > 0))
            def _():  # 确保数据已被消费后再覆写。
              plgpu.barrier_wait(consumed_barriers.at[slot])
```

最后，我们通过追加一行来修改 kernel epilogue：

```python
def matmul4(a, b, config):
  ...
  def kernel(...):
    ...
    def _mn_loop(...):
      ...  # 计算 + epilogue
      plgpu.wait_load_tmem()  # TMEM load 必须在 MMA 覆写 TMEM 之前完成。
```

正如注释所示，由于 [TMEM load 是异步的](reference.html#tmem-loads)，我们必须在移动到下一个输出 tile 并通过发出另一个 MMA 来覆写我们的 TMEM 分配之前等待其完成。

## 5. 专用 epilogue warpgroup

虽然 persistence 本身就很有用，但它还解锁了另一个优化。当我们 kernel 中的单个 Pallas 线程完成计算部分后，它会执行整个 epilogue。然而，这意味着在 epilogue 完成之前，它无法为 TensorCore 发出更多工作！

这引出了一个简单的解决方案：使用 2 个 Pallas 线程（warpgroup）！第一个只专注于获取 MMA 操作数和发射 MMA 操作，而第二个只执行 epilogue！当然，为了使它们能够并发运行，我们需要双缓冲用于累加器的 TMEM，并使用额外的 barrier 来同步：

```python
def matmul5(a, b, config):
  ...

  f = plgpu.kernel(
      ...,
      num_threads=2,
      thread_name="wg",
      scratch_shapes=dict(
          ...
          # 之前: plgpu.TMEM((tile_m, cluster_tile_n), jnp.float32, collective=True),
          acc_tmem=plgpu.TMEM(
              (tile_m, 2 * cluster_tile_n), jnp.float32, collective=True
          ),
          ...
          # mma_done_barrier（现在 2 个 barrier）+ 新的 store_done_barrier（也是 2 个 barrier）
          # 之前: plgpu.Barrier(num_arrivals=1, num_barriers=1, orders_tensor_core=True),
          mma_done_barrier=plgpu.Barrier(
              num_arrivals=1, num_barriers=2, orders_tensor_core=True
          ),
          store_done_barrier=plgpu.ClusterBarrier(
              collective_axes=("cluster",),
              num_arrivals=1,
              num_barriers=2,
              orders_tensor_core=True,
          ),
      ),
  )
```

kernel 的开头与之前类似。我们将 `acc_tmem` 重命名为 `acc_tmem_slots`，并在遍历输出 tile 的循环中在其两半之间切换：

```python
def matmul(a, b, config):
  ...

  def kernel(a_gmem, b_gmem, out_gmem,
             a_smem, b_smem, acc_tmem_slots, acc_smem,
             load_barriers, consumed_barriers, mma_done_barrier, store_done_barrier):
    wg_idx = lax.axis_index("wg")
    is_lead_block = ...

    @plgpu.nd_loop(...)
    def _mn_loop(...):
      ...
      acc_slot = lax.rem(loop_info.local_index, jnp.int32(2))
      acc_tmem = acc_tmem_slots.at[:, pl.ds(acc_slot * cluster_tile_n, cluster_tile_n)]

      ...
```

计算部分额外添加了 `wg_idx == 0` 的谓词判断。barrier 的使用也有两个重要变化。首先，如果我们想要重用 TMEM 分配进行 MMA（只在 `loop_info.local_index >= 2` 时发生），我们需要在想要重用的 TMEM 半区上等待 `store_done_barrier`（由 `acc_slot` 指示）。其次，当我们想要请求 TensorCore 在完成时到达 `mma_done_barrier` 时，我们同样需要选择对应当前使用的 TMEM 半区的两个 barrier 之一。

> **Warning**
>
> 注意，尽管 cluster 中只有一个 block 发射 MMA，但两个 block 都会等待 `store_done_barrier`。这是必要的，因为在没有 `wait` 的情况下连续两次到达同一个 barrier 有时会触发硬件断言。

```python
def matmul(a, b, config):
  ...
  def kernel(...):
    ...
    def _mn_loop(...):
      acc_slot = ...
      acc_tmem = ...

      @pl.when(wg_idx == 0)
      def _compute_wg():
        @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
        def _per_warp():
          warp_id = lax.axis_index("warp")

          @pl.when(warp_id == 0)
          def _memory():
            ... # 内存代码保持不变

          # 等待 store 完成（前两步除外）。
          @pl.when(jnp.logical_and(warp_id == 1, loop_info.local_index >= 2))
          def _wait_store():
            plgpu.barrier_wait(store_done_barrier.at[acc_slot])
          @pl.when(jnp.logical_and(warp_id == 1, is_lead_block))
          def _compute():
            ... # 计算循环保持不变
            plgpu.tcgen05_commit_arrive(mma_done_barrier.at[acc_slot], collective_axis="cluster")
```

最后，我们修改 epilogue，使得只有第二个 warpgroup 执行它，并让该 warpgroup 通过到达与其使用的 TMEM 半区关联的 `store_done_barrier` 来信号 store 的完成。

```python
def matmul(a, b, config):
  ...
  def kernel(...):
    ...
    def _mn_loop(...):
      ... # 计算

      @pl.when(wg_idx == 1)
      def _store_wg():
        ... # 未修改的 epilogue
        plgpu.wait_load_tmem()  # TMEM load 必须在我们发出信号之前完成。
        plgpu.barrier_arrive(store_done_barrier.at[acc_slot])
```

## 6. Grid tiling

我们对这个 kernel 的最后一个改动是改变生成输出 block 的顺序，以更好地利用 L2 cache。如前所述，计算单元相对于内存系统来说速度极快，因此我们需要尽可能地利用一切手段来让它们保持忙碌。

> **Note**
>
> 这个技巧有很多不同的名称。CUTLASS 称之为 "rasterization order"，ThunderKittens 称之为 "supergrouping"，而 Triton 教程称之为 "program re-ordering"。我们使用 "grid tiling" 这个名称。

我们的策略受 CUTLASS 启发，工作方式如下。首先，你选择迭代空间中两个维度中哪个是变化更快的（我们称之为 `grid_minor_dim`）。然后，你选择沿该维度的 tile 大小（`grid_tile_width`）。我们不是在递增更主要的索引之前遍历 grid 的整个次要维度，而是每次遍历 `grid_tile_width` 个元素后就做一次。一旦元素用完，我们就移动到下一个 tile。但有一个巧妙之处！我们不是跳到第二个 tile 的开头，而是从末尾开始向回工作。这确保了在切换 tile 时，我们可以重用其中一个操作数的一些近期 block。

由于这种策略非常常见，我们提供了一个辅助函数：[`plgpu.planar_snake`](../../_autosummary/jax.experimental.pallas.mosaic_gpu.planar_snake.html)。使用该辅助函数时，kernel 的改动非常简单：

```python
def matmul(a, b, config):
  ...
  def kernel(...):
    ...
    # 我们现在只遍历一个 1D 循环（但仍然在 cluster 间分割它）。
    @plgpu.nd_loop((m_iters * n_iters,), collective_axes="cluster_grid")
    def _mn_loop(loop_info: plgpu.NDLoopInfo):
      (lin_idx,) = loop_info.index
      m_index, n_index = plgpu.planar_snake(
          lin_idx,  # 线性索引。
          (m_iters, n_iters),  # 2D 迭代空间。
          config.grid_minor_dim,  # 0 或 1，表示最快变化的维度。
          config.grid_tile_width,  # 沿最快变化维度的 tile 宽度。
      )
      ... # 其余代码保持不变
```

这个简单的技巧 _效果惊人_，是达到 state of the art 性能的关键。

## 最终 kernel

恭喜你完成了本教程！在前面的章节中，我们只关注不同 kernel 之间的差异，很少列出完整的源代码。这在扩展实现时隐藏了无关细节，但看到完整源代码也会有帮助。所以这就是了！整个实现不到 150 行，就能达到 SOTA 性能（至少在我们 benchmark 使用的 shape 上）。

```python
def matmul6(a, b, config: TuningConfig):
  dtype = a.dtype
  m, k = a.shape
  _, n = b.shape
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  swizzle = plgpu.find_swizzle(tile_k * jnp.dtype(dtype).itemsize * 8)
  swizzle_elems = swizzle // jnp.dtype(dtype).itemsize
  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle)
  )
  if m % tile_m != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m=}")
  if n % tile_n != 0:
    raise ValueError(f"{n=} must be divisible by {tile_n=}")
  if k % tile_k != 0:
    raise ValueError(f"{k=} must be divisible by {tile_k=}")
  cluster_tile_m = 2 * tile_m
  cluster_tile_n = 2 * tile_n
  m_iters = m // cluster_tile_m
  n_iters = n // cluster_tile_n
  k_iters = k // tile_k
  max_concurrent_steps = config.max_concurrent_steps

  def kernel(a_gmem, b_gmem, out_gmem,
             a_smem, b_smem, acc_tmem, acc_smem,
             load_barriers, consumed_barriers, mma_done_barrier, store_done_barrier):
    wg_idx = lax.axis_index("wg")
    is_lead_block = lax.axis_index("cluster") == 0

    @plgpu.nd_loop((m_iters * n_iters,), collective_axes="cluster_grid")
    def _mn_loop(loop_info: plgpu.NDLoopInfo):
      (lin_idx,) = loop_info.index
      m_index, n_index = plgpu.planar_snake(
          lin_idx,
          (m_iters, n_iters),
          config.grid_minor_dim,
          config.grid_tile_width,
      )
      m_slice = pl.ds(m_index * cluster_tile_m, cluster_tile_m)
      n_slice = pl.ds(n_index * cluster_tile_n, cluster_tile_n)
      acc_slot = lax.rem(loop_info.local_index, jnp.int32(2))
      mn_acc_tmem = acc_tmem.at[:, pl.ds(acc_slot * cluster_tile_n, cluster_tile_n)]

      @pl.when(wg_idx == 0)
      def _compute_wg():
        @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
        def _per_warp():
          warp_id = lax.axis_index("warp")

          @pl.when(warp_id == 0)
          def _memory():
            def _loop_body(ki, _):
              slot = lax.rem(ki, max_concurrent_steps)
              @pl.when(jnp.logical_or(ki >= max_concurrent_steps, loop_info.local_index > 0))
              def _():  # 确保数据已被消费后再覆写。
                plgpu.barrier_wait(consumed_barriers.at[slot])
              k_slice = pl.ds(ki * tile_k, tile_k)
              plgpu.copy_gmem_to_smem(
                  a_gmem.at[m_slice, k_slice], a_smem.at[slot], load_barriers.at[slot],
                  collective_axes="cluster", leader_tracked=plgpu.CopyPartition.PARTITIONED(0)
              )
              plgpu.copy_gmem_to_smem(
                  b_gmem.at[k_slice, n_slice], b_smem.at[slot], load_barriers.at[slot],
                  collective_axes="cluster", leader_tracked=plgpu.CopyPartition.PARTITIONED(1)
              )

            lax.fori_loop(0, k_iters, _loop_body, None)

          # 等待 store 完成（前两步除外）。
          @pl.when(jnp.logical_and(warp_id == 1, loop_info.local_index >= 2))
          def _wait_store():
            plgpu.barrier_wait(store_done_barrier.at[acc_slot])
          @pl.when(jnp.logical_and(warp_id == 1, is_lead_block))
          def _compute():
            def _loop_body(ki, _):
              slot = lax.rem(ki, max_concurrent_steps)
              plgpu.barrier_wait(load_barriers.at[slot])  # 等待数据到达。
              plgpu.tcgen05_mma(
                  mn_acc_tmem,
                  a_smem.at[slot],
                  b_smem.at[slot],
                  consumed_barriers.at[slot],
                  accumulate=(ki > 0),
                  collective_axis="cluster",
              )
            lax.fori_loop(0, k_iters, _loop_body, None)
            plgpu.tcgen05_commit_arrive(
                mma_done_barrier.at[acc_slot],
                collective_axis="cluster",
            )

      @pl.when(wg_idx == 1)
      def _store_wg():
        # 确保前一个 mn 步骤的拷贝已完成。
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        plgpu.barrier_wait(mma_done_barrier.at[acc_slot])
        out_m_index = m_index * 2 + lax.axis_index("cluster")
        out_m_slice = pl.ds(out_m_index * tile_m, tile_m)
        out_gmem_window = out_gmem.at[out_m_slice, n_slice]
        for ni in range(cluster_tile_n // config.epilogue_tile_n):
          acc_smem_ni = acc_smem.at[ni % 2]
          ni_slice = pl.ds(ni * config.epilogue_tile_n, config.epilogue_tile_n)
          # 确保之前的拷贝完成后再覆写。
          plgpu.wait_smem_to_gmem(1, wait_read_only=True)
          acc_smem_ni[...] = plgpu.async_load_tmem(mn_acc_tmem.at[:, ni_slice]).astype(dtype)
          plgpu.commit_smem()
          plgpu.copy_smem_to_gmem(acc_smem_ni, out_gmem_window.at[:, ni_slice])
        plgpu.wait_load_tmem()  # TMEM load 必须在我们发出信号之前完成。
        plgpu.barrier_arrive(store_done_barrier.at[acc_slot])
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  num_sms = backend.get_default_device().core_count
  f = plgpu.kernel(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), dtype),
      grid=(num_sms // 2,),
      grid_names=("cluster_grid",),
      cluster=(2,),
      cluster_names=("cluster",),
      num_threads=2,
      thread_name="wg",
      scratch_shapes=dict(
          a_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_m, tile_k), dtype, transforms=transforms
          ),
          b_smem=plgpu.SMEM(
              (max_concurrent_steps, tile_k, tile_n), dtype, transforms=transforms
          ),
          acc_tmem=plgpu.TMEM(
              (tile_m, 2 * cluster_tile_n), jnp.float32, collective=True
          ),
          acc_smem=plgpu.SMEM(
              (2, tile_m, config.epilogue_tile_n), dtype, transforms=transforms
          ),
          load_barriers=plgpu.Barrier(
              num_arrivals=2, num_barriers=max_concurrent_steps
          ),
          consumed_barriers=plgpu.Barrier(
              num_arrivals=1,
              num_barriers=max_concurrent_steps,
              orders_tensor_core=True,
          ),
          mma_done_barrier=plgpu.Barrier(
              num_arrivals=1, num_barriers=2, orders_tensor_core=True
          ),
          store_done_barrier=plgpu.ClusterBarrier(
              collective_axes=("cluster",),
              num_arrivals=1,
              num_barriers=2,
              orders_tensor_core=True,
          ),
      )
  )
  return f(a, b)
```
