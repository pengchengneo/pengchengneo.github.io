# Pallas 核心级编程 (Core-specific Programming)

在本指南中，我们探讨如何使用 `pl.core_map` 编写 Pallas kernel。与 `pallas_call` 相比，`core_map` 具有以下几个关键特性：

- **核心级编程**：你为 TPU/GPU 的单个核心 (core) 编写代码，而不是为一个 JAX 设备编写。这使你可以完全控制每个核心上运行的内容，或者核心之间如何通信和分配工作。
- **集合通信 (Collectives)**：`core_map` 显式建模物理核心，因此可以安全地表达核心间通信。
- **平台通用**：`core_map` 编程模型适用于 TPU（TensorCore 和 SparseCore）以及 GPU，只需极少的样板代码修改。

本指南聚焦于 TPU。关于如何在 GPU 上使用 `core_map` 以获得更高的线程灵活性，请参阅 [Pallas GPU `core_map` 教程](https://docs.jax.dev/en/latest/pallas/gpu/reference.html#using-core-map)。

## 环境设置

现代加速器通常在一个设备下拥有多个核心。对于较新的 TPU 芯片（v4、v5p），每个 JAX 设备可能包含 2 个 TensorCore（即 [Megacore](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#chips)）。一些 TPU（v5p、v6e、7x）还包含 [SparseCore](https://openxla.org/xla/sparsecore#specifications_at_a_glance)，每个 SparseCore 由多个子核心 (subcore) 组成。

本指南在 v5p 芯片上编写，该芯片包含 4 个设备（每个设备 2 个 TensorCore）和 4 个 SparseCore（每个有 16 个子核心）。

```python
from functools import partial

import jax
from jax.sharding import NamedSharding
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import numpy as np


num_devices = jax.local_device_count()
assert num_devices > 1, "Please run this notebook with more than one device."

tpu_info = pltpu.get_tpu_info()  # 本 notebook 仅在 TPU 上运行
print(f"Running on {num_devices} TPU {tpu_info.chip_version} devices.")
```

```
Running on 4 TPU v5p devices.
```

除了典型的 TPU 设备 mesh 之外，你还需要创建一个核心的 mesh。可以将其视为在你使用的 4 设备 mesh 之上增加了一个名为 `core` 的维度，长度为 2，即总共 8 个核心。

```python
# 设备 mesh
mesh = jax.make_mesh((jax.device_count(),), ('device',))
print(mesh)

# JAX 设备内部的核心 mesh
tc_mesh = pltpu.create_tensorcore_mesh('core')
print(tc_mesh)

num_devices = mesh.size
num_cores = len(tc_mesh.devices)
print(f"There are {num_devices} devices, and {num_cores} cores each.")
```

```
Mesh('device': 4, axis_types=(Explicit,))
TensorCoreMesh(devices=array([TensorCore(id=0), TensorCore(id=1)], dtype=object), axis_names=('core',))
There are 4 devices, and 2 cores each.
```

## 一个简单的 per-core kernel

`pl.core_map` 允许你编写 per-core 的本地代码，就像 `jax.shard_map` 允许你编写 per-device 的代码一样。

在下面的示例 kernel 中，每个核心都有自己的 VMEM 和 semaphore 分配。与普通 kernel 一样，你可以使用 `pltpu.async_copy` 在 HBM 和 VMEM ref 之间发起数据拷贝。

**核心间通信**

在核心间通信之前，最佳实践是执行一次 barrier（使用 `pl.semaphore_signal`），以确保资源已分配且两个核心处于程序中的同一位置。

核心同步完成后，使用 `pltpu.make_async_remote_copy` 在核心之间发送数据。`device_id` 关键字参数通常允许向任何设备上的任何核心发送数据，但如果你只传入 `{'core': other_core_id}`，它将执行设备内的核心间拷贝（其他轴名保持不变）。

```python
# 此函数在每个核心上运行
def swap_cores_kernel(in_hbm, out_hbm,
                      in_vmem, scratch_vmem, out_vmem,
                      sem, send_sem, recv_sem):
  core_index = jax.lax.axis_index('core')
  num_cores = jax.lax.axis_size('core')
  slc_size = in_hbm.shape[-1] // num_cores
  slc = pl.ds(core_index * slc_size, slc_size)

  # 拷贝输入中与核心相关的切片
  pltpu.async_copy(in_hbm.at[:, slc], in_vmem, sem).wait()

  # barrier，确保所有核心都已进入 run_scoped
  # 如果不进行核心间通信，则不需要此步骤
  dst_core = (core_index + 1) % num_cores
  sem0 = pltpu.get_barrier_semaphore()
  pl.semaphore_signal(sem0, 1, device_id={'core': dst_core})
  pl.semaphore_wait(sem0, 1)

  # 在 core 0 和 core 1 之间交换数据
  the_copy = pltpu.make_async_remote_copy(
      in_vmem, scratch_vmem, send_sem, recv_sem, device_id={'core': dst_core},
  )
  the_copy.start()
  the_copy.wait()

  # 核心本地计算
  out_vmem[...] = scratch_vmem[...] * 2

  # 拷贝输出
  pltpu.async_copy(out_vmem, out_hbm.at[:, slc], sem).wait()
```

编写好本地 kernel 后：

- 在你的顶层 JAX 代码中以 HBM ref 开始，并在需要时分配输出 ref。
- 使用 `pl.core_map`（接受 TensorCore mesh）来开始 per-core 编程。
  - 你需要为 barrier semaphore 指定 `collective_id`。
- 在 `pl.core_map` 内部，调用 `pl.run_scoped` 来分配 per-core 的 scratch 空间（VMEM 和 semaphore）并运行本地 kernel。

```python
input_shape = (32, 256)
local_vmem_shape = (32 // num_devices, 256 // num_cores)
in_spec = jax.P('device', None)
sharding = NamedSharding(mesh, in_spec)

@jax.jit
@partial(jax.shard_map, mesh=mesh, in_specs=in_spec, out_specs=in_spec,
         check_vma=False)
def swap_cores(x):
  # 从输入和输出获取 buffer
  x_hbm_ref = jax.new_ref(x)
  o_hbm_ref = jax.new_ref(jax.lax.empty(x.shape, x.dtype))

  @pl.core_map(tc_mesh, compiler_params=pltpu.CompilerParams(collective_id=0))
  def _():
    pl.run_scoped(
        partial(swap_cores_kernel, x_hbm_ref, o_hbm_ref),
        *([pltpu.VMEM(local_vmem_shape, x.dtype)] * 3),  # VMEM 分配
        *([pltpu.SemaphoreType.DMA] * 3),                # semaphore
    )
  return o_hbm_ref[...]


x = jax.random.normal(jax.random.key(0), input_shape, jnp.float32)
x = jax.device_put(x, sharding)
y = swap_cores(x)

np.testing.assert_array_equal(y[:, 128:], x[:, :128] * 2)
np.testing.assert_array_equal(y[:, :128], x[:, 128:] * 2)
```

### 减少样板代码

你可以使用 `pl.kernel` 装饰器来封装 `core_map`、`run_scoped` 和输出 buffer 分配等样板代码。

注意，这应该在你可能拥有的任何顶层 `jax.shard_map` 内部运行。

```python
@jax.jit
@partial(jax.shard_map, mesh=mesh, in_specs=in_spec, out_specs=in_spec, check_vma=False)
def swap_cores(x):
  scratch_shapes = [pltpu.VMEM(local_vmem_shape, x.dtype)] * 3 + [pltpu.SemaphoreType.DMA] * 3
  return pl.kernel(swap_cores_kernel, out_shape=x, mesh=tc_mesh,
                   scratch_shapes=scratch_shapes,
                   compiler_params=pltpu.CompilerParams(collective_id=0))(x)

y = swap_cores(x)
np.testing.assert_array_equal(y[:, 128:], x[:, :128] * 2)
np.testing.assert_array_equal(y[:, :128], x[:, 128:] * 2)
```

## 在 `core_map` 中使用 Pipelining

注意，上面的 kernel 只进行了简单的拷贝和计算，没有通过 Pallas 的 `grid` 和 `BlockSpec` 实现自动 pipelining。要在 `core_map` 中实现 pipelining，请在核心本地 kernel 中使用 `pltpu.emit_pipeline`。

**在核心之间自动并行化工作**

简单的方式是将一个 block 轴标注为 `pltpu.PARALLEL`，Pallas 会自动沿该轴并行化工作。`pl.pallas_call` 和 `pltpu.emit_pipeline` 都支持此功能，分别通过 `core_axis` 和 `dimension_semantics` 参数。`pallas_call` 的示例在[另一份指南](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html#tpus-in-megacore-configuration)中，下面展示 `emit_pipeline` 的用法。

当提供 `PARALLEL` 标注时，对应的 grid 维度将被逻辑拆分并在不同核心上执行。（哪些 grid 维度在哪些核心上执行的确切语义是有保证的。）

**scratch shapes 分配**

注意，在下面的示例中，顶层的 `pl.run_scoped`（封装在 `kernel` 中）没有分配任何 VMEM scratch buffer。相反，`pltpu.emit_pipeline` 自行在 VMEM 中分配 scratch buffer 并用于其多级缓冲。

```python
def add_one_body(in_vmem, out_vmem):
  out_vmem[...] = in_vmem[...] + 1

input_shape = (1024, 1024)
in_spec = jax.P('device', None)

def add_one_kernel(x_hbm_ref, o_hbm_ref):
  in_shape = x_hbm_ref.shape
  pltpu.emit_pipeline(
      add_one_body,
      grid=(in_shape[0] // 8, in_shape[1] // 128),
      in_specs=[pl.BlockSpec(
          block_shape=(8, 128), index_map=lambda i, j: (i, j),
      )],
      out_specs=[pl.BlockSpec(
          block_shape=(8, 128), index_map=lambda i, j: (i, j),
      )],
      core_axis_name='core',
      dimension_semantics=(pltpu.PARALLEL, pltpu.ARBITRARY),
  )(x_hbm_ref, o_hbm_ref)


@jax.jit
@partial(jax.shard_map, mesh=mesh, in_specs=in_spec, out_specs=in_spec, check_vma=False)
def add_one(x):
  return pl.kernel(add_one_kernel, out_shape=x, mesh=tc_mesh, scratch_shapes=[])(x)


x = jax.random.normal(jax.random.key(0), input_shape, jnp.float32)
x = jax.device_put(x, NamedSharding(mesh, in_spec))
y = add_one(x)

np.testing.assert_array_equal(y, x + 1)
```

## Scalar Prefetch

下面的代码扩展了上面的 kernel，使用了 [scalar prefetch 和动态 block 索引](https://docs.jax.dev/en/latest/pallas/tpu/sparse.html) 来选择输入的特定子切片。

这涉及预分配一个 SMEM buffer（通过 `kernel` 内部的 `pl.run_scoped` 调用），并在 pipeline 启动前使用 `sync_copy` 填充该 buffer。在 `index_map` 中闭包捕获动态索引值即可使用。

**手动分配核心间的工作**

下面的代码示例还展示了 `core_map` 如何让你精确自定义工作在核心之间的分配方式，而无需依赖上面展示的自动 API。

为此，自定义你的 `index_map`，使用核心索引在不同核心上处理不同的切片。

```python
input_shape = (1024, 1024)
in_spec = jax.P('device', None)
output_shape = (1024, 512)

def indexed_add_one_kernel(in_refs, out_refs, i_smem_ref):
  (x_hbm_ref, i_hbm_ref), o_hbm_ref = in_refs, out_refs
  in_shape = x_hbm_ref.shape
  pltpu.sync_copy(i_hbm_ref, i_smem_ref)

  core_idx = jax.lax.axis_index('core')
  core_slc_size = in_shape[0] // num_cores
  i_map = lambda i: core_idx * core_slc_size // 8 + i  # 在核心之间分配工作
  j_map = lambda j: i_smem_ref[0] // 128 + j           # 使用预取的偏移量

  pltpu.emit_pipeline(
      add_one_body,
      grid=(core_slc_size // 8, output_shape[1] // 128),
      in_specs=[pl.BlockSpec(
          block_shape=(8, 128), index_map=lambda i, j: (i_map(i), j_map(j)),
      )],
      out_specs=[pl.BlockSpec(
          block_shape=(8, 128), index_map=lambda i, j: (i_map(i), j),
      )]
  )(x_hbm_ref, o_hbm_ref)


@jax.jit
@partial(jax.shard_map, mesh=mesh,
         in_specs=(in_spec, jax.P()), out_specs=in_spec, check_vma=False)
def indexed_add_one(x, index):
  out_shape = jax.ShapeDtypeStruct((x.shape[0], x.shape[1] // 2), x.dtype)
  return pl.kernel(indexed_add_one_kernel,
                   out_shape=out_shape, mesh=tc_mesh,
                   scratch_shapes=[pltpu.SMEM((1,), jnp.int32)])((x, index))


xs = jax.random.normal(jax.random.key(0), input_shape, jnp.float32)
xs = jax.device_put(xs, NamedSharding(mesh, in_spec))
idx = 256
y = indexed_add_one(xs, jnp.array([idx]))

np.testing.assert_array_equal(y, xs[:, idx:(idx+512)] + 1)
```

## 在 SparseCore 上映射

TPU v5p 包含 4 个 [SparseCore](https://openxla.org/xla/sparsecore)，它们专门用于稀疏内存访问和操作。本指南不会深入介绍 SparseCore 的全部功能，而是展示如何使用与 TensorCore 代码相同的语义和最小的修改在 SparseCore 上运行程序。

首先了解你芯片的基本 SparseCore 规格，并为向量操作创建一个 `VectorSubcoreMesh`。注意 TPU v5p 上每个 SparseCore 有 16 个（或其他数量的）子核心，`core_map` 将在每个子核心上 SPMD 运行你的代码。

```python
sc_info = pltpu.get_tpu_info().sparse_core
assert sc_info is not None
print(sc_info)

sc_mesh = plsc.VectorSubcoreMesh(
    core_axis_name="core", subcore_axis_name="subcore",
    num_cores=sc_info.num_cores
)
sc_num_cores = sc_info.num_cores
sc_num_subcores = sc_info.num_subcores
```

```
SparseCoreInfo(num_cores=4, num_subcores=16, num_lanes=8)
```

下面的代码与我们之前编写的 `add_one_kernel` 非常相似，但有几处不同：

1. 你需要在所有子核心之间分配工作，因此需要几行代码来计算每个子核心的特定切片。
2. SparseCore 寄存器计算允许更小的切片（int32 最大为 `4x16`），因此需要嵌套循环在计算阶段迭代该切片。

```python
input_shape = (4096, 128)
SC_REG_OP_SHAPE = (4, 16)

def sc_add_one_body(in_vmem, out_vmem):
  @pl.loop(0, in_vmem.shape[0], step=SC_REG_OP_SHAPE[0])
  def _reg_loop_0(c0):
    @pl.loop(0, in_vmem.shape[1], step=SC_REG_OP_SHAPE[1])
    def _reg_loop_1(c1):
      slc = (pl.ds(c0, SC_REG_OP_SHAPE[0]), pl.ds(c1, SC_REG_OP_SHAPE[1]))
      out_vmem[slc] = in_vmem[slc] + 1


def sc_add_one_kernel(x_hbm_ref, o_hbm_ref):
  in_shape = x_hbm_ref.shape
  core_idx = jax.lax.axis_index('core')
  subcore_idx = jax.lax.axis_index("subcore")
  cm_idx = core_idx * sc_num_subcores + subcore_idx  # core_map 上的索引
  slc_size = in_shape[0] // (sc_num_subcores * sc_num_cores)
  index_map = lambda i, j: (
      pl.ds(pl.multiple_of(cm_idx * slc_size + i * 8, 8), 8), j)

  pltpu.emit_pipeline(
      sc_add_one_body,
      grid=(slc_size // 8, in_shape[1] // 128),
      in_specs=[pl.BlockSpec(
          block_shape=(pl.BoundedSlice(8), 128), index_map=index_map,
      )],
      out_specs=[pl.BlockSpec(
          block_shape=(pl.BoundedSlice(8), 128), index_map=index_map,
      )]
  )(x_hbm_ref, o_hbm_ref)


@jax.jit
@partial(jax.shard_map, mesh=mesh, in_specs=in_spec, out_specs=in_spec, check_vma=False)
def sc_add_one(x):
  return pl.kernel(sc_add_one_kernel, out_shape=x, mesh=sc_mesh, scratch_shapes=[])(x)


x = jax.random.randint(jax.random.key(0), input_shape, 0, 64, jnp.int32)
x = jax.device_put(x, NamedSharding(mesh, in_spec))
y = sc_add_one(x)

np.testing.assert_array_equal(y, x + 1)
```
