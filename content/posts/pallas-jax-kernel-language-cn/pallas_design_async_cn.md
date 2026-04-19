# Pallas 异步操作

## 背景与动机

我们希望在 Pallas 中暴露 API，以便在**多个 kernel 之间**显式地重叠计算和通信。

### XLA Async Decomposition

作为动机，考虑以下 JAX 伪代码：

```python
def f(x):
  y = ppermute(x)
  z = x + 1
  return y, z
```

在这个函数中，我们可以同时执行 `ppermute` 和 `x + 1`。这是 XLA 自动完成的一项优化，方法是：

1. 将 `ppermute` 分解为 `ppermute_start` 和 `ppermute_done` 两个操作，它们通过一个 future 连接。
2. 将 `x + 1` 调度到 `ppermute_start` 和 `ppermute_done` 之间。

结果程序如下：

```python
def f(x):
  fut = ppermute_start(x)
  z = x + 1  # 与 ppermute 同时执行
  y = ppermute_done(fut)
  return y, z
```

### Kernel 内部的异步操作

现在假设我们不使用 XLA 的 `ppermute`，而是有自己的自定义 Pallas `ppermute`。

```python
def ppermute_kernel(x_ref, y_ref, send_sem, recv_sem):
  right_neighbor = ...
  descriptor = pltpu.make_async_remote_copy(x_ref, y_ref, send_sem, recv_sem, device_id=right_neighbor)
  descriptor.start()
  descriptor.wait_send()
  descriptor.wait_recv()

def ppermute(x):
  return pl.pallas_call(ppermute_kernel, out_shape=x, ...)(x)
```

目前，我们无法像 XLA 那样将 `ppermute` 分解为 `start/done` 对，因此我们改为将 `x + 1` 显式**融合**到 kernel 中。

```python
def add_one(x_ref, z_ref):
  z_ref[...] = x_ref[...] + 1

def ppermute_add_one_kernel(x_ref, y_ref, z_ref, send_sem, recv_sem):
  right_neighbor = ...
  descriptor = pltpu.make_async_remote_copy(x_ref, y_ref, send_sem, recv_sem, device_id=right_neighbor)
  descriptor.start()

  # 在 start/wait 之间显式调度内部 kernel
  pltpu.emit_pipeline(add_one)(x_ref, z_ref)

  descriptor.wait_send()
  descriptor.wait_recv()

def ppermute_and_add_one(x):
  return pl.pallas_call(ppermute_add_one_kernel, out_shape=(x, x), ...)(x)
```

我们的目标是能够分别编写启动 `ppermute` 和等待其完成的 kernel，这样就可以在它们之间使用普通的 `x + 1`（或任何我们想要的计算）。这使代码更具可读性、更易维护、更不容易出错。

## 如何实现分解的 Pallas 异步操作（在 TPU 上）？

实现分解异步操作时，最关键的问题是确定在两个操作之间传递的 `future` 包含什么。具体来说，它必须包含关于后台操作的一些重要状态信息。

查看 Pallas 代码可以发现，我们需要一个 "descriptor" 来启动和等待 remote copy。我们能否将这个 descriptor 从 Pallas kernel 中传出，然后传入另一个 kernel？大致上可以。底层 TPU 硬件通过一对 semaphore 来跟踪异步操作的进度：`send_sem` 让我们能够等待设备何时完成向邻居发送数据，`recv_sem` 跟踪从邻居发送到设备的数据传输。如果我们设想编写一个 start kernel 和一个 done kernel，从 start 传递到 done 的只需要 semaphore 以及关于在 semaphore 上等待多少的一些信息。

我们可以通过扩展 Pallas 来支持从 kernel 返回 semaphore 来实现这一点。

```python
def ppermute_start_kernel(
    in_ref, send_sem, recv_sem, out_ref, *, axis_name,
):
  axis_size = jax.lax.psum(1, axis_name)
  left_neighbor = jax.lax.rem(
      jax.lax.axis_index(axis_name) - 1 + axis_size, axis_size
  )
  right_neighbor = jax.lax.rem(jax.lax.axis_index(axis_name) + 1, axis_size)
  barrier_sem = pltpu.get_barrier_semaphore()
  pltpu.semaphore_signal(barrier_sem, device_id=left_neighbor)
  pltpu.semaphore_wait(barrier_sem, 1)
  pltpu.make_async_remote_copy(
      in_ref, out_ref, send_sem, recv_sem, device_id=right_neighbor
  ).start()

def ppermute_start(x, *, axis_name) -> tuple[Semaphore, Semaphore, Array]:
  send_sem, recv_sem, out = pl.pallas_call(
      functools.partial(ppermute_start_kernel, axis_name=axis_name),
      out_shape=(
          pltpu.SemaphoreType.DMA(()),
          pltpu.SemaphoreType.DMA(()),
          jax.ShapeDtypeStruct(
              x.shape,
              dtype=x.dtype,
          ),
      ),
      in_specs=[
          pl.BlockSpec(memory_space=pl.ANY),
      ],
      out_specs=(
          pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          pl.BlockSpec(memory_space=pl.ANY),
      ),
  )(x)
  return send_sem, recv_sem, out
```

注意这里有一个微妙的地方。Pallas 告诉 XLA 它希望某些输出是 semaphore（也称为 sync flag），XLA 会将它们视为"已预留"（即当它们在 XLA 程序中存活期间，这些 sync flag 不能被其他 kernel 分配）。它们的行为类似于 barrier semaphore，后者是由 XLA 管理的预留 semaphore。

另一个需要注意的地方是，我们在 start kernel 中返回了输出 buffer `out`——**而此时它正在被活跃地写入**。

现在我们编写执行阻塞操作的 `done` kernel。我们将 `out` 传入 kernel 以计算在 semaphore 上阻塞所需的 shape。

```python
def ppermute_done_kernel(ref, send_sem, recv_sem, _):
  pltpu.make_async_copy(ref, ref, send_sem).wait()
  pltpu.make_async_copy(ref, ref, recv_sem).wait()

def ppermute_done(send_sem, recv_sem, out) -> Array:
  out = pl.pallas_call(
      ppermute_done_kernel,
      out_shape=(
          jax.ShapeDtypeStruct(
              out.shape,
              dtype=out.dtype,
          ),
      ),
      in_specs=[
          pl.BlockSpec(memory_space=pl.ANY),
          pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
      ],
      out_specs=pl.BlockSpec(memory_space=pl.ANY),
      input_output_aliases={0:0}
  )(out, send_sem, recv_sem)
  return out
```

注意：我们在这里对输出 buffer 做了 i/o alias，以保证消费者位于 `ppermute_done` 的下游。

现在我们可以实现分解后的 collective permute 了。

```python
def f(x):
  fut = ppermute_start(x)
  z = x + 1  # 与 ppermute 同时执行
  y = ppermute_done(fut)
  return y, z
```

***是这样吗？***

## 为什么这样不行？

这种方案还有三个遗留问题，每个问题在 Pallas 之外也都一定程度上存在。以下是概述：

1. **调度（Scheduling）**——仅仅因为我们先写了 `ppermute_start`，再写 `x + 1`，最后写 `ppermute_done`，并不能保证它们会按该顺序执行。XLA 负责调度，因此当我们编写 JAX 程序时，我们建立的是 XLA 会尊重的数据依赖关系，但 XLA 不会尊重 JAX 中编写操作的具体顺序。

2. **生命周期（Lifetimes）**——XLA 假设一旦某个值在依赖图中不再被引用，其内存就可以被释放给其他值使用。如果我们有一个异步拷贝 x -> y 的操作，我们需要确保 x 在拷贝完成之前保持存活，否则我们将从垃圾内存中读取。

3. **防御性拷贝（Defensive copies）**——XLA 保留创建值副本的权利。我们需要确保不引入不必要的拷贝，以 a) 避免不必要的运行时开销，b) 确保正确性。

我们将逐一讨论这些问题并提出解决方案。

### 调度

如何在 JAX 中显式强制操作按特定顺序执行？注意这不是 Pallas 特有的问题，如果我们使用其他方法实现异步操作，也会遇到同样的问题。

一种方法是在 XLA 程序中引入 optimization barrier。optimization barrier 会阻止 XLA 在其两侧移动操作。

原始代码：

```python
def f(x):
  fut = ppermute_start(x)
  z = x + 1
  y = ppermute_done(fut)
  return y, z
```

XLA 可以选择在三个位置中的任何一个执行 `x + 1`：

```python
def f(x):
  z = x + 1
  fut = ppermute_start(x)
  y = ppermute_done(fut)
  return y, z

# 或者

def f(x):
  fut = ppermute_start(x)
  z = x + 1
  y = ppermute_done(fut)
  return y, z

# 或者

def f(x):
  fut = ppermute_start(x)
  y = ppermute_done(fut)
  z = x + 1
  return y, z
```

为了强制 `x + 1` 发生在两个 `ppermute` 操作之间，我们可以使用 `optimization_barrier`。它在语义上是恒等函数（即 `lambda x: x`），但引入了值之间的显式数据依赖关系。具体来说，如果我们让 `x + 1` 中使用的 `x` 依赖于 `ppermute_start` 返回的 `fut`，那么它必须在 `ppermute_start` 之后执行。

我们还引入一个依赖关系，强制输出值 `y` 依赖于 `z`。

```python
def f(x):
  fut = ppermute_start(x)
  x, fut = optimization_barrier((x, fut))  # 现在 x 依赖于 fut
  z = x + 1
  z, fut = optimization_barrier((z, fut))  # 现在 fut 依赖于 z
  y = ppermute_done(fut)
  return y, z
```

`optimization_barrier` 是一个足够好用的工具，可以让我们显式地写出调度顺序。

### 生命周期

让我们再看一下原始代码，假设操作按正确顺序执行。

```python
def f(x):
  fut = ppermute_start(x)
  z = x + 1
  y = ppermute_done(fut)
  return y, z
```

让我们看看程序中哪个时点 XLA 认为可以释放 `x` 的 buffer。它应该是 `x` 不再被使用的那个时点，具体来说就是 `z = x + 1` 之后。

```python
def f(x):
  fut = ppermute_start(x)
  z = x + 1
  # XLA 可以在此处释放 x！
  y = ppermute_done(fut)
  return y, z
```

如果 XLA 在 `z = x + 1` 完成后释放了 `x`，我们就会遇到一个非常严重的问题。`ppermute` 可能仍在将 `x` 活跃地拷贝给邻居，这意味着如果 `x` 被释放，`ppermute` 将从垃圾内存中读取！

如何将 `x` 的生命周期延长到 `ppermute_done`？我们可以引入数据依赖！我们需要稍微修改 kernel 来实现这一点。

首先，我们重写 `ppermute_start` 使其返回 `x`，通过 kernel 做 alias。

```python
def ppermute_start_kernel(
    in_ref, send_sem, recv_sem, out_ref, _, *, axis_name,
):
  axis_size = jax.lax.psum(1, axis_name)
  left_neighbor = jax.lax.rem(
      jax.lax.axis_index(axis_name) - 1 + axis_size, axis_size
  )
  right_neighbor = jax.lax.rem(jax.lax.axis_index(axis_name) + 1, axis_size)
  barrier_sem = pltpu.get_barrier_semaphore()
  pltpu.semaphore_signal(barrier_sem, device_id=left_neighbor)
  pltpu.semaphore_wait(barrier_sem, 1)
  pltpu.make_async_remote_copy(
      in_ref, out_ref, send_sem, recv_sem, device_id=right_neighbor
  ).start()

def ppermute_start(x, *, axis_name) -> tuple[Semaphore, Semaphore, Array, Array]:
  send_sem, recv_sem, x, out = pl.pallas_call(
      functools.partial(ppermute_start_kernel, axis_name=axis_name),
      out_shape=(
          pltpu.SemaphoreType.DMA(()),
          pltpu.SemaphoreType.DMA(()),
          jax.ShapeDtypeStruct(
              x.shape,
              dtype=x.dtype,
          ),
          jax.ShapeDtypeStruct(
              x.shape,
              dtype=x.dtype,
          ),
      ),
      in_specs=[
          pl.BlockSpec(memory_space=pl.ANY),
      ],
      out_specs=(
          pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          pl.BlockSpec(memory_space=pl.ANY),
          pl.BlockSpec(memory_space=pl.ANY),
      ),
      input_output_aliases={0:2}
  )(x)
  return send_sem, recv_sem, x, out
```

然后让 `ppermute_done` 接收 `x` 但不对其进行任何操作。

```python
def ppermute_done_kernel(_, ref, send_sem, recv_sem, _):
  pltpu.make_async_copy(ref, ref, send_sem).wait()
  pltpu.make_async_copy(ref, ref, recv_sem).wait()

def ppermute_done(send_sem, recv_sem, x, out) -> Array:
  out = pl.pallas_call(
      ppermute_done_kernel,
      out_shape=(
          jax.ShapeDtypeStruct(
              out.shape,
              dtype=out.dtype,
          ),
      ),
      in_specs=[
          pl.BlockSpec(memory_space=pl.ANY),
          pl.BlockSpec(memory_space=pl.ANY),
          pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
          pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
      ],
      out_specs=pl.BlockSpec(memory_space=pl.ANY),
      input_output_aliases={1:0}
  )(x, out, send_sem, recv_sem)
  return out
```

现在当我们写：

```python
def f(x):
  *sems, x, out = ppermute_start(x)
  z = x + 1
  y = ppermute_done(*sems, x, out)
  return y, z
```

XLA 无法再释放 `x`，因为它是 `ppermute_done` 的输入！这意味着 `x` 的生命周期与 `ppermute` 绑定，代码现在是正确的。

### 防御性拷贝

XLA 在其 buffer assignment pass 中分析哪些 buffer 互相 alias，并在 alias 了某个输入的操作不是该输入的最终消费者时插入 copy。

#### 背景

一个简单的例子：假设我们有一个操作 `add_one_inplace`，它接收一个数组并加一，但承诺就地完成。

以下代码是合法的：

```python
def f():
  x = jnp.arange(...)
  y = add_one_inplace(x)
  return y
```

然而，如果 `x` 还有其他消费者，程序可能无法正确执行。

```python
def f():
  x = jnp.arange(...)
  y = add_one_inplace(x)
  return y, x * 2  # 另一个 x 的消费者！
```

这是因为 `x * 2` 操作的是原始的 `x`，但 `add_one_inplace` 破坏了 `x` 中的值。`x * 2` 需要确保读取的是 `x` 的原始值，而不是加 1 之后的值。XLA 注意到这一点并插入一个 `copy` 操作（语义上是恒等操作，但输入和输出 buffer 不同）。

```python
def f(x):
  x2 = copy(x)
  y = add_one_inplace(x2)
  return y, x * 2
```

XLA 中的这个 pass 通过强制就地更新操作变成实质上的非就地操作（通过 `copy` 操作），确保了正确性。

#### 下游操作引起的拷贝

让我们重新审视在 `ppermute` 期间加 1 的例子。

```python
def f(x):
  fut = ppermute_start(x)
  z = x + 1
  y = ppermute_done(fut)
  return y, z
```

如果我们将 future 解包为其组成部分，就能看到 alias 模式：

```python
def f(x):
  *sems, x2, y = ppermute_start(x)
  z = x + 1
  y = ppermute_done((*sems, x2, y))
  return y, z
```

我们知道 `x` 未被 `ppermute_start` 修改（即 `x` 与 `x2` 相同），但 XLA 并不知道这一点。实际上，在 XLA 看来，这就像我们之前的 `add_one_inplace` 例子，它保守地假设 `ppermute_start` 修改了 `x`，而 `x2` 是新的 alias 结果。因此，当我们执行 `z = x + 1` 时，遇到了原始 buffer 的一个消费者。XLA 因此引入了一次拷贝！

```python
def f(x):
  x2 = copy(x)
  *sems, x2, y = ppermute_start(x2)
  z = x + 1
  y = ppermute_done((*sems, x2, y))
  return y, z
```

这次拷贝是不必要的，因为我们知道 `x2` 与 `x` 没有变化。要消除这次拷贝，我们需要某种机制来告知 XLA 我们只是在转发一个值。然而，在缺少这种机制的情况下，我们可以稍微重写程序，显式使用 `x2` 代替 `x`。

```python
def f(x):
  *sems, x2, y = ppermute_start(x)
  z = x2 + 1
  y = ppermute_done((*sems, x2, y))
  return y, z
```

现在 XLA 看不到 `x` 的单独消费者，因此不再引入拷贝。然而，这有一个很大的缺点：它迫使我们解包来自 `ppermute_start` 的 future。它将生命周期问题与拷贝问题耦合在了一起。

#### 循环中的 alias

让我们考虑一个稍微高级的例子。我们实现一个函数，使用 `while_loop` 和 `ppermute` 沿环形发送值。

```python
def f(x):
  def body(i, x):
    fut = ppermute_start(x)
    y = ppermute_done(fut)
    return y
  return fori_loop(0, 8, body, x)
```

`fori_loop` 的一个实现细节是，输入和输出 buffer 会自动相互 alias。注意我们在 `ppermute_start` 和 `ppermute_done` 操作中设置了额外的 alias。让我们通过为程序中的每个值着色来运行自己的 "buffer assignment"，确定需要多少个唯一 buffer。

首先，我们解包包含 alias 的 `x` 和 `out` buffer 的 `fut` 元组。

```python
def f(x):
  def body(i, x):
    *sems, x, y = ppermute_start(x)
    y = ppermute_done(*sems, x, y)
    return y
  return fori_loop(0, 8, body, x)
```

现在让我们根据分配的唯一 buffer 为每个值着色。我们有来自 `fori_loop` 的输入/输出 alias、来自 `ppermute_start` 的 `x` alias 和来自 `ppermute_done` 的 `y` alias。

```python
def f(x):
  def body(i, x):
    *sems, x, y = ppermute_start(x)
    y = ppermute_done((*sems, x, y))
    return y
  return fori_loop(0, 8, body, x)
```

如果你运行 alias 分析，你会发现所有 buffer 都被着上了相同的颜色！直觉上这是有问题的，因为如果我们在做一个循环的 `ppermute`，我们不能写入与发送相同的 buffer。我们通常需要一个额外的（即 "double"）buffer 来接收，然后通常在下一次迭代中交换发送/接收 buffer。XLA 实际上会做的是观察到 buffer 复用并防御性地插入一次拷贝。

```python
def f(x):
  def body(i, x):
    x = copy(x)
    *sems, x, y = ppermute_start(x)
    y = ppermute_done((*sems, x, y))
    return y
  return fori_loop(0, 8, body, x)
```

这次拷贝使得 `x` 和 `y` 不再互相 alias，程序将是正确的。然而，我们真的需要这次拷贝吗？如何引入 double buffer 来避免每次迭代的昂贵拷贝？答案是展开（unrolling）！

我们手动展开代码：

```python
def f(x):
  def body(i, x):
    *sems, x, x2 = ppermute_start(x)
    x2 = ppermute_done((*sems, x, x2))

    *sems, x2, y = ppermute_start(x2)
    y = ppermute_done((*sems, x2, y))
    return y
  return fori_loop(0, 4, body, x)
```

现在如果我们运行同样的 alias 分析，会发现所有 buffer 不再互相 alias，我们不需要插入防御性拷贝来保证正确性。

因此，消除这些拷贝的简单方案是使用 `unroll >= 2` 的 `fori_loop`。

```python
def f(x):
  def body(i, x):
    fut = ppermute_start(x)
    y = ppermute_done(fut)
    return y
  return fori_loop(0, 8, body, x, unroll=2)
```

这就足以实现无额外拷贝的循环了！

#### 跨循环边界传递 future

现在让我们看一个更高级的例子。我们实现与之前相同的程序，但将循环错开——在循环之前的 prologue 中启动 `ppermute`，并在循环开头等待 `ppermute`。

```python
def f(x):
  fut = ppermute_start(x)
  def body(i, fut):
    x = ppermute_done(fut)
    fut = ppermute_start(x)
    return fut
  fut = fori_loop(0, 7, body, fut)
  return ppermute_done(fut)
```

在这个例子中，我们不是从一次循环迭代向另一次传递值 `x`，而是传递一个 future 值。

让我们再次解包 future 看看发生了什么：

```python
def f(x):
  fut = ppermute_start(x)
  def body(i, fut):
    *sems, x, out = fut
    x = ppermute_done((*sems, x, out))
    (*sems, x, out) = ppermute_start(x)
    return (*sems, x, out)
  (*sems, x, out) = fori_loop(0, 7, body, x)
  return ppermute_done((*sems, x, out))
```

所以我们显式地将 semaphore、输入 buffer 和目标输出 buffer 作为 loop carry 传递。如果我们现在运行 alias 分析会怎样？我们会遇到与前一节相同的 alias 问题，其中 `x` 和 `out` 将互相 alias。XLA 将引入一次拷贝。

```python
def f(x):
  fut = ppermute_start(x)
  def body(i, fut):
    *sems, x, out = fut
    out = copy(out)
    x = ppermute_done((*sems, x, out))
    (*sems, x, out) = ppermute_start(x)
    return (*sems, x, out)
  (*sems, x, out) = fori_loop(0, 7, body, x)
  return ppermute_done((*sems, x, out))
```

在这种情况下，我们对 `out` 插入了拷贝。然而这是一个非常糟糕的场景，因为 `out` 正在被活跃地写入！即使我们对 `x` 插入拷贝，也会遇到问题，因为这样 `x` 的生命周期将不会延长到 `ppermute_done`。这是非常非常严重的问题！我们不仅会得到拷贝，还会得到不正确的结果！

解决方案正如我们之前观察到的——通过展开来避免所有 buffer 的 alias，从而避免拷贝。所以，如果我们：

```python
def f(x):
  fut = ppermute_start(x)
  def body(i, fut):
    x = ppermute_done(fut)
    fut = ppermute_start(x)
    return fut
  fut = fori_loop(0, 7, body, x, unroll=2)
  return ppermute_done(fut)
```

程序现在应该是正确的。

### 综合应用

总结一些经验法则：

1. 如果有操作依赖于 `ppermute` 的输入值，解包 future 以使用 alias 后的值代替原始值。
2. 在循环体中做 `ppermute` 时使用 `unroll >= 2`。

让我们将所有内容整合到一个函数中，在循环中做 `ppermute` 并累加结果。

```python
def f(x):
  out = jnp.zeros_like(x)
  fut = (*sems, x, out) = ppermute_start(x)
  out = out + x
  def body(i, carry):
    out, fut = carry
    x = ppermute_done(fut)
    fut = (*sems, x, out) = ppermute_start(x)
    out = out + x
    return out, fut
  out, fut = fori_loop(0, 7, body, (out, fut), unroll=2)
  return out, ppermute_done(fut)
```

注意在这个例子中，我们不需要 `optimization_barrier`，因为循环边界本身就充当了调度屏障，将 `start` 和 `done` 分隔开。

大功告成！这将是 Pallas 中异步操作的正式 API。感谢大家！任务完成！

***是这样吗？***

## State 的逆袭

虽然我们似乎通过一些巧妙的技巧解决了拷贝和正确性问题，但我们仍处于一个尴尬的境地。这个 API 功能强大，但有太多太多的陷阱和注意事项。很可能还有更多边界情况需要处理，甚至需要深入了解 XLA 才能预测或理解。我们应该发布这样一个 API 吗？还是有替代方案？

答案可能一直就在我们眼前。

让我们再从头走一遍整个过程，只是这次用有状态（stateful）的版本。这意味着每个自定义异步操作现在操作的是 `Ref` 而不是值。

```python
def ppermute_start_stateful(x_ref, y_ref) -> tuple[Semaphore, Semaphore]:
  ...

def ppermute_done_stateful(send_sem, recv_sem, x_ref, y_ref) -> None:
  ...
```

假设我们可以在 Pallas 中实现这些，看看新程序会是什么样子。先从基本的 collective permute 开始：

```python
def f(x):
  x_ref = make_ref(x)
  y_ref = make_ref(zeros_like(x))
  fut = ppermute_start_stateful(x_ref, y_ref)
  ppermute_done_stateful(*fut, x_ref, y_ref)
  return y_ref[...]
```

它比我们原始的基于值的版本稍微冗长一些，但有几个关键区别。首先，我们创建了一个"空"的 `Ref` 来接收 `ppermute` 的结果，而基于值的版本会自动为我们创建一个值。一个很好的特性是 `x_ref` 的生命周期在这里很清晰：它一直存活到 `ppermute_done_stateful`。我们不需要像之前那样将 `x` 值"偷偷"传入操作中。

另一个区别在我们尝试在 `start/done` 之间添加操作时变得更加明显。

```python
def f(x):
  x_ref = make_ref(x)
  y_ref = make_ref(zeros_like(x))
  fut = ppermute_start_stateful(x_ref, y_ref)
  x_ref[...] += 1
  ppermute_done_stateful(*fut, x_ref, y_ref)
  return y_ref[...]
```

之前我们遇到了调度歧义，XLA 可能相对于 `ppermute` 重新排序 add 操作。使用有状态语义时，我们实际上添加了排序约束！`x_ref[...] += 1` 修改了 `x_ref`，所以它不能相对于 `ppermute_done_stateful` 被移动。JAX 可以在 lowering 到 HLO 的过程中注入这些调度约束。

最后一个关键区别在我们尝试循环示例时变得明显。

```python
def f(x):
  x_ref = make_ref(x)
  y_ref = make_ref(zeros_like(x))
  def body(i, _):
    fut = ppermute_start_stateful(x_ref, y_ref)
    ppermute_done_stateful(*fut, x_ref, y_ref)
    # 现在切换为 y_ref -> x_ref
    fut = ppermute_start_stateful(y_ref, x_ref)
    ppermute_done_stateful(*fut, y_ref, x_ref)
  fori_loop(0, 8 // 2, body, None)
  return x_ref[...]
```

由于我们需要一个单独的 buffer 来接收 `ppermute`，我们被迫以展开的方式编写代码！XLA 中那种需要拷贝的版本在这里根本写不出来，因为那将涉及一个从 `Ref` 发送到自身的 `ppermute`，这没有意义。

要在不手动展开的情况下处理这个问题，我们可以创建一个带有前导维度 `2` 的 scratch buffer，作为跨迭代的发送/接收目标，每次迭代交换一次。这与我们在 Pallas kernel 内部编写手动重叠 kernel 时使用的模式相同。

这里的核心认识是，有状态设计迫使我们提前处理基于值语义时才会出现的许多问题。我们从定义上消除了这些问题！

1. **调度**——以 `Ref` 为输入的有状态操作强制了程序的顺序。注意，这会对操作同一个 `Ref` 的操作相互排序。我们可能还需要一个 `opt_barrier_stateful` 来施加更多排序约束。

2. **生命周期**——`Ref` 的生命周期可以通过 `run_state` 来限定范围，或者可以作为有状态操作的输入。

3. **防御性拷贝**——使用 `Ref` 迫使我们"手动"处理 buffer assignment，而 lowering 可以确保 alias 正确运作以避免任何拷贝。

另一个重要的根本限制是，我们最终会生成一个 HLO 程序，其中存活的 buffer 和 semaphore 被表示为 array value 类型。XLA 不对这些中间值的 buffer 生命周期或它们所在的 memory space 提供保证。**因此，即使 Pallas kernel 正在活跃地向 array value 进行拷贝，XLA 也有可能拷贝这些值。** 这在 HLO 中很容易验证，但它是使用 custom call 在 HLO 中表示异步操作的一个尖锐边界。

## 结论

我们讨论了 Pallas 和 JAX 中异步操作面临的一些棘手挑战。`Ref` 似乎是表示这些操作的一种有前景的方式，它规避了基于值语义时出现的一些问题。然而，一个缺点是它将有状态 JAX 推到了前台和中心位置，这是我们在 Pallas 之外尚未做过的。值得思考的是，我们应该教育用户了解有状态操作，还是提供一个更危险的 API。我们也不确定我们想做的所有事情是否都可以通过 `Ref` 来表达。我们还应该集思广益讨论 state 之外的替代方案，以充实设计空间。例如，如果 XLA 提供一个尊重生命周期的一等 future API，并且它可以自动做一些事情，比如对包含 future 的循环进行 double buffer 呢？这可能是一个可行的替代方案，但权衡在于将更多控制权交给编译器，而不是由用户显式控制。
