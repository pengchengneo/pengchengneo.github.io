---
title: "TorchTPU：让 TPU 成为 PyTorch 的一等公民"
date: 2026-08-24
draft: true
tags: ["TorchTPU", "PyTorch", "Torch/XLA", "TorchAx", "TPU", "XLA"]
categories: ["TPU"]
summary: "从 Torch/XLA、TorchAx 到 TorchTPU，分析 PyTorch 连接 TPU 的三条路径，以及 TorchTPU 如何将设备、执行、编译和分布式能力接入 PyTorch 原生体系。"
---

## TL;DR

Torch/XLA、TorchAx 和 TorchTPU 都能让 PyTorch 模型运行在 TPU 上。三者采用不同的执行模型。

TorchTPU 的目标是提供原生 PyTorch 设备体验。模型继续使用 Tensor、Module、autograd、`torch.compile` 和 PyTorch Distributed。StableHLO、XLA 和 PJRT 留在 backend 内部。

## 一、PyTorch 连接 TPU 的三条路径

```text
Torch/XLA
PyTorch → LazyTensor IR → XLA → PJRT → TPU

TorchAx
PyTorch API → Tensor Subclass → JAX → StableHLO → XLA → TPU

TorchTPU
PyTorch → ATen/FX → StableHLO → XLA → PJRT → TPU
```

| 方案 | 执行模型 | 程序员需要理解 |
|---|---|---|
| Torch/XLA | LazyTensor | `mark_step`、图边界、materialization |
| TorchAx | JAX | `jax.jit`、显式状态、JAX sharding |
| TorchTPU | PyTorch device backend | PyTorch、`torch.compile` |

## 二、Torch/XLA：LazyTensor 编程模型

Torch/XLA 通过 XLATensor 记录 PyTorch 操作。积累的 Lazy IR 随后交给 XLA 编译。

```text
PyTorch ATen → XLATensor → Lazy IR → XLA → TPU
```

程序员需要管理图的提交和同步。`xm.mark_step()`、Tensor materialization 和重新编译会直接影响程序行为。

Torch/XLA 已经支撑了大规模 TPU 训练。它也形成了一套需要单独学习的执行模型。

## 三、TorchAx：PyTorch 前端与 JAX 执行

TorchAx 使用 Tensor subclass 保存 `jax.Array`。`__torch_dispatch__` 将 PyTorch operator 映射到 JAX 实现。

```text
torchax.Tensor
  → __torch_dispatch__
  → JAX operator
  → jax.Array
```

这条路径适合 PyTorch/JAX 互操作。完整计算进入 `jax.jit` 后，可以获得 JAX/XLA 的稳态性能。

复杂模型仍要处理 JAX 的函数式语义。主要问题集中在状态更新、算子覆盖和调试。KV cache、Module buffer、RNG 和 checkpoint 都需要单独验证。

## 四、TorchTPU：原生 PyTorch 设备后端

`torch_tpu` 依赖 `torch`。PyTorch 通过 backend entry point 加载它。

TorchTPU 使用 PrivateUse1 注册 `tpu` 设备，并接入 ATen dispatcher。它还注册 `torch.compile("tpu")` 和 `tpu_dist`。

```python
device = torch.device("tpu")
model = model.to(device)

loss = model(inputs.to(device))
loss.backward()
optimizer.step()
```

“一等公民”包含三层含义：普通 PyTorch Tensor、标准编译接口、标准分布式接口。DDP、FSDP2 和 DTensor 可以沿用 PyTorch 的编程方式。

## 五、TorchTPU 如何生成 StableHLO

TorchTPU 的 eager 路径先把 ATen operator 记录为 DeferredOp。DeferredOp 保存输入、输出 shape 和 MLIR builder。backend 可以聚合多个操作，再生成 StableHLO。

```text
ATen operator
  → PrivateUse1 kernel
  → DeferredOp
  → StableHLO
  → XLA
```

`torch.compile` 路径从 TorchDynamo 捕获的 FX 图开始。AOTAutograd 整理前向和反向图。TorchTPU 使用 placeholder 重放 GraphModule，并通过 ATen kernel 构造 DeferredOp 图。

```text
Python
  → TorchDynamo FX
  → AOTAutograd
  → placeholder 重放
  → DeferredOp DAG
  → StableHLO
  → PjRtClient::CompileAndLoad
```

以 `aten.add` 为例：

```text
FX aten.add
  → TorchTPU Add kernel
  → DeferredOp(BuildAddShlo)
  → stablehlo.add
```

PJRT 将内存中的 MLIR module 交给 XLA。XLA 继续完成 HLO 优化和 TPU 专用 lowering。

## 六、被 backend 接管的复杂度

TorchTPU 在 backend 中处理状态语义、算子 lowering、延迟执行、编译缓存和 PJRT buffer。日常模型代码可以停留在 PyTorch API 层。

编译与性能优化仍然需要理解下层结构：

| 工作 | 需要关注 |
|---|---|
| 模型开发 | 算子覆盖、dynamic shape、graph break |
| 编译分析 | FX、AOTAutograd、StableHLO、HLO |
| 内核优化 | Pallas、Mosaic、LLO、TPU 内存层次 |

TorchTPU 将 PyTorch 与 XLA 之间的适配工作收进 backend。TPU 因此获得了与其他 PyTorch 设备一致的编程入口。
