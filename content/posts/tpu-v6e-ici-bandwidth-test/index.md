---
title: "TPU v6e ICI 带宽实测——单条 Link 93 GB/s，标称 800 GB/s 的真相"
date: 2026-04-15
draft: false
tags: ["TPU", "ICI", "JAX", "Benchmark", "Trillium"]
categories: ["TPU"]
summary: "在 GKE 上对 TPU v6e 2×2 集群进行 ICI 带宽实测，验证 device_put 跨 mesh 走 ICI 直连不经 Host，实测单条 Link 单向 93 GB/s、psum 4-device 聚合 445 GB/s，并分析标称 800 GB/s 与实际带宽的差距"
---

## TL;DR

我们在 GKE 上对 TPU v6e (Trillium) 2×2 集群进行了全面的 ICI（Inter-Chip Interconnect）带宽测试。三个核心发现：

1. **`jax.device_put` 跨 mesh 传输确实走 ICI 直连**，不经过 Host memory——通过 HLO dump、Profiler trace 和带宽反推三重证据确认
2. **单条 ICI Link 实测单向 ~93 GB/s、双向 ~186 GB/s**，远低于标称 800 GB/s。各 Link 完全独立、方向对称
3. **XLA psum 4-device 聚合带宽达 445 GB/s**，证明多条 Link 确实并行工作，但单 Link 仍未打满

---

## 测试环境

| 参数 | 值 |
|------|-----|
| 平台 | GKE `tpu-service` 集群 (us-east5) |
| 加速器 | TPU v6e (Trillium), 2×2 拓扑 (4 chips, 单主机) |
| 每芯片 HBM | 32 GB |
| ICI 标称带宽 | 800 GB/s 双向/chip |
| JAX 版本 | jax[tpu] pip latest (2026-04-15) |

---

## 测试拓扑

```
v6e 2×2 Torus:

  chip_0(0,0) ═══ x ═══ chip_1(1,0)
      ║                      ║
      y                      y
      ║                      ║
  chip_2(0,1) ═══ x ═══ chip_3(1,1)

  mesh_a = {chip_0, chip_1}    ← 上面一行
  mesh_b = {chip_2, chip_3}    ← 下面一行
```

---

## 核心结论

### 跨 mesh `device_put` 不经 Host

通过三重证据确认 `jax.device_put` 跨 mesh 传输走 ICI 直连：

| 证据 | 方法 | 结果 |
|------|------|------|
| HLO 分析 | dump 优化后 HLO，搜索 send/recv/infeed/outfeed | 无 Host 通路 |
| Profiler Trace | 搜索 `TransferFromDevice` 事件 | 0 条 |
| 带宽反推 | Host↔Device 仅 ~12 GB/s，device_put 达 93 GB/s | 不可能走 Host |

### ICI 带宽实测总览

| 传输方式 | 峰值带宽 | 说明 |
|----------|----------:|------|
| psum 4-device (XLA collective) | **444.9 GB/s** | 所有 Link 同时工作 |
| device_put 0↔1 双向并发 | **185.8 GB/s** | 单条 Link 双向 |
| device_put 2→2 跨 mesh | **184.9 GB/s** | 2 条 Link 并行 |
| psum 2-device (XLA collective) | **163.7 GB/s** | 单条 Link, XLA 优化 |
| all_gather 4-device | **141.2 GB/s** | XLA collective |
| device_put 1→1 单卡 | **92.9 GB/s** | 单条 Link 单向 |
| Host → Device | **11.7 GB/s** | PCIe-like |
| Device → Host | **8.3 GB/s** | PCIe-like |

### ICI Link 特性

- **各 Link 完全独立**：chip0 同时向 chip1 和 chip2 发送，每条 Link 带宽不受影响 (ratio = 1.00×)
- **x/y 方向对称**：x-link 和 y-link 带宽一致 (92.6 vs 92.6 GB/s)
- **所有芯片对等**：chip0→1, chip0→2, chip0→3, chip1→3 带宽完全一致
- **单条 Link 饱和点**：单向 ~93 GB/s，双向 ~186 GB/s，约为标称 800 GB/s 的 **23%**

---

## 详细测试结果

### device_put 带宽 vs 数据大小（单卡→单卡）

| 数据量 | Per Shard | 耗时 (ms) | 带宽 (GB/s) | 备注 |
|-------:|----------:|----------:|------------:|------|
| 4 KB | 2 KB | 0.247 | 0.02 | latency 主导 |
| 16 KB | 8 KB | 0.243 | 0.07 | 固定开销 ~0.24ms |
| 64 KB | 32 KB | 0.246 | 0.27 | |
| 256 KB | 128 KB | 0.241 | 1.09 | |
| 1 MB | 512 KB | 0.239 | 4.39 | |
| 4 MB | 2 MB | 0.239 | 17.58 | |
| 16 MB | 8 MB | 0.389 | 43.11 | bandwidth 开始主导 |
| 64 MB | 32 MB | 0.659 | 101.77 | |
| 256 MB | 128 MB | 1.699 | 157.95 | |
| 1 GB | 512 MB | 6.058 | 177.23 | |
| 2 GB | 1 GB | 11.78 | 182.24 | |
| 4 GB | 2 GB | 23.37 | 183.75 | |
| 8 GB | 4 GB | 46.46 | 184.90 | |
| 12 GB | 6 GB | 69.43 | 185.58 | 饱和 |
| 16 GB | — | — | OOM | |

> **Insight**：带宽曲线呈现经典的"阶梯型"模式——小数据量被固定延迟 (~0.24ms) 主导，约 16 MB 开始 bandwidth 主导，8 GB 以上进入饱和区。这说明 ICI DMA 的启动开销极低，适合大块数据传输。

### device_put 各方向带宽（4 GB 数据）

| 方向 | 耗时 (ms) | 带宽 (GB/s) |
|------|----------:|------------:|
| chip0→chip1 (x-neighbor) | 46.37 | 92.6 |
| chip0→chip2 (y-neighbor) | 46.39 | 92.6 |
| chip0→chip3 (diagonal) | 46.39 | 92.6 |
| chip1→chip3 (y-neighbor) | 46.41 | 92.6 |

所有方向带宽完全一致——2×2 Torus 中 diagonal 传输也是单跳（经 x 或 y 中转只需 2 跳，但 XLA 可能选择了双 Link 并行路径）。

### ICI Link 独立性验证（4 GB 数据）

**chip0 同时向 chip1 和 chip2 发送**：

| 测试 | chip0→chip1 | chip0→chip2 | 聚合 |
|------|------------:|------------:|-----:|
| 单独 (baseline) | 92.6 GB/s | 92.6 GB/s | — |
| 同时发送 | 92.6 GB/s | 92.7 GB/s | 185.3 GB/s |
| **Ratio** | **1.00×** | **1.00×** | — |

**全部 Link 同时传输**：

| 模式 | 每条 Link | 聚合带宽 |
|------|----------:|---------:|
| 所有 x-link (0→1, 1→0, 2→3, 3→2) | 92.6 GB/s | 370.5 GB/s |
| 所有 y-link (0→2, 2→0, 1→3, 3→1) | 92.6 GB/s | 370.6 GB/s |
| Ring (0→1, 1→3, 3→2, 2→0) | 92.6 GB/s | 370.3 GB/s |

> **Insight**：ICI Link 之间完全没有干扰——无论是同一芯片的不同 Link，还是不同芯片的 Link，并发传输时每条 Link 的带宽都保持在 92.6 GB/s 不变。这证明 ICI 是真正的点对点独立链路，没有共享的总线瓶颈。

### XLA Collective vs device_put

**psum (all-reduce) — 2 devices（单条 Link）**：

| 数据量 | 耗时 (ms) | algoBW (GB/s) |
|-------:|----------:|--------------:|
| 256 MB | 1.91 | 140.8 |
| 512 MB | 3.55 | 151.3 |
| 1 GB | 6.82 | 157.4 |
| 2 GB | 13.37 | 160.6 |
| 4 GB | 26.41 | 162.7 |
| 8 GB | 52.49 | **163.7** |

**psum (all-reduce) — 4 devices（所有 Link）**：

| 数据量 | 耗时 (ms) | 带宽 (GB/s) |
|-------:|----------:|------------:|
| 256 MB | 0.90 | 297.7 |
| 512 MB | 1.49 | 361.0 |
| 1 GB | 2.68 | 400.4 |
| 2 GB | 5.06 | 424.5 |
| 4 GB | 9.83 | 436.7 |
| 8 GB | 19.31 | **444.9** |

**device_put 双向并发 (0↔1)**：

| 数据量 | 0→1 | 1→0 | 双向聚合 (GB/s) |
|-------:|----:|----:|----------------:|
| 1 GB | 91.1 | 91.1 | 182.3 |
| 4 GB | 92.5 | 92.5 | 185.0 |
| 8 GB | 92.9 | 92.9 | **185.8** |

### Host ↔ Device 带宽

| 方向 | 数据量 | 耗时 (ms) | 带宽 (GB/s) |
|------|-------:|----------:|------------:|
| Host → Device | 64 MB | 5.62 | 11.9 |
| | 256 MB | 23.05 | 11.6 |
| | 1 GB | 91.56 | 11.7 |
| | 2 GB | 183.48 | **11.7** |
| Device → Host | 64 MB | 7.83 | 8.6 |
| | 256 MB | 32.26 | 8.3 |
| | 1 GB | 134.26 | 8.0 |
| | 2 GB | 268.10 | **8.0** |

> **Insight**：Host↔Device 带宽 (~12/8 GB/s) 与 ICI 带宽 (~93 GB/s) 相差 **8-11 倍**。这是 `device_put` 不可能走 Host 的最直接证据——如果中间需要经过 Host memory，跨 mesh 传输不可能达到 93 GB/s。

---

## Profiler Trace 分析

使用 `jax.profiler` 抓取 TPU trace，在 xprof/Perfetto 中验证数据流路径：

| 事件 | 说明 | 耗时 |
|------|------|-----:|
| TransferToDevice(chip_0, 512B) | Host→Device: scalar 种子值 | ~7 μs |
| broadcast_in_dim | chip_0: scalar → f32[4096] | ~0.5 μs |
| fusion/_multi_slice | 切成 shard | ~1.1 μs |
| CopyToMemorySpace ×1 | mesh_a 内部 shard 分发 | ~9 μs |
| CopyToMemorySpace ×2 | **跨 mesh ICI 传输** 到 mesh_b | ~20+10 μs |
| **TransferFromDevice** | **0 条** — 无 Device→Host 传输 | — |

关键证据：`TransferFromDevice` 事件为 0，证明整个传输过程中数据没有回到 Host。

---

## 开放问题

### 标称 800 GB/s vs 实测 ~186 GB/s 双向

无论 device_put 还是 XLA collective，单条 Link 实测带宽都远低于标称值（约 23%）。可能原因：

1. **800 GB/s 是峰值理论值**：类似 HBM 标称带宽 vs 实际可达带宽，硬件标称通常反映物理层极限
2. **PJRT/XLA 的 DMA 实现未完全利用硬件流水线**：可能存在协议开销或未充分 pipeline
3. **需要更低层级测试**：例如 Pallas remote DMA 或直接 ICI API 来逼近硬件极限

### psum 2-device 反而比 device_put 双向慢

psum 2-device (163.7 GB/s) < device_put 双向并发 (185.8 GB/s)。XLA 编译器的 ring all-reduce 在 2 设备场景下引入了 reduce-scatter + all-gather 两阶段的同步开销，反而不如裸 device_put。

### psum 4-device 超过单 Link 双向上限

psum 4-device 达到 444.9 GB/s，远超单条 Link 双向上限 (~186 GB/s)。这证明 XLA 确实在利用 2×2 Torus 的多条 Link 并行工作，但这是多条 Link 的聚合带宽，不是单 Link 打满。

---

## 测试脚本

<details>
<summary>5.1 跨 mesh HLO 分析</summary>

验证 `device_put` 跨 mesh 是否经过 Host：

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np
import os
import time

# Dump HLO/LLO
os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text=true'

devices = jax.devices()
print(f"Total devices: {len(devices)}")
for d in devices:
    print(f"  {d}")

mesh_a = Mesh(np.array(devices[:2]), ('x',))
mesh_b = Mesh(np.array(devices[2:]), ('x',))
print(f"\nMesh A devices: {mesh_a.devices}")
print(f"Mesh B devices: {mesh_b.devices}")

# === Test 1: device_put cross-mesh (outside jit) ===
print("\n" + "="*60)
print("TEST 1: device_put cross-mesh transfer (outside jit)")
print("="*60)

sharding_a = NamedSharding(mesh_a, P('x'))
data_a = jax.device_put(jnp.ones((128,)), sharding_a)
print(f"Data on mesh_a: devices={data_a.devices()}, shape={data_a.shape}")

sharding_b = NamedSharding(mesh_b, P('x'))
t0 = time.time()
data_b = jax.device_put(data_a, sharding_b)
data_b.block_until_ready()
t1 = time.time()
print(f"Data on mesh_b: devices={data_b.devices()}, shape={data_b.shape}")
print(f"Transfer time: {(t1-t0)*1000:.3f} ms")

# === Test 2: Compute on mesh_a then transfer result to mesh_b ===
print("\n" + "="*60)
print("TEST 2: Compute on mesh_a, then device_put to mesh_b")
print("="*60)

@jax.jit
def compute_on_a(x):
    return x * 2.0 + 1.0

with mesh_a:
    result_a = compute_on_a(data_a)
    result_a.block_until_ready()
    print(f"Result on mesh_a: devices={result_a.devices()}")

result_b = jax.device_put(result_a, sharding_b)
result_b.block_until_ready()
print(f"Result on mesh_b: devices={result_b.devices()}")

# === Dump HLO analysis ===
print("\n" + "="*60)
print("XLA DUMP ANALYSIS")
print("="*60)

print("\n--- After optimizations HLO (all modules) ---")
os.system("for f in /tmp/xla_dump/*.after_optimizations.txt; do "
          "echo '\\n===== '$f' ====='; cat $f; done 2>/dev/null")

print("\n--- Search for send/recv/copy-to-host/infeed/outfeed ---")
os.system("grep -rni 'send\\|recv\\|infeed\\|outfeed\\|host\\|transfer-to-host' "
          "/tmp/xla_dump/*.after_optimizations.txt 2>/dev/null | head -30 "
          "|| echo 'No send/recv/host operations found in HLO'")

# === Check JAX transfer guard ===
print("\n" + "="*60)
print("TEST 3: JAX transfer guard test")
print("="*60)
try:
    jax.config.update("jax_transfer_guard", "disallow")
    data_b3 = jax.device_put(data_a, sharding_b)
    data_b3.block_until_ready()
    print("Transfer succeeded even with disallow — device_put is explicit, "
          "transfer_guard only blocks implicit transfers")
except Exception as e:
    print(f"Transfer BLOCKED: {e}")
finally:
    jax.config.update("jax_transfer_guard", "allow")
```

</details>

<details>
<summary>5.2 Profiler Trace 抓取</summary>

抓取 `jax.profiler` trace 用于 xprof/Perfetto 分析：

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np
import os

devices = jax.devices()
mesh_a = Mesh(np.array(devices[:2]), ('x',))
mesh_b = Mesh(np.array(devices[2:]), ('x',))
sharding_a = NamedSharding(mesh_a, P('x'))
sharding_b = NamedSharding(mesh_b, P('x'))

# Warmup
data_a = jax.device_put(jnp.ones((1024,)), sharding_a)
data_b = jax.device_put(data_a, sharding_b)
data_b.block_until_ready()

# Profile
TRACE_DIR = "/tmp/tpu_trace"
jax.profiler.start_trace(TRACE_DIR)
for i in range(10):
    data_a = jax.device_put(jnp.ones((4096,)), sharding_a)
    data_a.block_until_ready()
    data_b = jax.device_put(data_a, sharding_b)
    data_b.block_until_ready()
jax.profiler.stop_trace()

print(f"Trace saved to {TRACE_DIR}")
os.system(f"find {TRACE_DIR} -type f")

# 查看方式:
# 1. xprof: xprof /path/to/trace -p 8791, 浏览器打开 localhost:8791
# 2. Perfetto: 打开 https://ui.perfetto.dev, 拖入 .trace.json.gz
```

</details>

<details>
<summary>5.3 device_put 带宽梯度测试</summary>

从 4KB 到 12GB 的完整带宽曲线：

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np
import time

devices = jax.devices()
mesh_a = Mesh(np.array(devices[:2]), ('x',))
mesh_b = Mesh(np.array(devices[2:]), ('x',))
sharding_a = NamedSharding(mesh_a, P('x'))
sharding_b = NamedSharding(mesh_b, P('x'))

# Warmup
data = jax.device_put(jnp.ones((1024,)), sharding_a)
_ = jax.device_put(data, sharding_b); _.block_until_ready()

print(f"\n{'Size':>12}  {'Per Shard':>10}  {'Time (ms)':>10}  {'BW (GB/s)':>10}")
print("-" * 55)

sizes = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304,
         16777216, 67108864, 268435456]

for nelems in sizes:
    total_bytes = nelems * 4
    shard_bytes = total_bytes // 2
    data_a = jax.device_put(jnp.ones((nelems,)), sharding_a)
    data_a.block_until_ready()

    # Warmup
    for _ in range(3):
        tmp = jax.device_put(data_a, sharding_b); tmp.block_until_ready()

    # Timed
    n_iters = max(1, min(100, 500_000_000 // total_bytes))
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        data_b = jax.device_put(data_a, sharding_b)
        data_b.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    med_time = np.median(times)
    bw_gbs = total_bytes / med_time / 1e9
    label = f"{total_bytes/1024:.0f}KB" if total_bytes < 1048576 \
            else f"{total_bytes/1048576:.0f}MB"
    shard_label = f"{shard_bytes/1024:.0f}KB" if shard_bytes < 1048576 \
                  else f"{shard_bytes/1048576:.0f}MB"
    print(f"{label:>12}  {shard_label:>10}  {med_time*1000:>10.3f}  {bw_gbs:>10.2f}")
```

</details>

<details>
<summary>5.4 ICI Link 独立性测试</summary>

验证同一 chip 向两个方向同时发送时带宽是否互相影响：

```python
import jax
import jax.numpy as jnp
import numpy as np
import time
import concurrent.futures

devices = jax.devices()
SIZE_MB = 4096
nelems = SIZE_MB * 1024 * 1024 // 4
total_bytes = nelems * 4

def bench_single(src, dst, label, n_warmup=5, n_iters=20):
    data = jax.device_put(jnp.ones((nelems,)), devices[src])
    data.block_until_ready()
    for _ in range(n_warmup):
        t = jax.device_put(data, devices[dst]); t.block_until_ready(); del t
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        t = jax.device_put(data, devices[dst]); t.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0); del t
    med = np.median(times)
    bw = total_bytes / med / 1e9
    print(f"  {label}: {med*1000:.2f} ms, {bw:.1f} GB/s")
    del data
    return bw

# Baselines
print("Single link baselines:")
bw_01 = bench_single(0, 1, "chip0→chip1 (x-link)")
bw_02 = bench_single(0, 2, "chip0→chip2 (y-link)")

# Simultaneous
print("\nchip0 → chip1 AND chip0 → chip2 simultaneously:")
data_0 = jax.device_put(jnp.ones((nelems,)), devices[0])
data_0.block_until_ready()

def transfer_to(dst_idx):
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        t = jax.device_put(data_0, devices[dst_idx]); t.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0); del t
    return np.median(times)

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    f1 = executor.submit(transfer_to, 1)
    f2 = executor.submit(transfer_to, 2)
    time_01, time_02 = f1.result(), f2.result()

bw_01_sim = total_bytes / time_01 / 1e9
bw_02_sim = total_bytes / time_02 / 1e9
print(f"  chip0→chip1: {bw_01_sim:.1f} GB/s")
print(f"  chip0→chip2: {bw_02_sim:.1f} GB/s")
print(f"  Total: {bw_01_sim + bw_02_sim:.1f} GB/s")
ratio = ((bw_01_sim + bw_02_sim) / 2) / bw_01
print(f"  Ratio: {ratio:.2f}x (1.0 = fully independent)")

# All links simultaneously
print("\nAll x-links simultaneously (0→1, 1→0, 2→3, 3→2):")
data = [jax.device_put(jnp.ones((nelems,)), devices[i]) for i in range(4)]
for d in data: d.block_until_ready()

pairs = [(0,1),(1,0),(2,3),(3,2)]
def do_transfer(src_dst):
    s, d = src_dst
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        t = jax.device_put(data[s], devices[d]); t.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0); del t
    return np.median(times)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(do_transfer, p): p for p in pairs}
    for f in concurrent.futures.as_completed(futures):
        p = futures[f]
        bw = total_bytes / f.result() / 1e9
        print(f"  chip{p[0]}→chip{p[1]}: {bw:.1f} GB/s")
```

</details>

<details>
<summary>5.5 XLA Collective vs device_put 综合对比</summary>

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from functools import partial
import numpy as np
import time
import concurrent.futures

try:
    from jax.shard_map import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map

devices = jax.devices()

def bench(fn, data, n_warmup=10, n_iters=30):
    for _ in range(n_warmup):
        out = fn(data); out.block_until_ready()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        out = fn(data); out.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0); del out
    return np.median(times)

sizes_mb = [256, 512, 1024, 2048, 4096, 8192]

# --- psum on 2 devices (single link) ---
for dev_ids, name in [([0,1], "x-link"), ([0,2], "y-link")]:
    mesh2 = Mesh(np.array([devices[i] for i in dev_ids]), ('x',))
    sharded = NamedSharding(mesh2, P('x'))

    @partial(shard_map, mesh=mesh2, in_specs=P('x'), out_specs=P('x'))
    def do_psum(x):
        return jax.lax.psum(x, 'x')
    do_psum_jit = jax.jit(do_psum)

    print(f"\npsum 2-device ({name}):")
    for size_mb in sizes_mb:
        nelems = size_mb * 1024 * 1024 // 4
        total_bytes = nelems * 4
        data = jax.device_put(jnp.ones((nelems,)), sharded)
        data.block_until_ready()
        med = bench(do_psum_jit, data)
        print(f"  {size_mb:>6}MB: {med*1000:.2f}ms, "
              f"{total_bytes/med/1e9:.1f} GB/s")

# --- psum on 4 devices (all links) ---
mesh4 = Mesh(np.array(devices), ('x',))
sharded4 = NamedSharding(mesh4, P('x'))

@jax.jit
@partial(shard_map, mesh=mesh4, in_specs=P('x'), out_specs=P('x'))
def psum4(x):
    return jax.lax.psum(x, 'x')

print(f"\npsum 4-device (all links):")
for size_mb in sizes_mb:
    nelems = size_mb * 1024 * 1024 // 4
    total_bytes = nelems * 4
    data = jax.device_put(jnp.ones((nelems,)), sharded4)
    data.block_until_ready()
    med = bench(psum4, data)
    print(f"  {size_mb:>6}MB: {med*1000:.2f}ms, "
          f"{total_bytes/med/1e9:.1f} GB/s")

# --- device_put bidirectional ---
print(f"\ndevice_put bidirectional (0↔1):")
for size_mb in [1024, 4096, 8192]:
    nelems = size_mb * 1024 * 1024 // 4
    total_bytes = nelems * 4
    d0 = jax.device_put(jnp.ones((nelems,)), devices[0])
    d1 = jax.device_put(jnp.ones((nelems,)), devices[1])
    d0.block_until_ready(); d1.block_until_ready()

    def xfer(src_data, dst_dev):
        for _ in range(5):
            t = jax.device_put(src_data, dst_dev)
            t.block_until_ready()
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            t = jax.device_put(src_data, dst_dev)
            t.block_until_ready()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        return np.median(times)

    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        f01 = ex.submit(xfer, d0, devices[1])
        f10 = ex.submit(xfer, d1, devices[0])
        t01, t10 = f01.result(), f10.result()
    bw01 = total_bytes / t01 / 1e9
    bw10 = total_bytes / t10 / 1e9
    print(f"  {size_mb:>6}MB: 0→1={bw01:.1f}, 1→0={bw10:.1f}, "
          f"bidir={bw01+bw10:.1f} GB/s")
```

</details>

<details>
<summary>5.6 Host ↔ Device 带宽测试</summary>

```python
import jax
import jax.numpy as jnp
import numpy as np
import time

devices = jax.devices()
sizes_mb = [64, 128, 256, 512, 1024, 2048]

# Host → Device
print("Host → Device:")
for size_mb in sizes_mb:
    nelems = size_mb * 1024 * 1024 // 4
    total_bytes = nelems * 4
    host_data = np.ones((nelems,), dtype=np.float32)
    for _ in range(3):
        tmp = jax.device_put(host_data, devices[0])
        tmp.block_until_ready()
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        d = jax.device_put(host_data, devices[0])
        d.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    med = np.median(times)
    print(f"  {size_mb:>6}MB: {med*1000:.2f}ms, "
          f"{total_bytes/med/1e9:.1f} GB/s")

# Device → Host (forced copy)
print("\nDevice → Host:")
for size_mb in sizes_mb:
    nelems = size_mb * 1024 * 1024 // 4
    total_bytes = nelems * 4
    dev_data = jax.device_put(jnp.ones((nelems,)), devices[0])
    dev_data.block_until_ready()
    for _ in range(3):
        host_buf = np.empty((nelems,), dtype=np.float32)
        np.copyto(host_buf, dev_data)
    times = []
    for _ in range(10):
        host_buf = np.empty((nelems,), dtype=np.float32)
        t0 = time.perf_counter()
        np.copyto(host_buf, dev_data)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    med = np.median(times)
    print(f"  {size_mb:>6}MB: {med*1000:.2f}ms, "
          f"{total_bytes/med/1e9:.1f} GB/s")
```

</details>

