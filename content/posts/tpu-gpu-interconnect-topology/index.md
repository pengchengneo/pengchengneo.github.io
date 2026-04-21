---
title: "AI 集群互联拓扑——从 TPU 3D Torus + OCS 到 NVLink 全光互联的未来"
date: 2026-04-15
draft: false
tags: ["TPU", "GPU", "NVIDIA", "NVLink", "OCS", "3D Torus", "Interconnect"]
categories: ["AI Infra"]
summary: "深入拆解 TPU 的 3D Torus + OCS 光交换架构如何将 9,216 颗芯片连成一张无交换机网络，对比 NVIDIA 从 NVLink8 到 NVLink576 的交换机扩展路线，以及 Feynman 架构拥抱 OCS 的未来展望"
---

## TL;DR

训练一个万亿参数的模型，单芯片算力再强也没用——**瓶颈在互联**。几千颗芯片如何连成一张高带宽、低延迟的网络，是 AI 基础设施的核心难题。Google 和 NVIDIA 给出了两条截然不同的答案：

**Google TPU** 走的是"无交换机"路线。每颗 TPU 通过 ICI（Inter-Chip Interconnect）直连 6 个邻居，组成 **3D Torus** 拓扑——没有 NVSwitch，没有中心化交换机，每颗芯片既是计算节点又是路由节点。机架间通过 **OCS（光电路交换）** 用 MEMS 微镜切换光路，功耗仅 108W 就能把 9,216 颗芯片连成一个 SuperPod。代价是编译器必须感知拓扑、管理多跳路由。

**NVIDIA GPU** 走的是"交换机全连接"路线。域内所有 GPU 通过 **NVSwitch** 构建单跳全连接——任意两块 GPU 之间只有一跳，通信延迟完全对称。NVLink 域从 8 GPU（Hopper）扩展到 72 GPU（Blackwell）再到 576 chiplet（Rubin Ultra），但域外仍然依赖 InfiniBand/Ethernet，存在约 **18× 的带宽断崖**。

两条路线正在交汇：NVIDIA 在 OFC 2026 上宣布 **Feynman 架构（2028）将引入共封装光学（CPO）和 OCS**，目标是打破 NVLink 域的物理边界，将光学互联扩展到 1,152 颗 GPU 甚至更多。

> 在[上一篇文章](/posts/nvidia-gpu-comparison/)中，我们对比了 GPU 和 TPU 的单芯片规格，结论是"竞争焦点转向集群互联与软件生态"。本文就来深入拆解这个互联层面的故事。

---

## TPU 的互联架构——3D Torus + OCS

### 从 2D 到 3D Torus

TPU v2/v3 时代使用 **2D Torus**——每颗芯片连接 4 个邻居（上下左右），组成一个二维网格，边缘处 wrap-around 相连：

```
2D Torus (4×4 示例, v2/v3)

  ┌──→──→──→──┐
  ↑            ↓
  ○──○──○──○
  ↑  │  │  │  ↓
  ○──○──○──○
  ↑  │  │  │  ↓
  ○──○──○──○
  ↑  │  │  │  ↓
  ○──○──○──○
  ↑            ↓
  └──←──←──←──┘

每颗芯片连接 4 个邻居
网络直径 ≈ √N（每轴最远 √N/2 跳）
```

2D Torus 的问题在于**扩展性**：对于 N 颗芯片排成 √N × √N 的网格，由于 Torus 的环形 wrap-around，每个轴上最远只需走 √N/2 跳，网络直径（最远两点之间的跳数）约为 √N。4,096 颗芯片意味着最坏情况下需要约 64 跳——这对集合通信的延迟是灾难性的。

TPU v4 转向 **3D Torus**，每颗芯片连接 6 个邻居（在 X/Y/Z 三个轴的正反方向各一个）：

```
3D Torus 概念图 (v4+)

        ○───○───○───○
       /|  /|  /|  /|
      ○───○───○───○ |
     /| ○/| ○/| ○/| ○     每颗芯片连接 6 个邻居
    ○───○───○───○ |/|     网络直径 ≈ (3/2)∛N
    | ○─|─○─|─○─|─○ |     4,096 chips: 64 跳 → 24 跳
    |/  |/  |/  |/  ○
    ○───○───○───○  /
    |   |   |   | /
    ○───○───○───○

每个轴的 wrap-around 连接未画出
```

3D Torus 将网络直径从 √N 降至 (3/2)∛N。同样 4,096 颗芯片，最远跳数从约 64 降到约 24——延迟降低了近 3 倍。更关键的是，**bisection bandwidth（对分带宽）提升了 2-4 倍**，这直接决定了 AllReduce 等集合通信操作的吞吐量。

### ICI 直连——无交换机的设计哲学

TPU 的互联层 ICI 有一个根本性的设计选择：**没有任何外部交换芯片**。

在 NVIDIA 的世界里，GPU 之间的通信必须经过 NVSwitch——一颗专用的交换 ASIC。而 TPU 的 ICI 是 chip-to-chip 直连：每颗 TPU 芯片内置了路由逻辑，负责转发经过自己的数据包。这意味着：

- 邻居之间的通信是单跳、零交换延迟
- 远端芯片的通信需要多跳，中间芯片充当路由器
- 编译器（XLA）必须感知拓扑，将计算和数据放置在合适的位置以最小化通信跳数

> **Insight**：TPU 的"无交换机"设计本质上是把网络复杂度从硬件转移到了软件。NVSwitch 让 NVIDIA 的集合通信库（NCCL）可以假设所有 GPU 等距，而 TPU 的 XLA 编译器必须解决一个组合优化问题——如何在 3D Torus 上放置张量分片，使通信量和跳数最小。

### 4×4×4 Cube——构建单元

TPU v4 及后续 3D Torus 系统的物理构建单元是 **4×4×4 Cube（立方体）**，每个 Cube 包含 64 颗 TPU 芯片，装在一个机架中：

```
4×4×4 Cube = 64 TPU chips / 机架

        ┌──────────────────────────┐
       /                          /|
      /     4 层 × 4 行 × 4 列   / |
     /       = 64 chips          /  |
    ┌──────────────────────────┐    |
    │                          │    |
    │   Cube 内部:             │    │
    │   PCB 走线 + DAC 铜缆    │   /
    │                          │  /
    │   向外连出 96 条光纤     │ /
    │   (每面 16 条 × 6 面)    │/
    └──────────────────────────┘

Cube 内部连接: 铜缆 (短距, 低延迟, 低成本)
Cube 之间连接: 光纤 → OCS (长距, 可重构)
```

Cube 内部的 ICI 连接使用 PCB 走线和 DAC（Direct Attach Copper）铜缆——距离短、延迟低、成本低。而 Cube 的 6 个面各伸出 16 条光纤，共 96 条光纤连接到外部的 OCS 光交换网络。

### OCS 光电路交换——Palomar

OCS（Optical Circuit Switching，光电路交换）是 TPU v4 架构的核心创新，也是 Google 能将数千颗芯片组网的关键技术。TPU v4 是**全球第一台部署可重构 OCS 的超级计算机**。

Google 自研的 **Palomar OCS** 基于 **3D MEMS（微机电系统）微镜**：

```
Palomar OCS 内部工作原理

输入光纤阵列              输出光纤阵列
    │                         │
    ▼                         ▼
  ┌───┐    ┌───────────┐   ┌───┐
  │   │    │  3D MEMS   │   │   │
  │ 光 │───→│  微镜阵列  │───→│ 光 │
  │ 纤 │    │           │   │ 纤 │
  │ 1  │    │  136×136  │   │ 1  │
  │ 2  │    │  独立可控  │   │ 2  │
  │ .  │    │           │   │ .  │
  │ .  │    │  "W"形光路 │   │ .  │
  │136 │    │           │   │136 │
  └───┘    └───────────┘   └───┘

端口数:     136×136 (128 有效 + 8 备用)
切换时间:   毫秒级 (MEMS 微镜旋转)
功耗:       整个 OCS 仅 108W
            (等效电交换机 EPS 约 3,000W)
延迟:       零交换延迟 (纯光通路, 无 O-E-O 转换)
成本:       < 5% 系统总成本
```

OCS 与传统电交换机（EPS）的本质区别：EPS 将光信号转换为电信号、查表路由、再转回光信号（O-E-O），每一跳都引入延迟和功耗。OCS 则是纯光通路——MEMS 微镜物理转向，光信号直接从一根光纤反射到另一根，**没有光电转换、没有数据包处理、没有缓冲区**。

另一个关键设计是**光环形器（optical circulator）**：这个三端口器件让一根光纤同时支持双向传输，端口数和光缆数量直接减半。

<details>
<summary>深入：OCS 为什么比电交换机省这么多电？</summary>

传统电交换机的功耗公式大致是：

- 每个端口的 SerDes（串行器/解串器）：光电转换
- 交换矩阵（crossbar）：电路由查表 + 数据搬运
- 缓冲区（buffer）：存储转发
- 所有环节都随端口速率线性增长

OCS 的功耗则几乎与端口速率无关：MEMS 微镜只需要在改变配置时消耗能量（旋转微镜到新角度），稳态下几乎不耗电。光信号在整个路径上不做任何处理，功耗仅来自控制电路和少量放大。

这就是为什么 Palomar OCS 整机仅 108W，而等效的 136 端口电交换机需要约 3,000W——**28 倍的功耗差距**。随着 AI 集群规模增长到数万甚至数十万芯片，这个功耗优势会变得越来越关键。

> **Insight**：OCS 的功耗优势来源于一个根本的物理事实——**光不需要被"处理"**。电交换机本质上是在做"光→电→处理→电→光"的工作，而 OCS 跳过了中间所有步骤。

</details>

### SuperPod 扩展——从 4,096 到 9,216 芯片

有了 Cube 和 OCS，构建 SuperPod 就像搭积木：

**TPU v4 SuperPod = 64 Cube × 48 OCS = 4,096 chips**

```
TPU v4 SuperPod 构建方式

64 个 Cube (每个 64 chips = 4,096 chips 总计)
    │
    │ 每个 Cube 伸出 96 条光纤
    │ 总计 64 × 96 = 6,144 条光纤
    │
    ▼
48 个 OCS (Palomar, 每个 136×136 端口)
    │
    │ 分为 3 组, 每轴 16 个 OCS
    │  X 轴: 16 个 OCS
    │  Y 轴: 16 个 OCS
    │  Z 轴: 16 个 OCS
    │
    ▼
6,144 条光纤 / 128 有效端口 ≈ 48 个 OCS
```

三个轴的 OCS 正交隔离——X 轴的 OCS 只管 X 方向的连接，Y 和 Z 同理。这种正交设计简化了路由，避免了死锁。

**TPU v7 (Ironwood) SuperPod = 144 Cube × 48 OCS = 9,216 chips**

v7 将 OCS 从 136×136 升级到 **144×144 端口**，Cube 数量从 64 增加到 144：

| 参数 | TPU v4 SuperPod | TPU v7 SuperPod |
|------|:-:|:-:|
| 芯片总数 | 4,096 | 9,216 |
| Cube 数量 | 64 | 144 |
| OCS 数量 | 48 | 48 |
| OCS 端口 | 136×136 | 144×144 |
| 光纤总数 | 6,144 | 13,824 |
| ICI 带宽/芯片 | ~4,800 Gbps | 9,600 Gbps (1.2 TB/s) |
| 集群算力 (BF16) | — | 42.5 ExaFLOPS |
| 集群 HBM | — | 1.77 PB |

### Twisted Torus——OCS 的杀手级特性

3D Torus 有一个经典的弱点：当拓扑不是正方体时（比如 4×4×8），最长轴方向的跳数更多，bisection bandwidth 更低。

OCS 让 Google 可以实现 **Twisted Torus（扭曲环面）**——在环绕连接时偏移若干步，让数据包不再沿着最长轴直线传播，而是走对角线。这只需要改变 OCS 微镜的角度和路由表，**不需要任何物理重布线**：

```
Regular Torus (4×4×8)          Twisted Torus (4×4×8, twist=4)

第 0 层:  0──1──2──3──→0       第 0 层:  0──1──2──3──→4 (偏移!)
第 1 层:  4──5──6──7──→4       第 1 层:  4──5──6──7──→0
  ...                            ...

效果:
- bisection bandwidth 理论提升 70%
- All-to-all 吞吐实测提升 1.63× (4×4×8)
- All-to-all 吞吐实测提升 1.31× (4×8×8)
- 最优扭转步长 ≈ 维度大小 / 2
```

> **Insight**：Twisted Torus 是 OCS 的杀手级应用——它把一个纯粹的物理布线问题变成了一个软件配置问题。在传统铜缆直连中，拓扑在机架接线时就已固定；而 OCS 让拓扑变成了一个可以在运行时调整的"超参数"。

### 动态可重构——OCS 的其他超能力

除了 Twisted Torus，OCS 还带来了几个传统互联做不到的能力：

**灵活分片（Flexible Slicing）**：一个 4,096 芯片的 Pod 可以被动态切分为任意大小的 Slice——512 芯片可以配置为 4×4×32、4×8×16 或 8×8×8，只需改变 OCS 配置，不需要物理操作。不同用户的 Slice 是物理隔离的。

**故障绕行（Fault Tolerance）**：如果某条 ICI 光链路或某颗芯片故障，OCS 可以通过调整微镜角度将流量路由到备用路径。长达 50 天的 PaLM 训练（540B 参数）能在 TPU v4 Pod 上达到 **57.8% 的 MFU**，很大程度上归功于这种动态故障恢复能力。

**增量部署（Incremental Deployment）**：TPU v3 要求所有 1,024 颗芯片全部就位并测试通过才能使用。TPU v4 的每个 64 芯片 Cube 可以独立上线——装好一个 Cube 就能立即投入生产，边部署边扩展。

---

## NVIDIA 的互联架构——NVLink + NVSwitch 的扩展之路

### NVLink 演进总览

| 代 | 架构 | 速率/Lane | GPU 带宽(双向) | NVLink 域大小 | 年份 |
|:-:|------|:-:|:-:|:-:|:-:|
| 1.0 | Pascal (P100) | 20 Gbps | 160 GB/s | 2 GPU | 2016 |
| 2.0 | Volta (V100) | 25 Gbps | 300 GB/s | 8 GPU (NVSwitch 1.0) | 2017 |
| 3.0 | Ampere (A100) | 50 Gbps | 600 GB/s | 8 GPU (NVSwitch 2.0) | 2020 |
| 4.0 | Hopper (H100) | 112 Gbps | 900 GB/s | 8 GPU (NVSwitch 3.0) | 2022 |
| 5.0 | Blackwell (B200) | 224 Gbps | 1,800 GB/s | 72 GPU (NVSwitch 5.0) | 2024 |
| 6.0 | Rubin | ~448 Gbps | 3,600 GB/s | 72 封装 / 144 die | 2026 |
| 7.0 | Rubin Ultra | — | 3,600 GB/s | 576 chiplet (NVL576) | 2027 |

每一代 NVLink 的 lane 速率基本翻倍，同时 link 数量也在增加，两者叠加使得每 GPU 带宽每两代翻一番。

### NVL8：节点内全连接（Hopper）

DGX H100 的 8 块 GPU 通过 **4 颗 NVSwitch 3.0** ASIC 实现全连接：

```
DGX H100 NVL8 拓扑

    NVSwitch 0    NVSwitch 1    NVSwitch 2    NVSwitch 3
      │ │ │ │      │ │ │ │      │ │ │ │      │ │ │ │
    ┌─┼─┼─┼─┼──────┼─┼─┼─┼──────┼─┼─┼─┼──────┼─┼─┼─┼─┐
    │ │ │ │ │      │ │ │ │      │ │ │ │      │ │ │ │ │
    GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
    │                                               │
    └─── 每块 GPU 连接到所有 4 颗 NVSwitch ───────────┘

    每块 GPU: 18 条 NVLink 4.0 = 900 GB/s 双向
    域内任意两块 GPU: 单跳, 延迟对称
    域大小: 8 GPU
```

关键特性：NVSwitch 构建了一个**非阻塞全连接网络**——域内任意两块 GPU 之间只有一跳，带宽完全对称。这让 NCCL 的集合通信实现非常简单：不需要感知拓扑，所有 GPU 等距。

### NVL72：机架级扩展（Blackwell）

GB200 NVL72 是一个质的飞跃——NVLink 域从节点内 8 GPU 扩展到机架级 72 GPU：

```
GB200 NVL72 机架拓扑

┌─────────────────────────────────────────────────┐
│                   NVL72 机架                      │
│                                                   │
│  ┌──────────────────────────────────────────────┐│
│  │    9 个 NVSwitch Tray (共 18 颗 NVSwitch 5)  ││
│  └────────────┬──────────────┬──────────────────┘│
│               │              │                    │
│  ┌────────────┼──────────────┼──────────────────┐│
│  │            │              │                   ││
│  │  18 个 Compute Tray                          ││
│  │  每个 Tray: 1 Grace CPU + 2 Blackwell GPU    ││
│  │  共 36 Grace + 72 Blackwell GPU              ││
│  │                                               ││
│  │  重要: Tray 内 GPU 之间没有直连 NVLink        ││
│  │  所有 GPU 通信都经过 NVSwitch 背板            ││
│  └───────────────────────────────────────────────┘│
│                                                   │
│  互联: 铜缆背板 (无光模块)                         │
│  每块 GPU: 1,800 GB/s 双向 NVLink 5.0             │
│  域聚合带宽: 130 TB/s                              │
│  域大小: 72 GPU, 单跳全连接                         │
└─────────────────────────────────────────────────┘
```

架构要点：

- **18 颗 NVSwitch 5.0** 构成多平面交换结构：每块 GPU 的 18 条 NVLink 各连一颗 NVSwitch，实现 72 GPU 的单跳全连接
- **Compute Tray 内无 GPU 直连**：所有 GPU-to-GPU 通信必须经过外部 NVSwitch 背板——这与 DGX H100 的设计完全不同
- **纯铜缆**：机架内距离短，不需要光模块
- 也有 NVL36x2 变体：72 GPU 分布在 2 个机架，用 36 颗 NVSwitch 维持全连接

### NVL576：多机架扩展（Rubin Ultra）

Rubin Ultra 的 Kyber 机架将域进一步扩展到 **576 chiplet**（144 个 GPU 封装，每个封装 4 个 chiplet）：

| 参数 | NVL8 (Hopper) | NVL72 (Blackwell) | NVL576 (Rubin Ultra) |
|------|:-:|:-:|:-:|
| 域大小 | 8 GPU | 72 GPU | 576 chiplet (144 封装) |
| 每 GPU 带宽 | 900 GB/s | 1,800 GB/s | 3,600 GB/s |
| 域聚合带宽 | ~7.2 TB/s | 130 TB/s | 1.5 PB/s |
| NVSwitch 数 | 4 | 18 | — |
| 物理范围 | 单节点 | 单机架 | 单 Kyber 机架 |
| 互联介质 | 铜缆 | 铜缆背板 | PCB 背板 |

Kyber 机架的创新在于将 compute tray 旋转 90° 变成 blade 形态，用 PCB 板代替铜缆背板以支持更高密度。但 576 chiplet 的全连接对 NVSwitch 的端口数和布线复杂度提出了巨大挑战——这也是 NVIDIA 开始认真考虑光互联的原因。

### Scale-up vs Scale-out 的带宽断崖

NVIDIA 架构中最关键的结构性问题是 **NVLink 域内外的带宽悬崖**：

```
NVIDIA 带宽层次 (Blackwell)

  NVLink 域内                    域外 (Scale-out)
  ┌─────────────────────┐       ┌─────────────────────┐
  │                     │       │                     │
  │  1,800 GB/s / GPU   │──────→│  100-200 GB/s / GPU │
  │  (NVLink 5.0)       │       │  (InfiniBand 800G)  │
  │                     │       │                     │
  │  单跳, 亚微秒延迟    │       │  多跳, 数微秒延迟    │
  │  全连接, 对称        │       │  Fat-tree, 非对称    │
  │                     │       │                     │
  └─────────────────────┘       └─────────────────────┘

          带宽比: ~18×
```

这个 18× 的断崖意味着：**跨域通信是性能的第一大杀手**。这解释了为什么 NVIDIA 持续扩大 NVLink 域——从 8 到 72 到 576，目标就是让更多的 GPU 待在高带宽域内。

<details>
<summary>深入：Rail-Optimized 拓扑</summary>

在 Scale-out 层（多个 NVLink 域之间），NVIDIA 使用 **Rail-Optimized** 设计来最大化跨节点通信效率：

```
Rail-Optimized 拓扑示意

  Node 0          Node 1          Node 2
  GPU0 ─────────── GPU0 ─────────── GPU0  ← Rail 0 (Leaf Switch 0)
  GPU1 ─────────── GPU1 ─────────── GPU1  ← Rail 1 (Leaf Switch 1)
  GPU2 ─────────── GPU2 ─────────── GPU2  ← Rail 2 (Leaf Switch 2)
  ...              ...              ...
  GPU7 ─────────── GPU7 ─────────── GPU7  ← Rail 7 (Leaf Switch 7)
```

每个节点的 GPU 0 都连到同一个 Leaf Switch（Rail 0），GPU 1 连到 Rail 1，以此类推。这种设计利用了数据并行训练中 AllReduce 操作的特征——同 rank 的 GPU 之间通信量最大，而 rail-optimized 保证了同 rank GPU 之间只有单交换机跳。

跨 rail 的通信（不同 rank 的 GPU）则需要经过 Spine 交换机，跳数更多、带宽更低。这就要求训练框架的并行策略必须与物理拓扑对齐——Tensor Parallelism 放在 NVLink 域内，Data Parallelism 放在 Rail 方向，Pipeline Parallelism 放在 Spine 方向。

> **Insight**：Rail-Optimized 设计本质上是在承认带宽不均匀的现实——与其追求理想的全连接，不如让最常见的通信模式走最快的路径。这和 TPU 3D Torus 上的"拓扑感知数据放置"是同一个思路，只是实现层面不同：NVIDIA 靠物理布线 + 并行策略配合，TPU 靠 XLA 编译器自动优化。

</details>

---

## 两种路线的本质对比

| 维度 | NVIDIA (NVLink + NVSwitch) | Google TPU (ICI + 3D Torus + OCS) |
|------|:-:|:-:|
| **拓扑类型** | 交换机全连接 (Clos/Fat-tree) | 无交换机 3D Torus 直连 |
| **域内跳数** | 1 跳 (任意 GPU 等距) | 多跳 (最远 (3/2)∛N 跳) |
| **每芯片带宽** | 1.8 TB/s (Blackwell) | 1.2 TB/s (Ironwood) |
| **最大 Scale-up 域** | 576 chiplet (Rubin Ultra) | 9,216 chips (Ironwood SuperPod) |
| **域外技术** | InfiniBand / Ethernet (~18× 带宽降级) | 同一 ICI 通过 OCS 扩展; DCN 用于 multi-pod |
| **交换硬件** | NVSwitch ASIC (18 颗/NVL72 机架) | 无 (ICI 直连); OCS 用于 inter-cube |
| **可重构性** | 固定拓扑; 拓扑感知调度 | OCS 动态重构; slice 大小/形状可变 |
| **编译器依赖** | 低 (NCCL 假设全连接) | 高 (XLA 必须拓扑感知) |
| **故障恢复** | Checkpoint + 重启 | OCS 动态绕行, 无需停机 |
| **功耗效率** | NVSwitch + InfiniBand 交换机功耗高 | OCS 108W vs EPS 3,000W |
| **典型 MFU** | ~52% (H100, LLM) | ~58% (TPU v4, LLM) |

```
两种互联哲学的根本区别

NVIDIA:                          Google TPU:
"用强力交换芯片让一切变简单"       "用聪明的编译器让硬件保持简单"

  ┌───┐                           ○──○──○──○
  │NV │                          /| /| /| /|
  │Sw │──→ 所有 GPU 等距          ○──○──○──○ |
  │it │                          | ○| ○| ○| ○
  │ch │                          |/ |/ |/ |/
  └───┘                          ○──○──○──○

  单跳, NCCL 不需要               多跳, XLA 编译器
  感知拓扑                        必须优化数据放置

  域外: 18× 带宽断崖              域内到 9,216 chips
  (InfiniBand/Ethernet)           无带宽断崖
```

> **Insight**：这两种路线反映了更深层的工程哲学差异。NVIDIA 的思路是"硬件解决问题"——用 NVSwitch 消除拓扑对软件的可见性，让 NCCL 假设一个扁平的通信世界。Google 的思路是"软件解决问题"——保持硬件简单（没有交换芯片），让 XLA 编译器承担拓扑优化的所有复杂度。两者都在自己的战场上取得了成功。

---

## 交汇点——光互联的未来

### NVIDIA Feynman 与共封装光学

尽管 NVIDIA 长期依赖铜缆互联，但物理限制正在逼近：更高的 lane 速率（>100 Gbps）在更长距离上的信号衰减越来越严重，铜缆的长度/带宽积约束使得跨机架的高速互联越来越困难。

2028 年的 **Feynman 架构**将是 NVIDIA 的光学转折点：

- **NVLink 8.0 + CPO（Co-Packaged Optics，共封装光学）**：使用台积电 COUPE 硅光子技术，将光学收发器直接封装到 NVSwitch 芯片上
- **8 代 NVSwitch** + **Spectrum-7 Ethernet with CPO** + **ConnectX-10 光学 InfiniBand**
- NVLink 域扩展到 **NVL1152**（288 个 GPU 封装 / 1,152 chiplet），跨多个机架

Jensen Huang 在 GTC 2026 上说："我们将同时有铜缆 Scale-up 的 Kyber 和共封装光学 Scale-up 的 Kyber。Feynman 将首次用光学来做 Scale-up。"

### OFC 2026：行业风向标

2026 年 3 月的 OFC（Optical Fiber Communication Conference）上，OCS 成为 AI 基础设施的核心议题：

**NVIDIA 的动作**：
- NVIDIA VP Alexis Bjornlin 披露了一个 **GB200 NVL576 原型**：机架内仍然使用铜缆，但**机架间已经切换到光纤**——这很可能就是 Rubin/Feynman 大规模部署的模板
- NVIDIA 研究人员认为 OCS 可以在数据中心交换层次的**所有层级**发挥作用
- NVIDIA 向 **Lumentum 投资 20 亿美元**支持先进光学制造

**OCS 厂商的进展**：

| 厂商 | 产品 | 规格 | 亮点 |
|------|------|------|------|
| Marvell + Lumentum | Aquila 1.6T DSP + R300 OCS | 300×300 端口 | 10 万 XPU 系统节能 65% |
| Coherent | DLX Switch | 64×64 ~ 512×512 端口 | 已出货 7 家客户 |
| Salience Labs | Photonic OCS | 32×32 端口 | SOA 放大光子交换 |
| Triple-Stone | MDC-based OCS | 512×512 端口 | 面向大规模 DC |

OCP（Open Compute Project）还宣布成立新的 OCS 子项目，由 iPronics 和 Lumentum 共同领导，参与者包括 Google、Microsoft 和 NVIDIA。

### CPO vs OCS——两种光学技术的定位

```
光学技术栈定位

                    CPO (Co-Packaged Optics)
                    ┌─────────────────────────┐
                    │ 把光学收发器封装到芯片上  │
                    │ 解决: 芯片→光纤的转换效率  │
                    │ 目标: 降低能耗, 提高密度    │
                    │ 距离: 短到中距 (机架内~间)  │
                    │ NVIDIA Feynman 的核心技术  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    OCS (Optical Circuit Switching)
                    ┌─────────────────────────┐
                    │ 用 MEMS/液晶切换光路     │
                    │ 解决: 大规模光纤的灵活路由  │
                    │ 目标: 可重构拓扑, 低功耗    │
                    │ 距离: 中到长距 (机架间~DC)  │
                    │ Google TPU 的核心技术      │
                    └─────────────────────────┘

两者互补, 不冲突:
CPO 解决 "芯片到光纤" 的最后一公里
OCS 解决 "光纤到光纤" 的灵活路由
```

CPO 和 OCS 解决的是互联栈中不同层级的问题。CPO 将光学收发器从插拔模块移到芯片封装内，消除电连接器损耗，提高每瓦特带宽。OCS 在光纤层面提供灵活的路由交换。**未来的 AI 集群很可能同时使用两者**：芯片用 CPO 出光纤，光纤之间用 OCS 做灵活路由。

### NVL1152——跨机架光学 NVLink 域

如果 Feynman 的 CPO + OCS 如期落地，NVIDIA 的互联架构将发生根本性变化：

| 里程碑 | 域大小 | 互联方式 | 物理范围 |
|------|:-:|------|:-:|
| NVL8 (2022) | 8 GPU | 铜缆 NVLink | 节点内 |
| NVL72 (2024) | 72 GPU | 铜缆背板 | 单机架 |
| NVL576 (2027) | 576 chiplet | PCB 背板 + 机架间光纤 | 单 Kyber 机架 |
| NVL1152 (2028) | 1,152 chiplet | CPO + OCS | 跨机架 |

NVL1152 将 288 个 GPU 封装分布在多个机架中，用 CPO 光学 NVLink 连接，中间可能用 OCS 做光路交换。这标志着 **NVIDIA 首次在 Scale-up 层面采用与 Google TPU 类似的光交换技术**。

<details>
<summary>深入：NVIDIA 会走向 Torus 吗？</summary>

随着 NVLink 域扩展到 576 甚至 1,152 chiplet，全连接 (fully connected) 拓扑的 NVSwitch 端口需求和布线复杂度呈二次方增长。一些分析师认为 NVIDIA 可能在 Rubin Ultra / Feynman 时代从全连接 Clos 拓扑转向 **Torus 或 Dragonfly** 拓扑，用 OCS 作为 spine 连接各域。

如果这真的发生，将是一个历史性的时刻——NVIDIA 用了十年的"交换机全连接"哲学可能向 Google 的"直连 + 光交换"方向靠拢。当然，NVIDIA 的优势在于 NCCL 和 CUDA 生态已经足够成熟，即使底层拓扑变化，上层软件也能相对平滑地过渡。

> **Insight**：拓扑的选择最终是一个工程权衡——全连接提供最低延迟和最简单的编程模型，但布线成本随规模二次增长；Torus 提供最高的可扩展性，但需要拓扑感知的通信优化。当域大小还在 8-72 时，全连接的代价可以承受；但到了 576-1,152，Torus 或 Dragonfly 可能成为必然选择。

</details>

### Google 的下一步

Google 也没有停步。在 OCS 技术层面，Google 正在探索用**压电（piezo）替代 MEMS** 来驱动 OCS 微镜——压电技术在插入损耗、回波损耗方面有先天优势，且可能实现更快的切换速度。

TPU v7 (Ironwood) 已经将 OCS 端口从 136×136 升级到 144×144，支持 9,216 芯片的 SuperPod。更大规模的扩展可能需要更高端口密度的 OCS 或多级 OCS 网络。

---

## 总结

AI 集群互联的故事，本质上是两种工程哲学之间的竞赛：

**NVIDIA** 选择了"硬件暴力美学"——用 NVSwitch 这颗专用交换芯片消除拓扑复杂度，让域内所有 GPU 看起来等距，大幅简化了软件（NCCL）的工作。代价是域的物理边界非常硬——一旦跨出 NVLink 域，带宽直降 18 倍。

**Google** 选择了"软件优雅路线"——没有任何交换芯片，每颗 TPU 直连 6 个邻居组成 3D Torus，用 OCS 光交换在机架间灵活布线。代价是编译器（XLA）必须深度理解拓扑并优化数据放置。但回报是从 64 到 9,216 芯片的无缝扩展，没有带宽断崖。

最有趣的是，**两条路线正在交汇**：NVIDIA 在 Feynman 架构中引入 CPO 和 OCS，本质上是在承认纯铜缆 + 交换机的路线无法无限扩展；而 Google 也在持续提升 ICI 带宽和 OCS 端口密度。到 2028 年，AI 集群的互联技术栈很可能演变为：

- **芯片级**：CPO 共封装光学
- **机架级**：铜缆/短距光纤直连
- **机架间**：OCS 光电路交换
- **数据中心**：电交换（InfiniBand/Ethernet）+ OCS 混合

互联，而非算力，正在成为 AI 集群规模化的终极瓶颈。谁能在光互联时代率先交付大规模、高效率、低功耗的解决方案，谁就能在下一轮 AI 基础设施竞赛中占据先机。
