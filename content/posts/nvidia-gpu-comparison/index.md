---
title: "GPU vs TPU 全家桶横评：从 A100 到 B300，从 TPU v2 到 v7x"
date: 2026-04-14
draft: false
tags: ["GPU", "TPU", "NVIDIA", "HBM", "NVLink"]
categories: ["GPU"]
summary: "对比 NVIDIA A100~B300 七款 GPU 与 Google TPU v2~v7x 七代 TPU 的算力、HBM 显存与带宽、互联带宽，梳理两大阵营的演进脉络"
---

## 核心参数对比

下表汇总了 NVIDIA 近三代数据中心 GPU（Ampere → Hopper → Blackwell）的关键规格，均为 **SXM 版本**。BF16 算力为 **Tensor Core dense（不含 Sparsity）** 数值。

| GPU | 架构 | FP32 (TFLOPS) | BF16 Tensor Core (TFLOPS) | HBM 代 | HBM 容量 | HBM 带宽 | NVLink 代 | NVLink 带宽 |
|-----|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A100** | Ampere | 19.5 | 312 | HBM2e | 80 GB | 2.0 TB/s | 3.0 | 600 GB/s |
| **A800** | Ampere | 19.5 | 312 | HBM2e | 80 GB | 2.0 TB/s | 3.0 | 400 GB/s |
| **H100** | Hopper | 67 | 989 | HBM3 | 80 GB | 3.35 TB/s | 4.0 | 900 GB/s |
| **H200** | Hopper | 67 | 989 | HBM3e | 141 GB | 4.8 TB/s | 4.0 | 900 GB/s |
| **H20** | Hopper | 44 | 148 | HBM3 | 96 GB | 4.0 TB/s | 4.0 | 900 GB/s |
| **B200** | Blackwell | 80 | 2,250 | HBM3e | 192 GB | 8.0 TB/s | 5.0 | 1,800 GB/s |
| **B300** | Blackwell Ultra | 75 | 2,250 | HBM3e | 288 GB | 8.0 TB/s | 5.0 | 1,800 GB/s |

> **关于 NVLink 带宽**：表中所有 NVLink 带宽数值均为 **双向聚合带宽（bidirectional aggregate）**，即单向带宽约为表中数值的一半。这是 NVIDIA 官方 datasheet 的标注惯例。例如 H100 的 NVLink 4.0 标称 900 GB/s 双向，单向约 450 GB/s。

> **关于 BF16 Sparsity**：NVIDIA 的结构化稀疏（2:4 Sparsity）可将 BF16 Tensor Core 吞吐翻倍。例如 H100 BF16 Tensor Core 在 Sparsity 下可达 1,979 TFLOPS，B200 可达 4,500 TFLOPS。表中统一使用 dense 数值以便公平对比。

---

## 架构演进解读

### Ampere：A100 与 A800

A100 于 2020 年发布，采用台积电 7nm 工艺和 HBM2e 显存，是当时 AI 训练的标杆。A800 是面向中国市场的合规版本，**唯一区别是 NVLink 带宽从 600 GB/s 降至 400 GB/s**，算力、显存等规格完全相同。

### Hopper：H100、H200 与 H20

H100（2022）切换到 4nm 工艺，FP32 算力跃升至 67 TFLOPS（3.4× A100），BF16 Tensor Core 达 989 TFLOPS（3.2× A100）。HBM 也从 HBM2e 升级到 HBM3，带宽从 2.0 TB/s 提升到 3.35 TB/s。

**H200** 与 H100 共享同一 GH100 die，算力相同，但换装了 **HBM3e** 显存——容量从 80 GB 增至 141 GB（+76%），带宽从 3.35 TB/s 提升至 4.8 TB/s（+43%）。对于受限于 KV Cache 显存的大模型推理场景，H200 的性价比显著优于 H100。

**H20** 是面向中国市场的出口合规版本，使用了大幅裁剪的 Hopper die（仅保留 78/144 个 SM，-46%），算力大幅缩水。但 NVIDIA 刻意保留了 96 GB HBM3 和 4.0 TB/s 带宽以及完整的 NVLink 4.0（900 GB/s），使其在推理场景中仍具竞争力——高显存 + 高带宽正是 LLM 推理的核心需求。

### Blackwell：B200 与 B300

B200（2024）采用 4NP 工艺双 die 封装，晶体管数量高达 2080 亿。FP32 达到 80 TFLOPS，BF16 Tensor Core 达 2,250 TFLOPS dense（2.3× H100）。HBM3e 显存达 192 GB / 8.0 TB/s，NVLink 升级到第五代，双向带宽翻倍至 1,800 GB/s。

**B300**（Blackwell Ultra，2025）在 B200 基础上进一步优化：
- **显存翻倍级增长**：288 GB HBM3e（采用 12-high 堆叠），+50% vs B200
- **FP4 算力大幅提升**：14,000 TFLOPS dense（+56% vs B200 的 9,000）
- **BF16 算力持平**：2,250 TFLOPS dense，与 B200 相同
- **FP64 大幅缩减**：仅 1.25 TFLOPS（B200 为 37 TFLOPS），明确面向推理而非 HPC
- **功耗上升**：TDP 1,400W（B200 为 1,000W）

B300 的设计哲学很清晰：**将 die 面积从 FP64 和通用计算转移到 FP4/FP8 Tensor Core 和显存**，全力服务大模型推理。

---

## Google TPU 核心参数对比

下表汇总了 Google TPU 七代产品的 per-chip 关键规格。TPU 没有独立的 FP32 CUDA Core，矩阵计算全部通过 MXU（Matrix Multiply Unit）完成，原生精度为 BF16 乘 / FP32 累加。

| TPU | 年份 | BF16 (TFLOPS) | FP8 (TFLOPS) | HBM 代 | HBM 容量 | HBM 带宽 | ICI 带宽 | ICI 拓扑 |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **v2** | 2017 | 45 | — | HBM | 16 GB | 600 GB/s | — | 2D Torus |
| **v3** | 2018 | 123 | — | HBM | 32 GB | 900 GB/s | — | 2D Torus |
| **v4** | 2021 | 275 | — | HBM2e | 32 GB | 1.2 TB/s | — | 3D Torus |
| **v5e** | 2023 | 197 | — | HBM2e | 16 GB | 819 GB/s | — | 2D Torus |
| **v5p** | 2023 | 459 | — | HBM2e | 95 GB | 2.76 TB/s | 1,200 GB/s | 3D Torus |
| **v6e (Trillium)** | 2024 | 918 | — | HBM | 32 GB | 1.64 TB/s | 800 GB/s | 2D Torus |
| **v7x (Ironwood)** | 2025 | 2,307 | 4,614 | HBM3e | 192 GB | 7.37 TB/s | 1,200 GB/s | 3D Torus |

> **关于 ICI 带宽**：表中 ICI（Inter-Chip Interconnect）带宽均为 **双向聚合带宽（bidirectional）**，与 NVLink 标注惯例一致。早期 TPU（v2~v4, v5e）的 ICI 带宽未被 Google 官方公开。

> **关于 v5e vs v5p**：v5e 是面向推理和轻量训练的性价比版本（1 TensorCore/chip, 2D Torus），v5p 是面向大规模训练的旗舰版本（2 TensorCores/chip, 3D Torus），两者同代但定位不同。

---

## TPU 架构演进解读

### v2 → v3：BF16 的开创者

TPU v2（2017）首次引入了 BF16 数据格式，也是第一代面向训练的 TPU。每个 chip 有 2 个 TensorCore，每个 TensorCore 含 1 个 128×128 MXU，算力 45 TFLOPS。v3（2018）在同样的 16nm 工艺上将每个 TensorCore 的 MXU 翻倍，加上频率提升，算力达到 123 TFLOPS（2.7× v2）。

### v4：进入 7nm，引入 3D Torus

TPU v4（2021）切换到 7nm 工艺，每个 chip 拥有 2 个 TensorCore × 4 个 MXU = 8 个 MXU，算力跃升至 275 TFLOPS。更重要的是引入了 **3D Torus** 互联拓扑，每个 chip 有 6 条 ICI 链路，支持最大 4,096 chip 的超级 Pod。

### v5e / v5p：分化路线

v5e（2023）是 Google 的"性价比"路线——单 TensorCore、16 GB HBM、2D Torus，每 chip 仅 197 TFLOPS，但成本低、适合推理。v5p 则走旗舰路线，95 GB HBM2e + 2.76 TB/s 带宽 + 3D Torus，算力 459 TFLOPS，支持 8,960 chip 级别的大规模训练。

### v6e (Trillium)：MXU 翻四倍

v6e（2024）将 MXU 从 128×128 扩大到 **256×256**，单次矩阵乘法 FLOPS 翻四倍。虽然只有 1 个 TensorCore（含 2 个 MXU），但 BF16 算力达到 918 TFLOPS（4.7× v5e）。不过显存仅 32 GB、ICI 采用 2D Torus，仍然定位于推理和中等规模训练。

### v7x (Ironwood)：推理时代的全面升级

TPU v7x（2025）是 Google 第一款明确以推理为核心目标的 TPU，采用 **双 chiplet 封装**——每个 chiplet 含 1 个 TensorCore + 2 个 SparseCore + 96 GB HBM3e，两个 chiplet 通过 D2D（Die-to-Die）接口连接，速度是单条 ICI 链路的 6 倍。

核心升级：
- **BF16 算力 2,307 TFLOPS**（2.5× v6e），首次支持 **FP8 原生计算 4,614 TFLOPS**
- **192 GB HBM3e**（6× v6e），7.37 TB/s 带宽（4.5× v6e）
- **3D Torus 回归**，ICI 1,200 GB/s 双向（1.5× v6e）
- 最大 Pod 支持 **9,216 chips**，集群算力达 42.5 EFLOPS（FP8）

---

## GPU vs TPU 关键趋势

1. **BF16 算力每代翻 2-3 倍**：GPU 侧 312 → 989 → 2,250 TFLOPS（A100 → H100 → B200），TPU 侧 275 → 459 → 918 → 2,307 TFLOPS（v4 → v5p → v6e → v7x）
2. **HBM 带宽每代翻倍**：GPU 侧 2.0 → 3.35 → 8.0 TB/s，TPU 侧 1.2 → 2.76 → 7.37 TB/s
3. **显存容量增长更快**：GPU 80 → 192 → 288 GB（A100 → B200 → B300），TPU 32 → 95 → 192 GB（v4 → v5p → v7x）
4. **互联带宽稳步提升**：NVLink 600 → 900 → 1,800 GB/s，ICI 800 → 1,200 GB/s（v6e → v7x）
5. **低精度算力成为主战场**：B300 FP4 达 14,000 TFLOPS，v7x FP8 达 4,614 TFLOPS，推理场景下量化计算已成标配
6. **旗舰对位**：B200（BF16 2,250 TFLOPS / 192 GB / 8.0 TB/s）与 v7x（BF16 2,307 TFLOPS / 192 GB / 7.37 TB/s）在单 chip 规格上已高度接近，竞争焦点转向集群互联与软件生态
