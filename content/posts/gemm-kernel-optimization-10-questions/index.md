---
title: "GEMM Kernel 性能优化十问"
date: 2026-04-04
draft: false
tags: ["Pallas", "TPU", "FP8", "Kernel", "MoE"]
categories: ["技术"]
summary: "基于 Pallas GMM FP8 blockwise 量化内核开发的实战问答，涵盖量化粒度、子通道循环、分阶段 tiling、编译链路、精度对齐方法论等核心话题。"
---

基于 Pallas GMM FP8 blockwise 量化内核开发的实战问答。

## Q1: 优化目标与初始挑战

**问：** GMM FP8 内核优化的出发点是什么？最初 FP8 相比 BF16 性能表现如何？瓶颈在哪里？

**答：** 目标是以 blockwise 方式实现 FP8 GMM，对齐 Megatron 的量化方案。初始 FP8 kernel 比 BF16 更慢，根本原因有两个：

1. **Scale 校正开销** — FP8 blockwise 量化需要在 matmul 前后做 scale 乘除（反量化），这些 VPU 运算未能与 MXU 计算有效 overlap
2. **Baseline 强制 `tk == block_size == 128`** — 量化 block 大小是 128，baseline 直接把 K-tile 也限死为 128，导致 MXU 每次计算量太小

因果链：`tk 被限死为 128 → 每次 MXU 计算量太小 → MXU 空闲等待 → 吞吐量低`

## Q2: Blockwise 量化粒度

**问：** `block_size=128` 具体是什么含义？LHS 和 RHS 的量化 block 形状有什么区别？

**答：**

| | LHS（激活） | RHS（权重） |
|---|---|---|
| Block 形状 | 1 × 128 | 128 × 128 |
| Scale 粒度 | 每行每 128 元素一个 scale | 每 128×128 块一个 scale |
| 设计原因 | 激活是动态的，逐行量化适应每个 token 的数值范围 | 权重是静态的，大 block 减少 scale 存储开销，可预计算 |

这种非对称设计与 DeepSeek 的实现对齐，是 FP8 训练的标准做法。

## Q3: 子通道循环（Subchannel Loop）

**问：** Baseline 的 `tk == block_size` 约束是怎么被突破的？子通道循环的核心思路是什么？

**答：** Baseline 的 `tk == block_size` 是过于保守的设计选择（每个 tile 恰好对应一个量化 block），子通道循环将 tiling 与量化**解耦**：

```text
Baseline:   tk == block_size == 128（一个 tile = 一个量化 block）
Subchannel: tk = block_size × num_subchannels（一个 tile 包含多个量化 block）
```

当 `tk=2048, block_size=128` 时，一个 K-tile 有 16 个 subchannel，内层循环独立处理每个 subchannel：取 scale → 反量化 → matmul → 累加。MXU 一次处理更大的 K 维度，利用率提升。

### 延伸：编译器如何利用更大的 tk？

Pallas kernel 由 **Mosaic 编译器**（不是 XLA）编译为 LLO 格式的 VLIW 指令：

```text
Pallas kernel → JAX trace → Jaxpr → Mosaic 编译器 → LLO (VLIW) → TPU 执行
```

更大的 tk 意味着更多的 subchannel 迭代，编译器获得更大的指令窗口，可以：
- **循环展开**：看到更多指令，发现并行机会
- **指令打包**：在 VLIW bundle 中将 VPU（scale 运算）和 MXU（matmul）并行执行
- **DMA 预取**：提前加载下一个 subchannel 数据，与当前计算 overlap

注意：Pallas kernel 对 XLA 是不透明的 `custom_call`，XLA 不优化 kernel 内部。Mosaic 才是编译 kernel body 的编译器。

### 延伸：VLIW 越短 = 性能越好？

**不一定。** VLIW bundle 数量不是可靠的性能指标。反例：将 tm 从 256 增至 512，VLIW 指令数增加，但性能更好——因为每条 MXU 指令处理的数据量翻倍。关键不是 bundle 数量，而是每个 bundle 内各 slot（MXU/VPU/DMA/SFU）的并行利用率。

## Q4: 分阶段 Tiling

**问：** 为什么 fwd、bwd、tgmm 需要不同的 tiling 配置？

**答：** 本质原因：**归约维度不同。**

```text
fwd:  C[M,N] += A[M,K] × B[K,N]     → 归约维度是 K
tgmm: dW[K,N] += X^T[K,M] × dY[M,N] → 归约维度是 M
```

同一个物理维度 M，在 fwd 中是输出维度（并行铺开），在 tgmm 中变成归约维度（逐步加载、累加、转置）。归约维度决定了每步迭代的 VMEM 占用模式。

### tgmm 中 tm=256 为什么导致寄存器压力？

**fwd**（归约步长 tk=128）：每步加载 `A[tm, 128]` + `B[128, tn]`，tm=256 只影响累加器大小，每步搬运量由 tk 决定，不大。

**tgmm**（归约步长 tm）：如果 tm=256，每步需要同时持有：

| 数据 | 大小 (tm=256, tk=128) |
|---|---|
| X 原始 | 256 × 128 = 32K elements |
| X 转置副本 | 128 × 256 = 32K elements |
| dY | 256 × tn |
| 累加器 | 128 × tn |

TPU VMEM 内转置需要同时保留原始和转置副本。类比：fwd 像"每次拿 128 张牌算，结果放 256 个格子"；tgmm 用 tm=256 像"每次抓 256 张牌还要翻转一摞"——手上东西太多。所以 tgmm 需要降低 tm。

生产最优 tiling 验证了这一点：
```text
wi (fwd): (256, 2048, 1024), sk=16
wo (fwd): (256, 512, 2048),  sk=4
tgmm:     更小的 tm 以降低 VMEM 压力
```

## Q5: 零反向量化

**问：** 为什么反向传播可以跳过 FP8 量化用 bf16 直接算，但前向不行？

**答：** 核心是**量化开销的收益比**不同：

| | 前向 | 反向 |
|---|---|---|
| 权重量化 | 离线预计算，开销被摊销 | 激活和梯度都需要**在线量化**，每步都算 scale |
| 量化代价 | scale 运算少，可被 overlap | 两个输入都要在线量化，VPU 开销翻倍 |
| 精度敏感度 | 前向对量化噪声较鲁棒 | 梯度本身有噪声，量化误差**累积**影响训练稳定性 |

反向跳过量化的本质：**在线量化开销大、精度风险高，收益不足以覆盖代价。** 实验验证去掉反向量化后精度提升且性能不降反升。

## Q6: 生产形状推导

**问：** 生产环境下 GMM 的 M、K、N、G 分别是多少？怎么推导的？Dev shape 和生产 shape 是什么关系？

**答：**

**wi 层（gate/up projection）：**
```text
M = per_device_batch × seq_len × top_k = 2 × 4096 × 8 = 65536
K = EMB_DIM = 2048
N = MOE_MLP_DIM = 512
G = NUM_EXPERTS = 256
```

**wo 层（down projection）：** M=65536, K=512, N=2048, G=256（K 和 N 互换）

**Dev shape 的来源：**
```text
生产 shape:         M=65536, G=256
EP=8 分片后每设备:   M=65536/8=8192, G=256/8=32
Dev shape:          M=8192, G=32  ← 直接取 EP=8 的单分片
```

Dev shape 是**故意设为 EP=8 分片后的形状**，确保开发阶段优化的 tiling 可以直接用于生产。

## Q7: HLO 分析与性能归因

**问：** FP8 多了 scale 运算，为什么 HLO 算子数和 BF16 一样？FP8 超越 BF16 的性能增益来自哪里？

**答：**

### 为什么 HLO 算子数一样？

Scale 运算在 **Pallas kernel 内部**。HLO 层面看到的只是 `custom_call` 节点——不管 kernel 里做了什么，从 HLO 角度就是一个算子。差异全部被封装在 kernel 内部，HLO 看不到。

### FP8 性能增益来源

两个因素共同贡献：

1. **MXU 硬件加速** — TPU v7x 的 MXU 对 FP8×FP8 有约 1.81x 吞吐加速（微基准实测：1662 TFLOPS vs BF16 的 918 TFLOPS），仅当两个操作数都是 FP8 时生效
2. **HBM 带宽减半** — FP8 数据量是 BF16 的一半，搬运更快

纯 matmul MXU 加速是 1.81x，但 GMM 实际只有 1.07x-1.24x，差距来自 GMM 的额外开销（scale 运算、group 索引、量化/反量化）。

注意：FP8 **不减少 FLOPs**。FLOPs 是运算次数，由矩阵维度决定（`2×M×K×N`），与精度无关。FP8 减少的是完成这些 FLOPs 所需的内存带宽和计算时间。

## Q8: 进化优化策略

**问：** 为什么用进化算法搜索 tiling？增长趋势说明了什么？

**答：**

### 为什么用进化算法？

**搜索空间太大，网格搜索不可行。** tiling 参数包括 3 个阶段 × 每阶段 3 个维度 + subchannel 数，组合是天文数字：

```text
网格搜索: 10^9 = 10亿种组合（假设每维度 10 个候选）→ 不可行
进化搜索: 每轮 ~5 个变体 × 8 轮 = ~40 次评估 → 达到 2.85x
```

进化算法通过 mutation + selection 高效收敛：基于当前最优解生成有方向的变体（不是随机改数字），保留最快的，淘汰慢的。LLM 驱动进一步提升变异质量——能理解"为什么这个配置快"。

注意：进化算法**不是剪枝搜索**。剪枝是排除法（系统地砍掉子空间），进化是生成法（基于好解生成新解），本质是**有方向的随机采样**，采样分布逐渐收敛到最优区域。

### 增长趋势分析

```text
Round 1-2:  1.015x → 2.294x  （跳跃式增长 → baseline 离最优很远）
Round 3-5:  2.335x → 2.651x  （稳步增长 → 逐步逼近）
Round 6-8:  2.769x → 2.847x  （收敛放缓 → 边际递减）
```

纯 tiling 调整就能 2.85x，说明原始 kernel 的瓶颈不在算法逻辑，而在调度和数据搬运。

## Q9: 精度对齐方法论

**问：** 如何系统性地验证 FP8 kernel 的计算精度？

**答：**

### 9.1 对齐目标与比较对象

精度验证分两层：

| 对比 | 目的 | 预期差异 |
|---|---|---|
| FP8 kernel vs FP8 CPU reference | 验证 kernel 实现正确性（tiling、subchannel 没引入 bug） | 应极小，仅浮点累加顺序差异 |
| FP8 kernel vs BF16 kernel | 量化本身带来的精度损失 | 可接受范围内即可 |

第一层是**正确性验证**，第二层是**量化影响评估**。两者不能混为一谈——如果直接拿 FP8 kernel 对比 BF16，分不清误差来自 kernel 实现还是量化本身。

### 9.2 精度指标体系

#### 绝对/相对误差（atol / rtol）

最基础的指标，逐元素比较：

```python
# allclose 判定：|actual - expected| <= atol + rtol * |expected|
np.testing.assert_allclose(kernel_out, ref_out, atol=1e-3, rtol=1e-2)
```

选取依据：
- FP8 kernel vs FP8 CPU ref：`atol=1e-3, rtol=1e-2`（严格，只允许累加顺序差异）
- FP8 kernel vs BF16 kernel：`atol=1e-1, rtol=5e-2`（宽松，量化本身有精度损失）

局限性：atol/rtol 是逐元素的，无法反映整体分布的偏移趋势。

#### 信噪比（SNR / SQNR）

衡量信号与量化噪声的比值，反映**整体精度质量**：

```text
SQNR = 10 × log10(||signal||² / ||noise||²)
     = 10 × log10(||ref||² / ||ref - kernel||²)
```

- SQNR > 30 dB：精度良好，量化噪声远小于信号
- SQNR 20-30 dB：可接受，需关注是否影响训练收敛
- SQNR < 20 dB：精度有风险，需排查

SNR 的优势在于它是一个**全局标量指标**，不受个别 outlier 元素影响，能直观反映"量化后的结果还有多少有效信息"。

#### ULP（Unit in the Last Place）误差

衡量浮点表示层面的精度距离：

```text
ULP 误差 = |actual - expected| / ULP(expected)
```

其中 `ULP(x)` 是 x 所在浮点区间的最小精度单位。ULP 误差 = 1 意味着结果与参考值相差恰好 1 个最小精度位。

ULP 的意义：
- **与数值大小无关** — 不像 atol 对大值宽松、对小值严格；ULP 在整个数值范围内给出均匀的精度评估
- **直接对应硬件精度极限** — FP8 E4M3 的 ULP 比 BF16 大得多，ULP 误差能直接反映"在该精度下还能做到多好"
- 典型阈值：kernel vs CPU ref 应在 1-2 ULP 以内（仅累加顺序差异），kernel vs BF16 可允许更大的 ULP 差异

#### 指标的配合使用

```text
atol/rtol  → 逐元素守门，快速发现 bug（某个元素算错）
SNR/SQNR   → 全局质量评估，判断整体可用性
ULP        → 精度归因，区分"算法误差"vs"精度极限"
```

三者结合才能完整评估：atol 过了但 SNR 低，说明多数元素偏差不大但整体噪声偏高；atol 没过但 ULP 在 1-2 以内，说明是浮点精度极限导致的，不是 bug。

### 9.3 分层验证策略

精度验证不能只看单次 matmul 输出，需要分层递进：

**Layer 1: 单算子级**
- 单次 GMM 调用，固定输入，对比 kernel vs CPU ref
- 覆盖所有 shape 组合（wi/wo）和所有 tiling 配置
- 指标：atol + ULP

**Layer 2: 单层级（MoE Layer）**
- 完整的 MoE 前向 + 反向，包含 router、expert 选择、combine
- 验证梯度是否正确回传
- 指标：atol + SNR

**Layer 3: 多步训练**
- 跑 100-1000 步训练，对比 FP8 和 BF16 的 loss 曲线
- 关注 loss 是否发散、收敛速度是否一致
- 指标：loss 曲线对比、梯度 SNR 随步数的变化趋势

**Layer 4: 端到端验证**
- 长时间训练（数千步以上），确认 FP8 不影响最终模型质量
- 这一层不再看逐元素精度，而是看训练指标（loss、eval metric）

### 9.4 关闭精度检查的问题

进化优化后期将 atol 设为 1e10（完全无约束）不合理。进化算法可能朝精度不可用的方向优化，浪费搜索预算。更合理的做法是**放宽但不关闭**：

```text
前期:  atol=1e-3, SNR>30dB  （严格，确保基本正确）
中期:  atol=1e-1, SNR>20dB  （放宽，允许更多探索）
后期:  atol=1e-3, SNR>30dB  （收紧，验证最终候选）
```

## Q10: 端到端实施路线

**问：** 在新 TPU 平台上为新模型实现 FP8 GMM kernel，从需求到交付的关键步骤？

**答：**

**Step 0: 了解新平台硬件规格**
- MXU 对 FP8 的吞吐（跑微基准确认硬件加速倍数）
- VMEM 大小（决定 tiling 上限）
- HBM 带宽（判断 kernel 是 compute-bound 还是 memory-bound）

**Step 1: 确定生产 shape**
- 从训练配置推导：per_device_batch、seq_len、top_k → M
- 从模型配置推导：EMB_DIM → K，MOE_MLP_DIM → N，NUM_EXPERTS/EP → G
- 确定 dev shape（取 EP 分片后的单卡 shape）

**Step 2: 实现 FP8 CPU reference**
- 精度正确的参考实现，作为 kernel 对齐目标

**Step 3: 设计 block_size 与量化方案**
- 确定 LHS/RHS 的 block 形状和 scale 粒度
- 决定是否采用零反向量化

**Step 4: 实现 BF16 baseline kernel**
- 性能基准线，没有 baseline 无法量化 FP8 收益

**Step 5: 实现 FP8 kernel 并优化**
- 实现子通道循环（解耦 tk 与 block_size）
- **分阶段**搜索 tiling：fwd/bwd/tgmm 独立优化
- 借助 profiling + LLO 分析 + 进化算法探索
- 全程保持精度对齐 CPU reference

**Step 6: 集成与端到端验证**
- 集成到训练框架，验证端到端性能提升
- 对比训练 loss 曲线，确保收敛性不受影响
