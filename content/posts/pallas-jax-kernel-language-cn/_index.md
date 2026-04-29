---
title: "Pallas：JAX 内核语言完整指南"
date: 2026-04-19
draft: false
tags: ["JAX", "Pallas", "TPU", "GPU", "Kernel"]
categories: ["TPU"]
summary: "JAX Pallas 文档完整中文版，覆盖快速入门、Grid/BlockSpec、软件流水线、TPU 内核编写（矩阵乘法、稀疏计算、分布式、SparseCore）、Mosaic GPU 内核编写（Blackwell 矩阵乘法、集合通信）、设计笔记等全部内容"
---

> 原文档：https://docs.jax.dev/en/latest/pallas/
>
> 本中文翻译覆盖 Pallas 文档的所有页面，保留完整代码和图片。

---

## 快速入门

- [Pallas 快速入门](pallas_quickstart_cn) — Hello World、编程模型、Grid、BlockSpec

## 核心概念

- [Grid 和 BlockSpec](pallas_grid_blockspec_cn) — Grid 迭代空间、BlockSpec 切分输入、索引模式
- [软件流水线](pallas_pipelining_cn) — 内存层次、双缓冲、流水线 API、性能分析

## Pallas TPU

- [使用 Pallas 编写 TPU 内核](pallas_tpu_details_cn) — TPU 架构、数组布局、支持的操作
- [TPU 流水线](pallas_tpu_pipelining_cn) — 内存空间、多重缓冲、emit_pipeline、Megacore
- [矩阵乘法](pallas_tpu_matmul_cn) — 分块矩阵乘法、性能分析、bf16、融合激活
- [标量预取与块稀疏计算](pallas_tpu_sparse_cn) — 动态块索引、稀疏矩阵乘法、预取映射
- [TPU 分布式计算](pallas_tpu_distributed_cn) — RDMA、信号量、双缓冲、ppermute/all_gather/psum/psum_scatter
- [核心特定编程](pallas_tpu_coremap_cn) — core_map、逐核心内核、SparseCore 映射
- [SparseCore 内核编写](pallas_tpu_sparsecore_cn) — SparseCore 硬件、流水线、Gather/Scatter
- [伪随机数生成](pallas_tpu_prng_cn) — jax.random API、硬件 PRNG、块不变采样

## Pallas:Mosaic GPU

- [使用 Pallas 编写 Mosaic GPU 内核](pallas_gpu_reference_cn) — SM 架构、内存空间、MMA、core_map、同步、异步拷贝
- [Mosaic GPU 流水线](pallas_gpu_pipelining_cn) — GPU 流水线、Warp 特化、矩阵乘法示例
- [Blackwell 高性能矩阵乘法](pallas_gpu_blackwell_matmul_cn) — 7 步渐进优化、2CTA MMA、持久化内核
- [集合矩阵乘法](pallas_gpu_collective_matmul_cn) — Ring All-Gather、设备间通信、JAX 集成

## 设计笔记

- [Pallas 设计](pallas_design_cn) — 设计理念、JAX 内核扩展、Lowering 流程
- [Pallas 异步操作](pallas_design_async_cn) — 异步 DMA、信号量状态、优化屏障
