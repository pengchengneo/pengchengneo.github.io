---
title: "How to Scale Your Model：大模型扩展完整指南"
date: 2026-04-29
draft: false
tags: ["JAX", "TPU", "GPU", "LLM", "Training", "Inference", "Scaling"]
categories: ["TPU"]
summary: "《How to Scale Your Model》完整中文翻译，覆盖 Roofline 分析、TPU/GPU 架构、分片矩阵乘法、Transformer 并行策略、LLaMA 训练与推理、性能分析等全部内容"
hideChildPages: true
---

> 原文：https://jax-ml.github.io/scaling-book/
>
> 本中文翻译覆盖全部 13 个章节，保留完整代码、图片和公式。

---

## 入门

- [Introduction（引言）](part0_introduction) — 全书概览与背景

## 基础概念

- [Intro to Rooflines（Roofline 模型入门）](part1_roofline) — 性能分析的基础工具
- [All About TPUs（TPU 详解）](part2_tpus) — TPU 架构、内存层次、互联拓扑
- [Sharded Matmuls（分片矩阵乘法）](part3_sharding) — 数据并行、模型并行、分片策略

## 模型与训练

- [Transformers](part4_transformers) — Transformer 架构与并行化
- [Training（训练）](part5_training) — 分布式训练策略
- [Training LLaMA（训练 LLaMA）](part6_applied_training) — 实战：训练 LLaMA 模型

## 推理与服务

- [Inference（推理）](part7_inference) — 推理优化与量化
- [Serving LLaMA（部署 LLaMA）](part8_applied_inference) — 实战：LLaMA 推理服务

## 工具与进阶

- [Profiling（性能分析）](part9_profiling) — 性能调优与工具
- [All About JAX（JAX 详解）](part10_jax) — JAX 核心概念与高级用法

## 扩展

- [Conclusions（总结）](part11_conclusion) — 总结与展望
- [GPUs](part12_gpus) — GPU 架构与对比
