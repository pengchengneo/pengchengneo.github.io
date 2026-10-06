---
title: "Ultra-Scale Playbook 简体中文版：在 GPU 集群上训练大语言模型"
date: 2026-10-06
draft: false
tags: ["GPU", "LLM", "Training", "Distributed", "Parallelism"]
categories: ["技术"]
summary: "Ultra-Scale Playbook 简体中文交互版，包含完整章节、附录和 16 个互动实验，覆盖 DP、TP、SP、CP、PP、EP、GPU 内核与混合精度。"
---

**[开始阅读完整简体中文交互版](/ultrascale-playbook/)**

本书从单 GPU 训练出发，逐步讲解在大规模 GPU 集群上训练大语言模型的方法。完整手册保留 12 个主要章节、4 个附录、参考资源及 16 个交互实验。

## 阅读目录

- [导论与全书概览](/ultrascale-playbook/chapters/ch00.html)
- [单 GPU 训练](/ultrascale-playbook/chapters/ch01.html)
- [数据并行与 ZeRO](/ultrascale-playbook/chapters/ch02.html)
- [张量并行与序列并行](/ultrascale-playbook/chapters/ch03.html)
- [上下文并行](/ultrascale-playbook/chapters/ch04.html)
- [流水线并行](/ultrascale-playbook/chapters/ch05.html)
- [专家并行](/ultrascale-playbook/chapters/ch06.html)
- [5D 并行速览](/ultrascale-playbook/chapters/ch07.html)
- [寻找最佳训练配置](/ultrascale-playbook/chapters/ch08.html)
- [GPU 架构与内核](/ultrascale-playbook/chapters/ch09.html)
- [内核融合、Flash Attention 与混合精度](/ultrascale-playbook/chapters/ch10.html)
- [结语](/ultrascale-playbook/chapters/ch11.html)
- 附录：[集体通信](/ultrascale-playbook/chapters/appa.html)、[性能分析](/ultrascale-playbook/chapters/appb.html)、[典型尺度](/ultrascale-playbook/chapters/appc.html)、[计算与通信重叠](/ultrascale-playbook/chapters/appd.html)
- [参考资源](/ultrascale-playbook/chapters/references.html)

## 版本与来源

原作：[The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook)，作者为 Hugging Face nanotron 团队。

中文底本：[Twinkle AI Community 繁体全译本](https://github.com/ai-twinkle/ultrascale-playbook-zh-tw)。本简体版基于上游 `c693694` 转换，统一大陆常用技术术语，并将 column／row 分别对应为“列／行”。正文、页面和交互实验同步本地化，原有英文插图保留。

[简体版源码](https://github.com/pengchengneo/ultrascale-playbook-zh-tw)，版本 `8b4e18d`。这是繁简转换与术语本地化版本，尚未对原译文逐句重新翻译或审校。保留原作者、Twinkle 翻译署名及 Apache 2.0 授权。

## 三部曲已完整发布

书中“敬请期待第三部”是出版时的旧表述。目前可依次阅读：

1. [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)：预训练数据集的构建与处理。
2. [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：GPU 集群和分布式训练。
3. [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)：模型架构、数据配比、预训练与后训练的完整实践。
