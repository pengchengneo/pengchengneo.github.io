---
title: "Conclusions（总结）"
date: 2026-04-29
draft: false
math: true
weight: 12
---

{{< katex >}}

# 总结与延伸阅读（Conclusions and Further Reading）

《How To Scale Your Model》的第 11 部分（[第 10 部分：JAX](../part10_jax) | [第 12 部分：GPU](../part12_gpus)）

感谢阅读！这里我们将提供一些可供进一步学习的参考资料。

**目录**

[致谢（Acknowledgments）](#acknowledgments)

[延伸阅读（Further Reading）](#further-reading)

[反馈（Feedback）](#feedback)

**感谢您完整地阅读了本书，恭喜您坚持到了最后。** 在结束之前，让我们做一些致谢：

## 致谢（Acknowledgments）

本文档凝聚了 Google DeepMind 许多同仁的大量心血，我们在此简要致谢！

- James Bradbury、Reiner Pope 和 Blake Hechtman 最初推导了本书中的许多思想，他们较早地从系统视角理解了 Transformer。
- Sholto Douglas 撰写了本文档的第一个版本，并启动了整个项目。本文档的整体叙事框架很大程度上要归功于他。
- Jacob Austin 主导了将这份初稿从粗略笔记打磨为更精致、更全面的成品的工作。他完成了大量编辑、排版和发布工作，并协调了其他作者的贡献。
- 大多数图表和动画由 Anselm Levskaya 和 Charlie Chen 制作。
- Charlie Chen 撰写了推理（inference）章节，并绘制了许多推理相关的图表。
- Roy Frostig 在出版、编辑以及整个旅程的许多环节中提供了帮助。

我们还要感谢在整个过程中提供了关键反馈的众多同仁，特别是 Zak Stone、Nikhil Sethi、Caitlin Stanton、Alek Dimitriev、Sridhar Lakshmanamurthy、Albert Magyar、Diwakar Gupta、Jeff Dean、Corry Wang、Matt Johnson、Peter Hawkins 等等。同时感谢 Ruiqi Gao 在 HTML 排版方面的帮助。

**感谢大家！**

在您离开之前，您可能还会喜欢阅读关于 NVIDIA GPU 的全新[第 12 部分](../part12_gpus)！

## 延伸阅读（Further Reading）

还有许多相关的写作资料，包括以下内容：

- [**TPU Deep Dive**](https://henryhmko.github.io/posts/tpu/tpu.html)：一篇精彩的 TPU 架构深度剖析文章，与本书的精神一脉相承。
- [**Domain specific architectures for AI inference**](https://fleetwood.dev/posts/domain-specific-architectures)：一篇硬件与模型的深度剖析文章，与本书风格相近。
- [**A Domain-Specific Supercomputer for Training Deep Neural Networks**](https://dl.acm.org/doi/pdf/10.1145/3360307)：最早的 TPU 论文之一，包含许多本书未涉及的 Google TPU 项目细节。
- [**Making Deep Learning Go Brrrr From First Principles**](https://horace.io/brrr_intro.html)：一篇更侧重于 GPU 和 PyTorch 的教程，介绍 LLM 的 roofline 模型与性能工程。
- [**Writing TPU Kernels with Pallas**](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html)：TPU 编程越来越多地涉及在 Pallas 中编写自定义 kernel。这一系列讨论了如何编写 kernel 以及许多本书未提及的更底层 TPU 细节。
- [**How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog**](https://siboehm.com/articles/22/CUDA-MMM)：尽管这是 GPU 和 CUDA 专属的文章，但它是一篇展示如何在 CUDA 中优化 matmul kernel 的优秀博文。这可能是深入理解 TPU 与 GPU 差异的好材料。
- [**Distributed arrays and automatic parallelization**](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)：一份关于 JAX 中并行 API 的优秀指南，是学习如何实际实现本书所讨论思想的好途径。
- [**Rafi Witten's High Performance LLMs 2024 Class**](https://github.com/rwitten/HighPerfLLMs2024)：我们的前同事 Rafi 开设了一门关于 TPU 性能工程的优秀课程，全部幻灯片都在 GitHub 上。它对许多内容的讨论比本书更深入。
- [**\[2211.05102\] Efficiently Scaling Transformer Inference**](https://arxiv.org/abs/2211.05102)：一篇关于 Transformer 推理数学的详细论文，是本文档许多内容的灵感来源。
- [**Huggingface Ultra-Scale Playbook**](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：可视为本书的 GPU 版本，更深入地讨论了 PyTorch 在训练期间如何实现并行技术和节省内存的技术。
- [**Transformer Inference Arithmetic**](https://kipp.ly/transformer-inference-arithmetic/)：一篇博客，包含许多与本书相同的思想和一些精彩的图示。
- [**Stanford CS336 Slides and Videos**](https://stanford-cs336.github.io/spring2025/index.html#coursework)：斯坦福大学一门极佳的课程，涵盖了 LLM 训练与服务的许多细节，并包含一些有用的练习。Assignment 1 和 Assignment 2 尤其相关。
- [**Stas Bekman's ML Engineering Handbook**](https://github.com/stas00/ml-engineering)：一本极具实用性的 ML 基础设施指南，涵盖了本书未讨论的主题，例如如何与云服务商谈判、集群管理以及 GPU 吞吐量的实测数据。

这一领域仍有大量可供全面写作的空间，因此我们希望本手稿能鼓励更多的相关写作！我们也相信这是一个值得研究和探索的富有成果的领域。在许多情况下，即使手头没有许多硬件加速器，也可以开展研究。

## 反馈（Feedback）

请留下您的评论或问题，以便我们进一步改进。您可以通过 jacobaustin123 \[at\] gmail \[dot\] com 联系我们的通讯作者 Jacob Austin，或者通过在 [GitHub](https://github.com/jax-ml/scaling-book) 上发布 issue、pull request 或 discussion 来建议修改。

### 杂项（Miscellaneous）

\*工作完成于 Google DeepMind，现就职于 MatX。

### 引用（Citation）

如需在学术语境中引用本作品，请按以下方式引用：

```
Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.
```

或作为 BibTeX 条目：

```
@article{scaling-book,
  title = {How to Scale Your Model},
  author = {Austin, Jacob and Douglas, Sholto and Frostig, Roy and Levskaya, Anselm and Chen, Charlie and Vikram, Sharad
  and Lebron, Federico and Choy, Peter and Ramasesh, Vinay and Webson, Albert and Pope, Reiner},
  publisher = {Google DeepMind},
  howpublished = {Online},
  note = {Retrieved from https://jax-ml.github.io/scaling-book/},
  year = {2025}
}
```
