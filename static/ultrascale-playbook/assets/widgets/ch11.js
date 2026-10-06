/* 第 11 章交互元件：全书回顾小测验（10 题单选） */
(function () {
  'use strict';

  // ---------- 章节数据（推荐重读连结用） ----------
  const CHAPTERS = {
    ch01: '第 1 章 第一步：单 GPU 训练',
    ch02: '第 2 章 数据并行',
    ch03: '第 3 章 张量并行',
    ch04: '第 4 章 上下文并行',
    ch05: '第 5 章 流水线并行',
    ch06: '第 6 章 专家并行',
    ch10: '第 10 章 融合内核、Flash Attention 与混合精度',
  };

  // ---------- 题库（ans 为正解索引；ch 为对应章节） ----------
  const QUESTIONS = [
    {
      q: '在 bf16 混合精度搭配 Adam 训练时，「优化器状态」（fp32 主权重＋一阶动量＋二阶动量）每个参数共需多少位元组？',
      opts: ['4 位元组', '8 位元组', '12 位元组', '16 位元组'],
      ans: 2, ch: 'ch02',
      why: 'fp32 主权重、一阶动量、二阶动量各占 4 位元组，合计 12Ψ；再加上 bf16 参数 2Ψ 与 bf16 梯度 2Ψ，整体约 16Ψ（不含 fp32 梯度累积）。',
    },
    {
      q: '使用「完全」激活重算（full activation recomputation）节省内存，主要代价是什么？',
      opts: [
        'GPU 之间的通信量大幅增加',
        '反向传播时要重跑一次前向，计算时间增加约 30–40%',
        '梯度会变得不精确，必须调低学习率',
        '优化器状态的内存占用加倍',
      ],
      ans: 1, ch: 'ch01',
      why: '重算是用计算换内存：只存少数检查点、反向时即时重算激活。选择性重算更划算——GPT-3 只花 2.7% 计算就省下 70% 激活内存。',
    },
    {
      q: 'ZeRO-2 沿数据并行维度分片（shard）了哪些东西？',
      opts: [
        '只有优化器状态',
        '优化器状态＋梯度',
        '优化器状态＋梯度＋模型参数',
        '模型参数＋激活值',
      ],
      ans: 1, ch: 'ch02',
      why: 'ZeRO-1 切优化器状态；ZeRO-2 再加上梯度（把 all-reduce 换成 reduce-scatter）；ZeRO-3／FSDP 才进一步切参数。激活值是 ZeRO 切不了的。',
    },
    {
      q: '张量并行（TP）为什么通常限制在单一节点内（例如 TP ≤ 8）？',
      opts: [
        '跨节点时矩阵切分无法对齐，计算结果会出错',
        'NCCL 不支援跨节点的 all-reduce 操作',
        '跨节点会让激活值内存爆增',
        'TP 每层都要通信且难与计算重叠，跨节点带宽远低于 NVLink，会严重拖慢',
      ],
      ans: 3, ch: 'ch03',
      why: 'TP 的通信位于关键路径、每层前后都要做，得靠节点内 NVLink 的高带宽才撑得住；一跨到节点间网络，从 TP=8 到 TP=16 吞吐量就大幅下滑。',
    },
    {
      q: 'AFAB／1F1B 流水线排程中，有 p 个流水线阶段、m 个微批次时，气泡时间占理想计算时间的比例是？',
      opts: ['p − 1', '(p − 1) / m', 'm / (p − 1)', '(m − 1) / p'],
      ans: 1, ch: 'ch05',
      why: '气泡固定为 (p−1)·(t_f+t_b)，理想计算时间为 m·(t_f+t_b)，故比例为 (p−1)/m——增加微批次数 m 可以把气泡摊薄。',
    },
    {
      q: 'Ring Attention 中，各 GPU 在「环」上依序传递给下一张 GPU 的是什么？',
      opts: ['查询（Q）分块', '键与值（K/V）分块', '注意力分数矩阵', '各层的梯度'],
      ans: 1, ch: 'ch04',
      why: '每张 GPU 保留自己的 Q 分块不动，一边计算局部注意力、一边把 K/V 传给环上的下一张 GPU，让通信与计算重叠起来。',
    },
    {
      q: '训练 MoE 模型时，专家并行（EP）的 all-to-all 通信发生在哪里？',
      opts: [
        'MoE 层前后：router 把 token 分派给各 GPU 上的专家，算完再收回',
        '注意力层内部，用来交换各头的 K/V',
        '优化器步骤时，用来同步优化器状态',
        '每个 epoch 结束时，用来重新平衡专家',
      ],
      ans: 0, ch: 'ch06',
      why: 'token 由 router 动态指派给散在各 GPU 上的专家：先 all-to-all 分发（dispatch），专家算完再 all-to-all 收回结果（combine）。',
    },
    {
      q: '同时使用梯度累积与数据并行时，全局批次大小 gbs 等于？',
      opts: ['mbs × grad_acc', 'mbs × dp', 'mbs × grad_acc × dp', 'mbs + grad_acc + dp'],
      ans: 2, ch: 'ch02',
      why: 'gbs ＝ 微批次大小 × 梯度累积步数 × DP 度。固定目标 gbs 时，这三个旋钮可以互相取舍（例如加大 dp 就能减少 grad_acc）。',
    },
    {
      q: 'Flash Attention 之所以又快又省内存，关键在于？',
      opts: [
        '改用线性注意力近似，把复杂度降到 O(n)',
        '分块计算、避免把注意力分数矩阵 S 具现化到 HBM，尽量在 SRAM 内完成',
        '以 fp8 低精度计算 softmax 来减少运算量',
        '跳过因果遮罩下三角以外的所有计算',
      ],
      ans: 1, ch: 'ch10',
      why: '朴素实现得把巨大的 S、P 矩阵写回慢速 HBM 再读回；Flash Attention 分块计算、只保留 softmax 所需统计量，是精确计算而非近似。',
    },
    {
      q: '与 fp16 相比，bf16 的主要优势是什么？',
      opts: [
        '尾数位更多，数值精度比 fp16 高',
        '占用内存只有 fp16 的一半',
        '在所有 GPU 上运算速度都是 fp16 的两倍',
        '指数位和 fp32 一样是 8 位，动态范围与 fp32 相同，较不易溢位／下溢',
      ],
      ans: 3, ch: 'ch10',
      why: 'bf16 牺牲尾数（7 位，精度其实低于 fp16 的 10 位）换取 fp32 等级的动态范围，因此通常不需要 loss scaling 就能稳定训练。',
    },
  ];

  // ---------- 小工具 ----------
  function el(tag, attrs, children) {
    const n = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === 'text') n.textContent = attrs[k];
      else if (k === 'html') n.innerHTML = attrs[k];
      else if (k === 'style') n.style.cssText = attrs[k];
      else n.setAttribute(k, attrs[k]);
    }
    (children || []).forEach((c) => n.appendChild(c));
    return n;
  }
  // 简单 LCG 伪随机洗牌
  function shuffled(arr) {
    let seed = (Date.now() % 2147483647) || 42;
    const rnd = () => (seed = (seed * 48271) % 2147483647) / 2147483647;
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }
  function chapterLink(ch) {
    return el('a', { href: ch + '.html', text: CHAPTERS[ch] });
  }

  // ---------- 元件本体 ----------
  window.ChapterWidget = {
    title: '全书回顾小测验',
    intro: '10 题单选题，涵盖单 GPU 训练到 5D 并行与 GPU 核心的核心概念。每题作答后立即显示解析；全部答完会依错题推荐值得重读的章节。',
    render(root) {
      const state = { order: [], idx: 0, score: 0, wrong: [] };

      // 进度列
      const progText = el('div', { style: 'font-size:.88rem; color:var(--fg-muted); margin-bottom:.35rem; font-variant-numeric:tabular-nums;' });
      const progFill = el('div', { style: 'height:100%; width:0; background:var(--accent); border-radius:99px; transition:width .25s;' });
      const progBar = el('div', { style: 'height:8px; background:var(--panel-2); border:1px solid var(--border); border-radius:99px; overflow:hidden;' }, [progFill]);
      const progPanel = el('div', { style: 'margin-bottom:1rem;' }, [progText, progBar]);

      // 题目面板
      const qPanel = el('div', { class: 'widget-panel' });
      root.appendChild(progPanel);
      root.appendChild(qPanel);

      function updateProgress() {
        const done = state.idx;
        progText.textContent = '进度 ' + Math.min(done, QUESTIONS.length) + ' / ' + QUESTIONS.length + ' 题 · 目前答对 ' + state.score + ' 题';
        progFill.style.width = (done / QUESTIONS.length * 100) + '%';
      }

      function showQuestion() {
        updateProgress();
        qPanel.textContent = '';
        const item = state.order[state.idx];
        qPanel.appendChild(el('div', { style: 'font-size:.8rem; color:var(--accent); font-weight:600; margin-bottom:.4rem;', text: '第 ' + (state.idx + 1) + ' 题' }));
        qPanel.appendChild(el('div', { style: 'font-weight:600; margin-bottom:.8rem; line-height:1.6;', text: item.q }));

        const optWrap = el('div', { style: 'display:flex; flex-direction:column; gap:.5rem;' });
        const btns = item.opts.map((opt, i) => {
          const b = el('button', { class: 'secondary', style: 'width:100%; text-align:left; white-space:normal; line-height:1.5;', text: String.fromCharCode(65 + i) + '. ' + opt });
          b.addEventListener('click', () => answer(item, i, btns));
          optWrap.appendChild(b);
          return b;
        });
        qPanel.appendChild(optWrap);
      }

      function answer(item, picked, btns) {
        const correct = picked === item.ans;
        if (correct) state.score++;
        else state.wrong.push(item);

        btns.forEach((b, i) => {
          b.disabled = true;
          if (i === item.ans) {
            b.style.borderColor = 'var(--accent)';
            b.style.background = 'var(--accent-soft)';
            b.style.color = 'var(--fg)';
            b.style.fontWeight = '600';
            b.textContent = '✓ ' + b.textContent;
          } else if (i === picked) {
            b.style.opacity = '.6';
            b.style.textDecoration = 'line-through';
            b.textContent = '✗ ' + b.textContent;
          } else {
            b.style.opacity = '.55';
          }
        });

        const fb = el('div', { style: 'margin-top:.9rem; padding:.7rem .9rem; border-left:3px solid var(--accent); background:var(--accent-soft); border-radius:0 8px 8px 0; font-size:.9rem; line-height:1.65;' });
        fb.appendChild(el('div', { style: 'font-weight:700; margin-bottom:.25rem;', text: correct ? '✔ 答对了！' : '✘ 答错了，正解是 ' + String.fromCharCode(65 + item.ans) + '。' }));
        const detail = el('div', { text: item.why + '（见' });
        detail.appendChild(chapterLink(item.ch));
        detail.appendChild(document.createTextNode('）'));
        fb.appendChild(detail);
        qPanel.appendChild(fb);

        const last = state.idx === QUESTIONS.length - 1;
        const nextBtn = el('button', { style: 'margin-top:.9rem;', text: last ? '查看总结 →' : '下一题 →' });
        nextBtn.addEventListener('click', () => {
          state.idx++;
          if (last) showSummary(); else showQuestion();
        });
        qPanel.appendChild(nextBtn);
        state.idx++; updateProgress(); state.idx--; // 进度以「已作答数」计
      }

      function showSummary() {
        updateProgress();
        qPanel.textContent = '';
        const s = state.score, n = QUESTIONS.length;
        const msg = s === n ? '满分！你已经把 Ultra-Scale Playbook 融会贯通了 🎉'
          : s >= 8 ? '非常扎实！只差一点点就全对了。'
          : s >= 6 ? '基础不错，针对错题章节再复习一轮吧。'
          : '别气馁——分布式训练本来就环环相扣，照下面的清单重读最有效率。';
        qPanel.appendChild(el('div', { style: 'font-size:.8rem; color:var(--accent); font-weight:600; margin-bottom:.4rem;', text: '测验总结' }));
        qPanel.appendChild(el('div', { style: 'font-size:2rem; font-weight:700; font-variant-numeric:tabular-nums;', text: s + ' / ' + n }));
        qPanel.appendChild(el('div', { style: 'color:var(--fg-muted); margin:.3rem 0 1rem; line-height:1.6;', text: msg }));

        if (state.wrong.length) {
          qPanel.appendChild(el('div', { style: 'font-weight:600; margin-bottom:.4rem;', text: '建议重读的章节：' }));
          const seen = [];
          state.wrong.forEach((it) => { if (!seen.includes(it.ch)) seen.push(it.ch); });
          seen.sort();
          const ul = el('ul', { style: 'margin:0 0 1rem 1.2rem; line-height:1.9;' });
          seen.forEach((ch) => {
            const cnt = state.wrong.filter((it) => it.ch === ch).length;
            const li = el('li');
            li.appendChild(chapterLink(ch));
            li.appendChild(el('span', { style: 'color:var(--fg-muted); font-size:.85rem;', text: '（错 ' + cnt + ' 题）' }));
            ul.appendChild(li);
          });
          qPanel.appendChild(ul);
        }

        const row = el('div', { class: 'widget-row' });
        const retry = el('button', { text: '↻ 重新测验（重新洗牌）' });
        retry.addEventListener('click', start);
        row.appendChild(retry);
        qPanel.appendChild(row);
      }

      function start() {
        state.order = shuffled(QUESTIONS);
        state.idx = 0; state.score = 0; state.wrong = [];
        showQuestion();
      }
      start();
    },
  };
})();
