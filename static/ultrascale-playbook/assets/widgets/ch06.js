/* 第 6 章交互元件：MoE 路由与专家并行（EP）可视化 */
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const TOKENS = ['今天', '天气', '模型', '训练', '猫咪', '跳舞', '量子', '翻译'];
  // 写死的伪 router 偏好：[首选专家, 次选专家]（专家 2 刻意较热门，示范负载不均）
  const PREF = [[0, 1], [1, 2], [2, 3], [2, 4], [5, 0], [5, 6], [6, 7], [7, 2]];
  const N = 8, SLOT = 95, CX0 = 47.5; // 8 个栏位，token 与专家垂直对齐
  const state = { ep: 2, k: 2, noise: null, routed: false };
  let svg, statsEl, warnEl, interpEl;

  function sampleNoise() {
    state.noise = TOKENS.map(() => PREF.map(() => Math.random() * 1.5));
  }
  function scores(t) {
    return PREF.map((_, e) => {
      const base = PREF[t][0] === e ? 2.0 : PREF[t][1] === e ? 1.1 : 0;
      return base + state.noise[t][e];
    });
  }
  function topK(t) {
    const s = scores(t);
    return s.map((v, e) => [v, e]).sort((a, b) => b[0] - a[0]).slice(0, state.k).map((p) => p[1]);
  }
  const cx = (i) => CX0 + SLOT * i;                    // 第 i 栏的中心 x
  const gpuOf = (i) => Math.floor(i / (N / state.ep)); // token / 专家所在 GPU

  function S(tag, attrs, text) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    return n;
  }
  function H(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function draw() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const per = N / state.ep;
    // GPU 框
    for (let g = 0; g < state.ep; g++) {
      const x1 = cx(g * per) - 44, x2 = cx((g + 1) * per - 1) + 44;
      svg.appendChild(S('rect', { x: x1, y: 178, width: x2 - x1, height: 104, rx: 10, fill: 'var(--panel)', stroke: 'var(--border)', 'stroke-width': 1.5 }));
      svg.appendChild(S('text', { x: x1 + 10, y: 196, 'font-size': 12, 'font-weight': 700, fill: 'var(--fg-muted)' }, 'GPU ' + g));
    }
    // 路由结果
    let counts = new Array(N).fill(0), cross = 0, total = 0, assign = [];
    if (state.routed) {
      for (let t = 0; t < N; t++) topK(t).forEach((e) => {
        counts[e]++; total++;
        const isCross = gpuOf(t) !== gpuOf(e);
        if (isCross) cross++;
        assign.push([t, e, isCross]);
      });
    }
    // 连线（画在专家节点之下、GPU 框之上）
    assign.forEach(([t, e, isCross]) => {
      const p = S('path', {
        d: 'M ' + cx(t) + ' 60 C ' + cx(t) + ' 120, ' + cx(e) + ' 150, ' + cx(e) + ' 202',
        fill: 'none', 'stroke-width': 1.6, opacity: 0.85,
        stroke: isCross ? 'var(--accent-2)' : 'var(--accent)'
      });
      if (isCross) p.setAttribute('stroke-dasharray', '5 4');
      svg.appendChild(p);
    });
    // token 卡（上方）
    for (let t = 0; t < N; t++) {
      svg.appendChild(S('rect', { x: cx(t) - 36, y: 16, width: 72, height: 44, rx: 8, fill: 'var(--panel)', stroke: 'var(--border)', 'stroke-width': 1.5 }));
      svg.appendChild(S('text', { x: cx(t), y: 36, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 600, fill: 'var(--fg)' }, TOKENS[t]));
      svg.appendChild(S('text', { x: cx(t), y: 52, 'text-anchor': 'middle', 'font-size': 10, fill: 'var(--fg-muted)' }, 'GPU ' + gpuOf(t)));
    }
    // 专家节点与负载条
    const avg = total / N;
    for (let e = 0; e < N; e++) {
      const hot = state.routed && counts[e] > 2 * avg && counts[e] >= 2;
      const col = hot ? 'var(--accent-2)' : 'var(--accent)';
      svg.appendChild(S('rect', { x: cx(e) - 29, y: 202, width: 58, height: 30, rx: 7, fill: 'var(--accent-soft)', stroke: state.routed && counts[e] ? col : 'var(--border)', 'stroke-width': hot ? 2.2 : 1.5 }));
      svg.appendChild(S('text', { x: cx(e), y: 222, 'text-anchor': 'middle', 'font-size': 12, 'font-weight': 700, fill: 'var(--fg)' }, '专家' + e));
      svg.appendChild(S('rect', { x: cx(e) - 32, y: 246, width: 64, height: 9, rx: 4.5, fill: 'var(--code-bg)', stroke: 'var(--border)', 'stroke-width': 1 }));
      if (state.routed && counts[e] > 0) {
        svg.appendChild(S('rect', { x: cx(e) - 32, y: 246, width: 64 * Math.min(counts[e], N) / N, height: 9, rx: 4.5, fill: col }));
      }
      svg.appendChild(S('text', { x: cx(e), y: 272, 'text-anchor': 'middle', 'font-size': 11, fill: hot ? 'var(--accent-2)' : 'var(--fg-muted)', 'font-weight': hot ? 700 : 400 },
        state.routed ? counts[e] + ' 个' + (hot ? ' ⚠' : '') : '—'));
    }
    updateText(counts, cross, total, avg);
  }

  function updateText(counts, cross, total, avg) {
    if (!state.routed) {
      statsEl.textContent = '统计：尚未路由。';
      warnEl.hidden = true;
      interpEl.textContent = '按「路由」，让每个 token 依（写死的伪 router 分数＋随机扰动）选出 top-' + state.k + ' 专家；再试著调整 EP 度与 k，观察跨 GPU 通信量与负载变化。';
      return;
    }
    const pct = total ? Math.round(100 * cross / total) : 0;
    statsEl.textContent = '统计：共 ' + total + ' 个 token→专家指派｜跨 GPU ' + cross + ' 个（' + pct + '%）｜平均负载 ' + avg.toFixed(1) + '、最大 ' + Math.max.apply(null, counts) + '。';
    const hotList = counts.map((c, e) => (c > 2 * avg && c >= 2) ? '专家' + e + '（' + c + ' 个）' : null).filter(Boolean);
    warnEl.hidden = hotList.length === 0;
    warnEl.textContent = '⚠ ' + hotList.join('、') + ' 收到的 token 超过平均的 2 倍：负载不均——这是 MoE 训练的核心难题。若不加以平衡，热门专家会拖慢整步训练，其余专家则闲置。';
    let msg;
    if (state.ep === 1) {
      msg = 'EP=1 时所有专家都在同一颗 GPU 上，路由完全不需跨 GPU 通信；但这颗 GPU 得放下全部 8 个专家的前馈层参数，等于没有分摊内存。';
    } else {
      msg = '目前 EP=' + state.ep + '，8 个专家分散在 ' + state.ep + ' 颗 GPU（每颗 ' + (N / state.ep) + ' 个）。本轮有 ' + pct + '% 的指派要把 token 的隐藏状态送到别颗 GPU 上的专家，这正是 EP 需要 all-to-all 通信的原因——EP 度越大、k 越大，跨 GPU 比例通常越高。';
    }
    msg += ' 由于各专家的前馈层彼此独立，EP 不必像 TP 那样切分矩阵乘法，只需把 token 路由给正确的专家，因此相对轻量；实践中会再搭配 DP 切分输入批次。为了压低通信开销，DeepSeek-V3 便在 router 中限制每个 token 至多送往 M 个节点（其设定为 4），尽量把 token 留在单一节点上。';
    interpEl.textContent = msg;
  }

  function render(rootEl) {
    const panel = H('div', 'widget-panel');
    // 控制列
    const row = H('div', 'widget-row');
    const epLabel = H('label', null, 'EP 度（GPU 数）：');
    const epSel = document.createElement('select');
    [1, 2, 4].forEach((v) => {
      const o = document.createElement('option');
      o.value = v; o.textContent = v + ' 颗 GPU'; if (v === state.ep) o.selected = true;
      epSel.appendChild(o);
    });
    epLabel.appendChild(epSel);
    const kLabel = H('label', null, 'top-k：');
    const kSel = document.createElement('select');
    [1, 2].forEach((v) => {
      const o = document.createElement('option');
      o.value = v; o.textContent = 'k = ' + v; if (v === state.k) o.selected = true;
      kSel.appendChild(o);
    });
    kLabel.appendChild(kSel);
    const routeBtn = H('button', null, '路由');
    const resampleBtn = H('button', 'secondary', '重新抽样');
    row.appendChild(epLabel); row.appendChild(kLabel);
    row.appendChild(routeBtn); row.appendChild(resampleBtn);
    panel.appendChild(row);
    // SVG 画布
    svg = S('svg', { viewBox: '0 0 760 292', role: 'img', 'aria-label': 'MoE 路由与专家并行示意图' });
    svg.style.width = '100%';
    svg.style.height = 'auto';
    svg.style.display = 'block';
    svg.style.marginTop = '.8rem';
    panel.appendChild(svg);
    // 图例
    const legend = H('div', 'widget-row');
    legend.style.cssText = 'margin-top:.5rem;font-size:.82rem;color:var(--fg-muted);gap:1.2rem';
    const mk = (borderStyle, color, label) => {
      const item = H('span');
      const line = H('span');
      line.style.cssText = 'display:inline-block;width:26px;border-top:2px ' + borderStyle + ' ' + color + ';vertical-align:middle;margin-right:.35rem';
      item.appendChild(line);
      item.appendChild(document.createTextNode(label));
      return item;
    };
    legend.appendChild(mk('solid', 'var(--accent)', 'GPU 内路由（免通信）'));
    legend.appendChild(mk('dashed', 'var(--accent-2)', '跨 GPU 路由：需要 all-to-all 通信'));
    panel.appendChild(legend);
    // 统计、警示、解读
    statsEl = H('div');
    statsEl.style.cssText = 'margin-top:.6rem;font-size:.88rem;color:var(--fg)';
    warnEl = H('div');
    warnEl.style.cssText = 'margin-top:.45rem;font-size:.86rem;font-weight:600;color:var(--accent-2);border:1px solid var(--accent-2);border-radius:8px;padding:.5rem .7rem;background:var(--accent-soft)';
    warnEl.hidden = true;
    interpEl = H('div');
    interpEl.style.cssText = 'margin-top:.6rem;font-size:.86rem;line-height:1.7;color:var(--fg-muted);border-left:3px solid var(--accent);padding-left:.7rem';
    panel.appendChild(statsEl); panel.appendChild(warnEl); panel.appendChild(interpEl);
    rootEl.appendChild(panel);
    // 事件
    epSel.addEventListener('change', () => { state.ep = +epSel.value; draw(); });
    kSel.addEventListener('change', () => { state.k = +kSel.value; draw(); });
    routeBtn.addEventListener('click', () => { state.routed = true; draw(); });
    resampleBtn.addEventListener('click', () => { sampleNoise(); state.routed = true; draw(); });
    sampleNoise();
    draw();
  }

  window.ChapterWidget = {
    title: 'MoE 路由与专家并行（EP）',
    intro: '上方是一小批 token，下方是分散在多颗 GPU 上的 8 个专家。按「路由」看每个 token 被送往哪些专家：GPU 内是实线、跨 GPU 是虚线（需要 all-to-all 通信）。切换 EP 度与 top-k，观察通信比例与专家负载如何变化。',
    render: render
  };
})();
