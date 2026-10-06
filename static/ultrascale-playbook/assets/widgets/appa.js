/* 附录 A0：集体通信操作动画台（Collective Operations Playground） */
(function () {
  'use strict';
  const N = 5;
  const NS = 'http://www.w3.org/2000/svg';
  const IDX = Array.from({ length: N }, (_, i) => i);
  const rnd = () => 1 + Math.floor(Math.random() * 9);

  const OPS = [
    { id: 'broadcast', name: 'Broadcast（广播）', root: true,
      desc: '将 root（rank 0）上的整份数据原样复制到所有其他节点，数据不切分、不合并。',
      use: '训练启动时，把 rank 0 初始化的模型权重广播给所有 GPU，确保大家从同一组权重出发。' },
    { id: 'reduce', name: 'Reduce（归约）', root: true,
      desc: '用函数 f()（此处为加总）把每个节点上的数据合并起来，结果只送到 root；其他节点的数据维持不变。',
      use: '把各节点的部分结果（如 loss、评估统计）加总到单一节点，回报整体结果。' },
    { id: 'allreduce', name: 'All-Reduce（全归约）',
      desc: '把所有节点的数据合并（加总），且每个节点都得到合并结果。概念上等于 Reduce 后再 Broadcast；实践中多以 Ring All-Reduce 实现。',
      use: '数据并行（DP）的梯度同步：反向传播后把所有 GPU 的梯度加总，每张卡都拿到同一份梯度再更新权重。' },
    { id: 'gather', name: 'Gather（收集）', root: true,
      desc: '每个节点各有一块独立数据，把所有数据块收集到 root 上依 rank 排好，只搬移、不合并。',
      use: '把散在各 GPU 的结果（如评估输出）收集到 rank 0 汇整或存档。' },
    { id: 'allgather', name: 'All-Gather（全收集）',
      desc: '与 Gather 相同，但收集到『所有』节点：结束后每个节点都拥有完整的数据清单。',
      use: 'ZeRO-3／FSDP 在前向与反向之前，把分片存放的参数重组成完整参数；也是 Ring All-Reduce 的第二阶段。' },
    { id: 'scatter', name: 'Scatter（分散）', root: true,
      desc: '把 root 上的数据切成分片，分发给各节点各一块；与 Broadcast 不同（不复制整份），是 Gather 的逻辑反向。',
      use: '把 rank 0 上的数据或初始分片切分后分发给各 GPU。' },
    { id: 'reducescatter', name: 'Reduce-Scatter（归约分散）',
      desc: '先像 Reduce 一样对『对应分片』做加总，再把不同分片的结果分给不同节点：GPU j 得到所有节点第 j 块的总和。',
      use: 'ZeRO 的梯度同步：每张 GPU 只保留自己负责分片的梯度总和，通信成本约为 all-reduce 的一半；也是 Ring All-Reduce 的第一阶段。' },
    { id: 'ring', name: 'Ring All-Reduce（环状全归约）',
      desc: 'All-Reduce 的可扩展实现：每个节点把数据切成 N 块，先沿环做 N−1 步 Reduce-Scatter（传递并累加分片），再做 N−1 步 All-Gather（转发归约完成的分片）。',
      use: 'NCCL 用它实现 DP 的梯度 all-reduce：即使节点间带宽有限也能被有效利用，是大规模训练的通信基石。' }
  ];

  function texify(el, src) {
    if (window.katex) window.katex.render(src, el, { throwOnError: false });
    else el.textContent = src;
  }
  function mk(tag, attrs, css, parent, text) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
    for (const k in css || {}) e.style[k] = css[k];
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }

  window.ChapterWidget = {
    title: '集体通信操作动画台',
    intro: '选择一种集体通信操作并按播放，观察数据如何在 5 张 GPU 之间复制、合并与分发——这些正是多 GPU 训练背后的通信基本功。',
    render(rootEl) {
      let vals = IDX.map(rnd);
      let chunks = IDX.map(() => IDX.map(rnd));
      let opId = 'broadcast', prog = null, model = null, token = 0;

      /* ---------- 依目前数据为指定操作建立「状态快照 + 动画步骤」 ---------- */
      function buildProgram() {
        const op = OPS.find(o => o.id === opId);
        const v = vals.slice();
        const sum = v.reduce((a, b) => a + b, 0);
        const st = [], steps = [];
        const cell = (val, hot, dim) => ({ v: val, hot: !!hot, dim: !!dim });
        const snap = m => m.map(r => r.map(c => (c ? cell(c.v, c.hot, c.dim) : null)));
        if (op.id === 'broadcast') {
          st.push(IDX.map(i => [cell(i === 0 ? v[0] : 0)]));
          st.push(IDX.map(i => [cell(v[0], i > 0)]));
          steps.push({ label: 'root 将整份数据复制给所有节点',
            flights: IDX.slice(1).map(i => ({ f: [0, 0], t: [i, 0], v: v[0] })) });
        } else if (op.id === 'reduce') {
          st.push(IDX.map(i => [cell(v[i])]));
          st.push(IDX.map(i => [i === 0 ? cell(sum, true) : cell(v[i])]));
          steps.push({ label: '各节点数据送往 root 加总（其余节点不变）',
            flights: IDX.slice(1).map(i => ({ f: [i, 0], t: [0, 0], v: v[i] })) });
        } else if (op.id === 'allreduce') {
          st.push(IDX.map(i => [cell(v[i])]));
          st.push(IDX.map(i => [i === 0 ? cell(sum, true) : cell(v[i])]));
          st.push(IDX.map(() => [cell(sum, true)]));
          steps.push({ label: '先归约：加总到节点 0',
            flights: IDX.slice(1).map(i => ({ f: [i, 0], t: [0, 0], v: v[i] })) });
          steps.push({ label: '再广播：结果送回所有节点',
            flights: IDX.slice(1).map(i => ({ f: [0, 0], t: [i, 0], v: sum })) });
        } else if (op.id === 'gather') {
          st.push(IDX.map(i => (i === 0 ? IDX.map(j => (j === 0 ? cell(v[0]) : null)) : [cell(v[i])])));
          st.push(IDX.map(i => (i === 0 ? IDX.map(j => cell(v[j], j > 0)) : [cell(v[i])])));
          steps.push({ label: '各节点的数据块搬到 root、依 rank 排列（不合并）',
            flights: IDX.slice(1).map(i => ({ f: [i, 0], t: [0, i], v: v[i] })) });
        } else if (op.id === 'allgather') {
          st.push(IDX.map(i => IDX.map(j => (j === i ? cell(v[i]) : null))));
          st.push(IDX.map(i => IDX.map(j => cell(v[j], j !== i))));
          const fl = [];
          IDX.forEach(i => IDX.forEach(j => { if (i !== j) fl.push({ f: [i, i], t: [j, i], v: v[i] }); }));
          steps.push({ label: '每个节点把自己的数据块分享给所有其他节点', flights: fl });
        } else if (op.id === 'scatter') {
          st.push(IDX.map(i => (i === 0 ? IDX.map(j => cell(v[j])) : [null])));
          st.push(IDX.map(i => [cell(v[i], i > 0)]));
          steps.push({ label: 'root 把分片一人一块分发出去',
            flights: IDX.slice(1).map(i => ({ f: [0, i], t: [i, 0], v: v[i] })) });
        } else if (op.id === 'reducescatter') {
          const m = IDX.map(i => IDX.map(j => cell(chunks[i][j])));
          st.push(snap(m));
          IDX.forEach(j => {
            m[j][j] = cell(IDX.reduce((a, i) => a + chunks[i][j], 0), true);
            st.push(snap(m));
            steps.push({ label: '所有节点的第 ' + j + ' 块加总到 GPU ' + j,
              flights: IDX.filter(i => i !== j).map(i => ({ f: [i, j], t: [j, j], v: chunks[i][j] })) });
          });
          st[st.length - 1] = st[st.length - 1].map((r, i) =>
            r.map((c, j) => (j === i ? c : cell(c.v, false, true))));
        } else { /* ring all-reduce */
          const M = chunks.map(r => r.slice());
          const hot = IDX.map(() => IDX.map(() => false));
          const snapM = () => M.map((r, i) => r.map((x, j) => cell(x, hot[i][j])));
          st.push(snapM());
          for (let t = 0; t < N - 1; t++) { // 阶段一：reduce-scatter，沿环传递并「累加」
            const fl = IDX.map(i => { const s = (i - t + N) % N; return { f: [i, s], t: [(i + 1) % N, s], v: M[i][s] }; });
            fl.forEach(f => { M[f.t[0]][f.t[1]] += f.v; });
            if (t === N - 2) IDX.forEach(i => { hot[i][(i + 1) % N] = true; });
            st.push(snapM());
            steps.push({ label: '阶段一 Reduce-Scatter：累加分片（' + (t + 1) + '/' + (N - 1) + '）', flights: fl });
          }
          for (let t = 0; t < N - 1; t++) { // 阶段二：all-gather，沿环「转发」完成的分片
            const fl = IDX.map(i => { const s = (i + 1 - t + N) % N; return { f: [i, s], t: [(i + 1) % N, s], v: M[i][s] }; });
            fl.forEach(f => { M[f.t[0]][f.t[1]] = f.v; hot[f.t[0]][f.t[1]] = true; });
            st.push(snapM());
            steps.push({ label: '阶段二 All-Gather：转发归约完成的分片（' + (t + 1) + '/' + (N - 1) + '）', flights: fl });
          }
        }
        const layout = op.id === 'ring' ? 'ring' : 'row';
        const maxSlots = Math.max.apply(null, st.map(m2 => Math.max.apply(null, m2.map(r => r.length))));
        return { op, layout, states: st, steps, vb: layout === 'ring' ? 360 : 52 + maxSlots * 30 };
      }

      /* ---------- 几何与绘制 ---------- */
      function ringP(i) {
        const a = -Math.PI / 2 + (i * 2 * Math.PI) / N;
        return { x: 320 + 150 * Math.cos(a), y: 205 + 150 * Math.sin(a) };
      }
      function cellPos(i, slot) {
        if (prog.layout === 'ring') { const p = ringP(i); return { x: p.x - 56 + slot * 28, y: p.y + 8 }; }
        return { x: 64 + i * 128, y: 58 + slot * 30 };
      }
      function drawCell(g, x, y, w, hgt, c, fs) {
        const r = mk('rect', { x: x - w / 2, y: y - hgt / 2, width: w, height: hgt, rx: 5 }, {}, g);
        if (!c || c.v == null) {
          r.style.fill = 'none'; r.style.stroke = 'var(--border)'; r.style.strokeDasharray = '4 3';
          return;
        }
        r.style.fill = c.hot ? 'var(--accent-soft)' : 'var(--panel)';
        r.style.stroke = c.hot ? 'var(--accent)' : 'var(--border)';
        if (c.dim) r.style.opacity = 0.3;
        mk('text', { x, y: y + fs * 0.36, 'text-anchor': 'middle' },
          { fill: 'var(--fg)', fontSize: fs + 'px', fontWeight: c.hot ? '600' : '400', opacity: c.dim ? 0.4 : 1 },
          g, c.v);
      }
      function renderScene() {
        svg.innerHTML = '';
        svg.setAttribute('viewBox', '0 0 640 ' + prog.vb);
        const defs = mk('defs', {}, {}, svg);
        [['appa-aw1', 'var(--accent-2)'], ['appa-aw2', 'var(--fg-muted)']].forEach(pair => {
          const m = mk('marker', { id: pair[0], viewBox: '0 0 10 10', refX: 8, refY: 5,
            markerWidth: 6.5, markerHeight: 6.5, orient: 'auto-start-reverse' }, {}, defs);
          mk('path', { d: 'M0,0 L10,5 L0,10 z' }, { fill: pair[1] }, m);
        });
        if (prog.layout === 'ring') IDX.forEach(i => { // 环的方向箭头（i → i+1）
          const A = ringP(i), B = ringP((i + 1) % N);
          const s = { x: A.x + (B.x - A.x) * 0.3, y: A.y + (B.y - A.y) * 0.3 };
          const e = { x: A.x + (B.x - A.x) * 0.7, y: A.y + (B.y - A.y) * 0.7 };
          const mx = (A.x + B.x) / 2 - 320, my = (A.y + B.y) / 2 - 205, L = Math.hypot(mx, my) || 1;
          mk('path', { d: 'M' + s.x + ',' + s.y + ' Q' + (320 + mx + (mx / L) * 30) + ',' +
              (205 + my + (my / L) * 30) + ' ' + e.x + ',' + e.y, 'marker-end': 'url(#appa-aw2)' },
            { fill: 'none', stroke: 'var(--fg-muted)', opacity: 0.5 }, svg);
        });
        model.forEach((cells, i) => {
          const g = mk('g', {}, {}, svg);
          if (prog.layout === 'ring') {
            const p = ringP(i);
            mk('text', { x: p.x, y: p.y - 10, 'text-anchor': 'middle' },
              { fill: 'var(--fg)', fontSize: '12px', fontWeight: '600' }, g, 'GPU ' + i);
            cells.forEach((c, j) => drawCell(g, p.x - 56 + j * 28, p.y + 8, 26, 20, c, 10));
          } else {
            const x = 64 + i * 128;
            mk('text', { x, y: 20, 'text-anchor': 'middle' },
              { fill: 'var(--fg)', fontSize: '13px', fontWeight: '600' }, g, 'GPU ' + i);
            if (prog.op.root && i === 0) mk('text', { x, y: 36, 'text-anchor': 'middle' },
              { fill: 'var(--accent)', fontSize: '11px', fontWeight: '600' }, g, 'root');
            cells.forEach((c, j) => drawCell(g, x, 58 + j * 30, 58, 26, c, 13));
          }
        });
      }

      /* ---------- 动画 ---------- */
      function animateFlights(flights, myTok, done) {
        const layer = mk('g', {}, {}, svg);
        const small = prog.layout === 'ring';
        const w = small ? 26 : 36, hgt = small ? 18 : 22, fs = small ? 10 : 12;
        const items = flights.map(f => {
          const a = cellPos(f.f[0], f.f[1]), b = cellPos(f.t[0], f.t[1]);
          mk('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y, 'marker-end': 'url(#appa-aw1)' },
            { stroke: 'var(--accent-2)', strokeDasharray: '5 4', opacity: 0.5 }, layer);
          const g = mk('g', { transform: 'translate(' + a.x + ',' + a.y + ')' }, {}, layer);
          mk('rect', { x: -w / 2, y: -hgt / 2, width: w, height: hgt, rx: 5 },
            { fill: 'var(--accent-soft)', stroke: 'var(--accent)' }, g);
          mk('text', { x: 0, y: fs * 0.36, 'text-anchor': 'middle' },
            { fill: 'var(--fg)', fontSize: fs + 'px', fontWeight: '600' }, g, f.v);
          return { a, b, g };
        });
        const t0 = performance.now(), dur = 750;
        (function frame(now) {
          if (myTok !== token) { layer.remove(); return; }
          const p = Math.min(1, (now - t0) / dur);
          const e = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
          items.forEach(it => it.g.setAttribute('transform',
            'translate(' + (it.a.x + (it.b.x - it.a.x) * e) + ',' + (it.a.y + (it.b.y - it.a.y) * e) + ')'));
          if (p < 1) requestAnimationFrame(frame);
          else { layer.remove(); done(); }
        })(t0);
      }
      function play() {
        token++; const myTok = token;
        playBtn.disabled = true;
        model = prog.states[0]; renderScene();
        let k = 0;
        (function step() {
          if (myTok !== token) return;
          if (k >= prog.steps.length) {
            status.textContent = '完成 ✔ 上图为各节点的最终状态（框线高亮＝本次操作写入的结果）。';
            playBtn.disabled = false; return;
          }
          const s = prog.steps[k];
          status.textContent = '步骤 ' + (k + 1) + ' / ' + prog.steps.length + '　·　' + s.label;
          animateFlights(s.flights, myTok, () => {
            if (myTok !== token) return;
            model = prog.states[++k]; renderScene();
            setTimeout(step, 300);
          });
        })();
      }
      function setOp(id) {
        opId = id; token++; playBtn.disabled = false;
        prog = buildProgram();
        model = prog.states[0]; renderScene();
        status.textContent = '按「播放 ▶」开始（共 ' + prog.steps.length + ' 步）。';
        descEl.textContent = prog.op.desc;
        useEl.textContent = '在 LLM 训练中：' + prog.op.use;
        formulaEl.style.display = opId === 'ring' ? '' : 'none';
      }

      /* ---------- 版面 ---------- */
      const panel = document.createElement('div');
      panel.className = 'widget-panel';
      const row = document.createElement('div');
      row.className = 'widget-row';
      row.style.marginBottom = '.75rem';
      const selLabel = document.createElement('label');
      selLabel.textContent = '操作：';
      const sel = document.createElement('select');
      OPS.forEach(o => { const opt = document.createElement('option'); opt.value = o.id; opt.textContent = o.name; sel.appendChild(opt); });
      sel.addEventListener('change', () => setOp(sel.value));
      selLabel.appendChild(sel);
      const button = (txt, fn) => {
        const b = document.createElement('button');
        b.type = 'button'; b.textContent = txt; b.addEventListener('click', fn);
        return b;
      };
      const playBtn = button('播放 ▶', play);
      const replayBtn = button('重播', play);
      const randBtn = button('换一组随机数据', () => {
        vals = IDX.map(rnd); chunks = IDX.map(() => IDX.map(rnd)); setOp(opId);
      });
      row.append(selLabel, playBtn, replayBtn, randBtn);
      const status = document.createElement('div');
      status.style.cssText = 'font-size:.88em;color:var(--fg-muted);margin:.25rem 0 .5rem;min-height:1.4em;';
      const svg = mk('svg', { viewBox: '0 0 640 200', role: 'img', 'aria-label': '集体通信操作动画' },
        { width: '100%', height: 'auto', display: 'block' });
      const descEl = document.createElement('p');
      descEl.style.cssText = 'margin:.75rem 0 .25rem;font-size:.95em;color:var(--fg);';
      const useEl = document.createElement('p');
      useEl.style.cssText = 'margin:.25rem 0 0;font-size:.9em;color:var(--fg-muted);border-left:3px solid var(--accent-2);padding-left:.6rem;';
      const formulaEl = document.createElement('div');
      formulaEl.style.cssText = 'margin-top:.75rem;padding:.6rem .8rem;border:1px solid var(--border);' +
        'border-radius:8px;background:var(--code-bg);font-size:.92em;line-height:2;overflow-x:auto;';
      const f1 = document.createElement('div');
      f1.appendChild(document.createTextNode('通信量对比（每张 GPU；K 为参数总数）：Ring All-Reduce 为 '));
      const k1 = document.createElement('span');
      texify(k1, '2(N{-}1)\\tfrac{K}{N} \\approx 2K');
      f1.appendChild(k1);
      f1.appendChild(document.createTextNode('，与 N 无关；'));
      const f2 = document.createElement('div');
      f2.appendChild(document.createTextNode('而 naive all-reduce（每张 GPU 直接把完整数据送给所有其他 GPU）为 '));
      const k2 = document.createElement('span');
      texify(k2, '2(N{-}1)K');
      f2.appendChild(k2);
      f2.appendChild(document.createTextNode('，随 N 线性成长。以本例 N=5：环状每卡 1.6K vs. naive 8K。'));
      formulaEl.append(f1, f2);
      panel.append(row, status, svg, descEl, useEl, formulaEl);
      rootEl.appendChild(panel);
      setOp(opId);
    }
  };
})();
