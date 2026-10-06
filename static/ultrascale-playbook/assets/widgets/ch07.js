/* 第 7 章交互元件：5D 并行配置探索器 */
(function () {
  'use strict';

  // ---------- DOM / SVG 小工具 ----------
  function el(tag, attrs, children) {
    const n = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === 'text') n.textContent = attrs[k];
      else if (k === 'style') n.style.cssText = attrs[k];
      else n.setAttribute(k, attrs[k]);
    }
    (children || []).forEach((c) => n.appendChild(c));
    return n;
  }
  const SVG_NS = 'http://www.w3.org/2000/svg';
  function svgEl(tag, attrs, parent) {
    const n = document.createElementNS(SVG_NS, tag);
    if (attrs) for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }

  // ---------- 五个维度的关键属性（取自本章总表） ----------
  const DIMS = [
    { key: 'dp', name: 'DP 数据并行', what: '批次维度（降低本地批次）', comm: '梯度 all-reduce（可与 ZeRO 结合）', cap: '受限于最大全局批次大小' },
    { key: 'tp', name: 'TP 张量并行（+SP）', what: '隐藏维度／序列（权重＋激活值）', comm: 'matmul 前后 all-reduce / all-gather', cap: '需高带宽，典型 TP ≤ 8 留在节点内' },
    { key: 'pp', name: 'PP 流水线并行', what: '模型的层', comm: '阶段之间点对点传激活值', cap: '有闲置气泡，需大 grad_acc 摊平' },
    { key: 'cp', name: 'CP 上下文并行', what: '序列长度（激活值）', comm: '注意力 K/V（ring attention）', cap: '适用 128k+ 超长序列' },
    { key: 'ep', name: 'EP 专家并行', what: '专家维度（MoE 层）', comm: 'token 路由 all-to-all', cap: '需要 MoE 架构' },
  ];
  const DEFAULTS = { dp: 2, tp: 2, pp: 2, cp: 1, ep: 1, gbs: 64 };
  const LLAMA = { dp: 8, tp: 8, pp: 4, cp: 1, ep: 1, gbs: 1024 };
  const TP_COLORS = ['var(--accent)', 'var(--link)']; // --accent-2 保留给警示

  // ---------- 合理性检查规则 ----------
  function evalRules(s) {
    const out = [];
    const total = s.dp * s.tp * s.pp * s.cp * s.ep;
    const m = Math.floor(s.gbs / s.dp); // 每副本 microbatch 数（假设 mbs=1）
    if (s.tp > 8) out.push({ lv: 'bad', text: 'TP=' + s.tp + '：超过单节点 8 GPU。TP 通信位于计算关键路径上，跨节点带宽不足会严重拖慢训练' });
    else if (s.tp === 8) out.push({ lv: 'note', text: 'TP=8：已达典型节点上限（8 GPU／节点）。TP 群组应留在节点内，走 NVLink 高带宽——这正是建议用法' });
    else out.push({ lv: 'ok', text: 'TP=' + s.tp + ' ≤ 8：TP 群组可留在节点内（高带宽通信）' });
    if (s.dp > s.gbs) out.push({ lv: 'bad', text: 'DP=' + s.dp + ' > gbs=' + s.gbs + '：每个 DP 副本连 1 个样本都分不到。DP 度数不可超过全局批次大小' });
    else out.push({ lv: 'ok', text: 'DP × gbs：每个 DP 副本分到 ' + (s.gbs / s.dp) + ' 个样本（gbs=' + s.gbs + '）' });
    if (s.pp > 1 && m >= 1) {
      const bubble = (s.pp - 1) / (m + s.pp - 1);
      const pct = Math.round(bubble * 100) + '%';
      if (bubble > 0.2) out.push({ lv: 'bad', text: 'PP=' + s.pp + ' 但每副本仅 ' + m + ' 个 microbatch：气泡占比 ≈ ' + pct + '。PP 偏好较大的 grad_acc（提高 gbs 或降低 DP）来隐藏气泡' });
      else out.push({ lv: 'ok', text: 'PP 气泡占比 ≈ ' + pct + '（m=' + m + '，(p−1)/(m+p−1)），已被足够的 microbatch 摊平' });
    } else if (s.pp === 1) out.push({ lv: 'ok', text: 'PP=1：无流水线气泡' });
    if (total > 512) out.push({ lv: 'bad', text: '总 GPU=' + total + '：超过 512，超出本章示例集群规模（图仅示意 DP×PP×TP 前三维）' });
    if (s.cp > 1) out.push({ lv: 'note', text: 'CP=' + s.cp + '：适用超长序列（128k+）。注意力层以 ring attention 交换 K/V，其余层独立处理切分后的序列' });
    if (s.ep > 1) out.push({ lv: 'note', text: 'EP=' + s.ep + '：仅适用 MoE 模型（如 DeepSeek-V3 的 256 个专家），以 all-to-all 将 token 路由至专家' });
    return out;
  }

  // ---------- SVG：GPU 格点图（DP 副本 → PP 阶段 → TP 组） ----------
  function drawGrid(svg, s) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const repCols = s.dp <= 2 ? s.dp : (s.dp === 4 ? 2 : 4);
    const repRows = Math.ceil(s.dp / repCols);
    const padIn = 8, labelH = 18, gap = 14, rowH = 8; // rowH：阶段列的额外留白
    let cs = Math.floor((640 - repCols * padIn * 2 - (repCols - 1) * gap) / (repCols * s.tp));
    cs = Math.max(10, Math.min(30, cs));
    const leftM = s.pp > 1 ? 26 : 4;
    const blockW = s.tp * cs + padIn * 2;
    const blockH = labelH + s.pp * (cs + rowH) + padIn;
    const badges = [];
    if (s.cp > 1) badges.push('CP=' + s.cp + '：每格再沿序列切 ' + s.cp + ' 份');
    if (s.ep > 1) badges.push('EP=' + s.ep + '：专家分散于 ' + s.ep + ' 组 GPU');
    const badgeH = badges.length ? 30 : 0;
    const totW = Math.max(repCols * blockW + (repCols - 1) * gap + leftM, badges.length ? 300 : 120);
    const totH = repRows * blockH + (repRows - 1) * gap + badgeH + 4;
    svg.setAttribute('viewBox', '0 0 ' + totW + ' ' + totH);
    svg.style.width = '100%';
    svg.style.maxWidth = Math.max(340, Math.round(totW * 1.05)) + 'px';

    for (let r = 0; r < s.dp; r++) {
      const bx = leftM + (r % repCols) * (blockW + gap);
      const by = Math.floor(r / repCols) * (blockH + gap) + 2;
      // DP 副本：实线区块（副本之间以间隔分隔）
      svgEl('rect', { x: bx, y: by, width: blockW, height: blockH, rx: 6, fill: 'var(--panel-2)', stroke: 'var(--border)' }, svg);
      const lab = svgEl('text', { x: bx + padIn, y: by + 13, 'font-size': 10, fill: 'var(--fg-muted)' }, svg);
      lab.textContent = 'DP 副本 ' + (r + 1);
      for (let p = 0; p < s.pp; p++) {
        const sy = by + labelH + p * (cs + rowH);
        if (s.pp > 1) { // PP 阶段：虚线边框
          svgEl('rect', { x: bx + padIn - 3, y: sy - 3, width: s.tp * cs + 6, height: cs + 6, rx: 3, fill: 'none', stroke: 'var(--fg-muted)', 'stroke-dasharray': '4 3' }, svg);
          if (r % repCols === 0) {
            const st = svgEl('text', { x: bx - 6, y: sy + cs / 2 + 4, 'font-size': 10, fill: 'var(--fg-muted)', 'text-anchor': 'end' }, svg);
            st.textContent = 'S' + (p + 1);
          }
        }
        for (let t = 0; t < s.tp; t++) { // TP 组：同一列同色
          const c = svgEl('rect', { x: bx + padIn + t * cs + 1, y: sy + 1, width: cs - 2, height: cs - 2, rx: 3, fill: TP_COLORS[p % TP_COLORS.length], 'fill-opacity': 0.75, stroke: 'var(--bg)' }, svg);
          const title = svgEl('title', null, c);
          title.textContent = 'GPU #' + ((r * s.pp + p) * s.tp + t) + '｜DP 副本 ' + (r + 1) + '・PP 阶段 ' + (p + 1) + '・TP rank ' + (t + 1);
        }
      }
    }
    badges.forEach((txt, i) => { // CP / EP：文字徽章
      const bw = txt.length * 11 + 18, bx = 4 + i * (bw + 10), by = totH - 26;
      svgEl('rect', { x: bx, y: by, width: bw, height: 22, rx: 11, fill: 'var(--accent-soft)', stroke: 'var(--accent)' }, svg);
      const t = svgEl('text', { x: bx + bw / 2, y: by + 15, 'font-size': 11, fill: 'var(--fg)', 'text-anchor': 'middle' }, svg);
      t.textContent = txt;
    });
  }

  // ---------- 元件本体 ----------
  window.ChapterWidget = {
    title: '5D 并行配置探索器',
    intro: '调整 DP、TP、PP、CP、EP 五个维度与全局批次大小 gbs，观察 GPU 如何被分组：同色一列＝同一 TP 组（节点内高带宽）、虚线框＝PP 阶段、实线区块＝DP 副本；CP/EP 是每颗 GPU 内激活值／专家的进一步切分，以徽章表示。下方「合理性检查」对照本章总表提示常见的组合陷阱。',
    render(root) {
      const state = Object.assign({}, DEFAULTS);

      // 控制面板：五个维度卡片 + gbs
      const panel = el('div', { class: 'widget-panel' });
      const grid = el('div', { style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;' });
      const selects = {};
      DIMS.forEach((d) => {
        const sel = el('select');
        [1, 2, 4, 8].forEach((v) => sel.appendChild(el('option', { value: String(v), text: String(v) })));
        sel.addEventListener('change', () => { state[d.key] = +sel.value; update(); });
        selects[d.key] = sel;
        grid.appendChild(el('div', { style: 'border:1px solid var(--border);border-radius:8px;padding:8px;background:var(--panel-2);' }, [
          el('div', { style: 'display:flex;justify-content:space-between;align-items:center;gap:6px;margin-bottom:6px;' }, [
            el('strong', { text: d.name, style: 'font-size:.82rem;' }), sel,
          ]),
          el('div', { text: '切什么：' + d.what, style: 'font-size:.72rem;color:var(--fg-muted);' }),
          el('div', { text: '通信：' + d.comm, style: 'font-size:.72rem;color:var(--fg-muted);' }),
          el('div', { text: '上限：' + d.cap, style: 'font-size:.72rem;color:var(--fg-muted);' }),
        ]));
      });
      panel.appendChild(grid);

      const gbsSel = el('select');
      [4, 16, 64, 256, 1024].forEach((v) => gbsSel.appendChild(el('option', { value: String(v), text: String(v) })));
      gbsSel.addEventListener('change', () => { state.gbs = +gbsSel.value; update(); });
      const totalBox = el('div', { style: 'font-size:.95rem;margin-left:auto;' });
      const row = el('div', { class: 'widget-row', style: 'margin-top:10px;flex-wrap:wrap;align-items:center;gap:10px;' }, [
        el('label', { text: '全局批次大小 gbs（样本数，假设 mbs=1）：', style: 'font-size:.82rem;color:var(--fg-muted);' }), gbsSel, totalBox,
      ]);
      panel.appendChild(row);

      const llamaBtn = el('button', { text: 'Llama-3 70B 风格（TP8 · PP4 · DP8）' });
      const resetBtn = el('button', { text: '重设' });
      llamaBtn.addEventListener('click', () => { Object.assign(state, LLAMA); update(); });
      resetBtn.addEventListener('click', () => { Object.assign(state, DEFAULTS); update(); });
      panel.appendChild(el('div', { class: 'widget-row', style: 'margin-top:8px;gap:8px;' }, [llamaBtn, resetBtn]));
      root.appendChild(panel);

      // 可视化面板
      const vizPanel = el('div', { class: 'widget-panel' });
      const svg = svgEl('svg', { role: 'img', 'aria-label': 'GPU 并行分组格点图' });
      const caption = el('div', { style: 'font-size:.75rem;color:var(--fg-muted);margin-top:6px;' });
      vizPanel.appendChild(svg);
      vizPanel.appendChild(caption);
      vizPanel.appendChild(el('div', {
        text: '图例：同色一列＝同一 TP 组（应留在节点内）｜虚线框＝PP 阶段（S1、S2⋯）｜实线区块＝DP 副本｜CP／EP＝每格内的进一步切分（徽章）',
        style: 'font-size:.72rem;color:var(--fg-muted);margin-top:4px;',
      }));
      root.appendChild(vizPanel);

      // 合理性检查面板
      const rulePanel = el('div', { class: 'widget-panel' });
      rulePanel.appendChild(el('strong', { text: '合理性检查', style: 'font-size:.85rem;' }));
      const ruleList = el('div', { style: 'margin-top:8px;display:flex;flex-direction:column;gap:6px;' });
      rulePanel.appendChild(ruleList);
      root.appendChild(rulePanel);

      function update() {
        DIMS.forEach((d) => { selects[d.key].value = String(state[d.key]); });
        gbsSel.value = String(state.gbs);
        const total = state.dp * state.tp * state.pp * state.cp * state.ep;
        totalBox.textContent = '';
        totalBox.appendChild(el('span', { text: '总 GPU = ', style: 'color:var(--fg-muted);font-size:.82rem;' }));
        totalBox.appendChild(el('strong', {
          text: state.dp + '×' + state.tp + '×' + state.pp + '×' + state.cp + '×' + state.ep + ' = ' + total,
          style: 'color:' + (total > 512 ? 'var(--accent-2)' : 'var(--accent)') + ';',
        }));
        if (total > 512) totalBox.appendChild(el('span', { text: '（超过 512！）', style: 'color:var(--accent-2);font-size:.8rem;' }));

        drawGrid(svg, state);
        const n3 = state.dp * state.tp * state.pp;
        caption.textContent = '格点显示 DP×PP×TP = ' + n3 + ' 颗 GPU；每颗再依 CP×EP = ' + (state.cp * state.ep) + ' 细分，总计 ' + total + ' 颗。';

        ruleList.textContent = '';
        evalRules(state).forEach((rule) => {
          const color = rule.lv === 'bad' ? 'var(--accent-2)' : rule.lv === 'note' ? 'var(--accent)' : 'var(--fg-muted)';
          const chip = el('span', {
            text: rule.lv === 'bad' ? '警示' : rule.lv === 'note' ? '提示' : '通过',
            style: 'flex:0 0 auto;font-size:.7rem;border:1px solid ' + color + ';color:' + color + ';border-radius:999px;padding:1px 8px;',
          });
          ruleList.appendChild(el('div', { style: 'display:flex;gap:8px;align-items:baseline;' }, [
            chip,
            el('span', { text: rule.text, style: 'font-size:.8rem;color:' + (rule.lv === 'bad' ? 'var(--accent-2)' : 'var(--fg)') + ';' }),
          ]));
        });
      }
      update();
    },
  };
})();
