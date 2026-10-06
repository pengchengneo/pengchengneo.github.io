/* 第 10 章交互元件：浮点格式探索器（FP32 / BF16 / FP16 / FP8） */
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const P2 = (n) => Math.pow(2, n);
  // 各格式的位元帐：max 为精确的最大有限值；E4M3 无 inf（全 1 样式保留给 NaN）
  const FORMATS = [
    { id: 'fp32', name: 'FP32', e: 8, m: 23, bias: 127, max: (2 - P2(-23)) * P2(127) },
    { id: 'bf16', name: 'BF16', e: 8, m: 7, bias: 127, max: (2 - P2(-7)) * P2(127) },
    { id: 'fp16', name: 'FP16', e: 5, m: 10, bias: 15, max: 65504 },
    { id: 'e4m3', name: 'FP8-E4M3', e: 4, m: 3, bias: 7, max: 448, noInf: true },
    { id: 'e5m2', name: 'FP8-E5M2', e: 5, m: 2, bias: 15, max: 57344 }
  ];
  FORMATS.forEach((f) => {
    f.bits = 1 + f.e + f.m;
    f.minNormal = P2(1 - f.bias);            // 最小正规值 2^(1-bias)
    f.minSub = P2(1 - f.bias - f.m);         // 最小次正规值
    f.eps = P2(-f.m);                        // 1.0 之后第一个可表示数的间距
    f.range = Math.round(Math.log10(f.max / f.minSub));
  });
  const INTERP = {
    fp32: 'FP32 是训练的安全基准：约 83 个数量级的范围加上 1.19×10⁻⁷ 的 epsilon，范围与精度都绰绰有余——代价是每个数占 4 位元组。混合精度训练中，主权重与优化器状态通常仍保留在 FP32。',
    bf16: 'BF16 的指数位与 FP32 一样是 8 位，范围几乎相同——梯度再小也不易下溢，因此不需要损失缩放（loss scaling）。代价是尾数只剩 7 位、epsilon ≈ 7.8×10⁻³：精度粗得多，所以要靠 FP32 主权重与 FP32 累加来补救。',
    fp16: 'FP16 的指数只有 5 位，最小正规值约 6.1×10⁻⁵——而梯度值常小于这个下溢边界，会被直接舍成 0。这就是需要损失缩放（loss scaling）的原因：反向传播前先放大损失、之后再还原缩放；BF16 因为范围够大所以不用。',
    e4m3: 'FP8-E4M3 把较多位元给尾数（精度相对好），但最大值只有 448、1 到 2 之间仅 7 个可表示数。FP8 训练最大的挑战是稳定性：数值不稳定常让损失发散，DeepSeek-V3 得靠逐 tile 缩放归一化等技巧才稳住大规模训练。',
    e5m2: 'FP8-E5M2 把较多位元给指数，保住与 FP16 相近的范围，但 1 到 2 之间只剩 3 个可表示数。FP8 训练最大的挑战是稳定性：低精度下数值不稳定常让损失发散，难以追平高精度训练的准确度。'
  };
  const TAIL_16 = 'FP8 的 GEMM 在 H100 上理论 FLOPS 是 BF16 的两倍，诱因十足——但范围与精度双双受限，稳定性是 FP8 预训练最大的挑战。';
  const TAIL_8 = '对照 16 位元：FP16 需要损失缩放来对抗梯度下溢，BF16 范围够大所以不用。';

  const state = { fmt: 'bf16', x: 0.0001 };
  let bitsSvg, formulaEl, cardsEl, tableWrap, noteEl, lineSvg, interpEl, inputEl;
  const btns = {};

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
  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }
  const SUP = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  const sup = (s) => String(s).split('').map((c) => SUP[c] || c).join('');
  function fmtNum(v) {
    if (!isFinite(v)) return v > 0 ? 'inf' : '-inf';
    if (v === 0) return '0';
    const a = Math.abs(v);
    if (a >= 1e5 || a < 1e-3) {
      const [mant, ex] = v.toExponential(2).split('e');
      return mant + ' × 10' + sup(String(+ex));
    }
    return String(parseFloat(v.toPrecision(5)));
  }

  // 把一个实数舍入到指定格式（就近舍入），回传 {v, kind: ok|sub|under|over, err}
  function quantize(x, f) {
    if (x === 0 || !isFinite(x)) return { v: x, kind: 'ok', err: 0 };
    const sgn = Math.sign(x), a = Math.abs(x);
    let e = Math.floor(Math.log2(a));
    const emin = 1 - f.bias;
    if (e < emin) e = emin; // 次正规区用固定间距 2^(emin-m)
    const q = P2(e - f.m);
    const r = Math.round(a / q) * q;
    if (r > f.max) return { v: sgn * Infinity, kind: 'over', err: Infinity };
    if (r === 0) return { v: 0, kind: 'under', err: 1 };
    return { v: sgn * r, kind: r < f.minNormal ? 'sub' : 'ok', err: Math.abs(r - a) / a };
  }
  const cur = () => FORMATS.find((f) => f.id === state.fmt);

  function drawBits() {
    clear(bitsSvg);
    const f = cur(), W = 680, y = 36, h = 30;
    const cw = Math.min((W - 24) / f.bits, 46);
    const x0 = (W - cw * f.bits) / 2;
    const segs = [
      { n: 1, label: '符号', color: 'var(--accent-2)' },
      { n: f.e, label: '指数', color: 'var(--accent)' },
      { n: f.m, label: '尾数', color: 'var(--link)' }
    ];
    let bit = 0;
    segs.forEach((seg) => {
      for (let i = 0; i < seg.n; i++) {
        bitsSvg.appendChild(S('rect', {
          x: (x0 + (bit + i) * cw + 1).toFixed(1), y: y, width: (cw - 2).toFixed(1), height: h, rx: 3,
          fill: seg.color, 'fill-opacity': 0.22, stroke: seg.color, 'stroke-opacity': 0.7
        }));
      }
      bitsSvg.appendChild(S('text', {
        x: (x0 + (bit + seg.n / 2) * cw).toFixed(1), y: 26, 'text-anchor': 'middle',
        'font-size': 13, 'font-weight': 700, fill: seg.color
      }, seg.label + ' ' + seg.n + ' 位'));
      bit += seg.n;
    });
    bitsSvg.appendChild(S('text', { x: W / 2, y: 88, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--fg-muted)' },
      '共 ' + f.bits + ' 位元（' + f.bits / 8 + ' 位元组）　指数偏移 bias = ' + f.bias));
    // 公式
    clear(formulaEl);
    const tex = '(-1)^{s}\\times 1.\\text{尾数}\\times 2^{\\,\\text{指数}-' + f.bias + '}';
    if (window.katex) window.katex.render(tex, formulaEl, { throwOnError: false });
    else formulaEl.textContent = '值 = (−1)^符号 × 1.尾数 × 2^(指数 − ' + f.bias + ')';
  }

  function drawCards() {
    clear(cardsEl);
    const f = cur();
    const mk = (label, value, sub) => {
      const c = H('div');
      c.style.cssText = 'background:var(--panel-2);border:1px solid var(--border);border-radius:8px;padding:.5rem .65rem';
      c.appendChild(H('div', null, label)).style.cssText = 'font-size:.75rem;color:var(--fg-muted)';
      c.appendChild(H('div', null, value)).style.cssText = 'font-size:.98rem;font-weight:700;color:var(--fg);margin:.15rem 0';
      c.appendChild(H('div', null, sub)).style.cssText = 'font-size:.72rem;color:var(--fg-muted)';
      return c;
    };
    cardsEl.appendChild(mk('最大值', fmtNum(f.max), f.noInf ? '再大就成 NaN（此格式无 inf）' : '再大就溢位成 inf'));
    cardsEl.appendChild(mk('最小正规值', fmtNum(f.minNormal), '更小就进入次正规区、终至下溢'));
    cardsEl.appendChild(mk('epsilon（1 之后的间距）', fmtNum(f.eps), '= 2' + sup('-' + f.m) + '，决定有效位数'));
    cardsEl.appendChild(mk('动态范围', '约 ' + f.range + ' 个数量级', '含次正规数：' + fmtNum(f.minSub) + ' ～ ' + fmtNum(f.max)));
  }

  function drawTable() {
    clear(tableWrap);
    const x = state.x;
    if (!isFinite(x)) { noteEl.hidden = false; noteEl.textContent = '请输入一个有效的数字。'; return; }
    const tbl = H('table');
    tbl.style.cssText = 'width:100%;min-width:430px;border-collapse:collapse;font-size:.85rem';
    const thr = H('tr');
    ['格式', '实际储存值', '相对误差', '状态'].forEach((t) => {
      const th = H('th', null, t);
      th.style.cssText = 'text-align:left;padding:.35rem .5rem;color:var(--fg-muted);font-weight:600;border-bottom:1px solid var(--border)';
      thr.appendChild(th);
    });
    tbl.appendChild(thr);
    const bad = [];
    FORMATS.forEach((f) => {
      const r = quantize(x, f);
      const tr = H('tr');
      if (f.id === state.fmt) tr.style.background = 'var(--accent-soft)';
      const td = (text, css) => {
        const d = H('td', null, text);
        d.style.cssText = 'padding:.35rem .5rem;border-bottom:1px solid var(--border);color:var(--fg)' + (css || '');
        tr.appendChild(d);
      };
      td(f.name, f.id === state.fmt ? ';font-weight:700;color:var(--accent)' : '');
      const warn = ';color:var(--accent-2);font-weight:700';
      if (r.kind === 'over') {
        td((x < 0 ? '-' : '') + (f.noInf ? 'NaN' : 'inf'), warn);
        td('—');
        td('溢位 ⚠', warn);
        bad.push(f.name + '（溢位）');
      } else if (r.kind === 'under') {
        td('0', warn);
        td('100%');
        td('下溢 → 0 ⚠', warn);
        bad.push(f.name + '（下溢）');
      } else {
        td(fmtNum(r.v));
        td(r.err < 1e-12 ? '0（精确）' : fmtNum(r.err * 100) + '%');
        td(r.kind === 'sub' ? '次正规（精度更差）' : '正常', r.kind === 'sub' ? ';color:var(--accent)' : ';color:var(--fg-muted)');
      }
      tbl.appendChild(tr);
    });
    tableWrap.appendChild(tbl);
    noteEl.hidden = bad.length === 0;
    if (bad.length) noteEl.textContent = '⚠ ' + fmtNum(x) + ' 无法被 ' + bad.join('、') + ' 表示——训练中一个这样的值就可能让 loss 变成 NaN 或让权重卡在 0。';
  }

  function drawLine() {
    clear(lineSvg);
    const W = 760, padL = 92, padR = 14, LG0 = -46, LG1 = 40, axisY = 160;
    const X = (lg) => padL + ((lg - LG0) / (LG1 - LG0)) * (W - padL - padR);
    const l10 = Math.log10;
    // 典型梯度量级（约 10^-8 ～ 10^-3）
    lineSvg.appendChild(S('rect', { x: X(-8).toFixed(1), y: 18, width: (X(-3) - X(-8)).toFixed(1), height: axisY - 18, fill: 'var(--accent-2)', 'fill-opacity': 0.07 }));
    lineSvg.appendChild(S('text', { x: X(-5.5).toFixed(1), y: 13, 'text-anchor': 'middle', 'font-size': 10, fill: 'var(--accent-2)' }, '典型梯度量级'));
    // 座标轴与刻度
    lineSvg.appendChild(S('line', { x1: padL, y1: axisY, x2: W - padR, y2: axisY, stroke: 'var(--fg-muted)', 'stroke-width': 1 }));
    for (let lg = -40; lg <= 40; lg += 10) {
      lineSvg.appendChild(S('line', { x1: X(lg).toFixed(1), y1: 18, x2: X(lg).toFixed(1), y2: axisY + 4, stroke: 'var(--border)', 'stroke-width': 1 }));
      lineSvg.appendChild(S('text', { x: X(lg).toFixed(1), y: 176, 'text-anchor': 'middle', 'font-size': 10, fill: 'var(--fg-muted)' }, '10' + sup(lg)));
    }
    // 各格式的可表示范围带
    FORMATS.forEach((f, i) => {
      const sel = f.id === state.fmt, y = 24 + i * 26;
      const col = sel ? 'var(--accent)' : 'var(--link)';
      lineSvg.appendChild(S('text', { x: 8, y: y + 11, 'font-size': 12, 'font-weight': sel ? 700 : 500, fill: sel ? 'var(--accent)' : 'var(--fg-muted)' }, f.name));
      lineSvg.appendChild(S('rect', { x: X(l10(f.minSub)).toFixed(1), y: y, width: (X(l10(f.minNormal)) - X(l10(f.minSub))).toFixed(1), height: 14, fill: col, 'fill-opacity': sel ? 0.3 : 0.15 }));
      lineSvg.appendChild(S('rect', { x: X(l10(f.minNormal)).toFixed(1), y: y, width: (X(l10(f.max)) - X(l10(f.minNormal))).toFixed(1), height: 14, rx: 3, fill: col, 'fill-opacity': sel ? 0.8 : 0.35 }));
    });
    // 输入值标记
    if (isFinite(state.x) && state.x !== 0) {
      const lg = l10(Math.abs(state.x));
      if (lg >= LG0 && lg <= LG1) {
        lineSvg.appendChild(S('line', { x1: X(lg).toFixed(1), y1: 18, x2: X(lg).toFixed(1), y2: axisY, stroke: 'var(--fg)', 'stroke-width': 1.5, 'stroke-dasharray': '4 3' }));
        lineSvg.appendChild(S('text', { x: X(lg).toFixed(1), y: 194, 'text-anchor': 'middle', 'font-size': 10, 'font-weight': 700, fill: 'var(--fg)' }, '↑ 你输入的数字'));
      }
    }
  }

  function refresh() {
    FORMATS.forEach((f) => {
      const b = btns[f.id], sel = f.id === state.fmt;
      b.style.background = sel ? 'var(--accent-soft)' : '';
      b.style.borderColor = sel ? 'var(--accent)' : '';
      b.style.color = sel ? 'var(--accent)' : '';
      b.style.fontWeight = sel ? '700' : '';
      b.setAttribute('aria-pressed', sel);
    });
    drawBits(); drawCards(); drawTable(); drawLine();
    interpEl.textContent = INTERP[state.fmt] + ' ' + (state.fmt === 'e4m3' || state.fmt === 'e5m2' ? TAIL_8 : TAIL_16);
  }

  function render(rootEl) {
    // 面板一：格式选择、位元布局与属性卡
    const p1 = H('div', 'widget-panel');
    const row = H('div', 'widget-row');
    row.style.cssText = 'flex-wrap:wrap;gap:.4rem;margin-bottom:.6rem';
    FORMATS.forEach((f) => {
      const b = H('button', null, f.name + '　1+' + f.e + '+' + f.m);
      b.type = 'button';
      b.addEventListener('click', () => { state.fmt = f.id; refresh(); });
      btns[f.id] = b;
      row.appendChild(b);
    });
    p1.appendChild(row);
    bitsSvg = S('svg', { viewBox: '0 0 680 96', role: 'img', 'aria-label': '位元布局图' });
    bitsSvg.style.cssText = 'width:100%;height:auto;display:block';
    p1.appendChild(bitsSvg);
    formulaEl = H('div');
    formulaEl.style.cssText = 'text-align:center;font-size:.95rem;color:var(--fg);margin:.2rem 0 .6rem';
    p1.appendChild(formulaEl);
    cardsEl = H('div');
    cardsEl.style.cssText = 'display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:.5rem';
    p1.appendChild(cardsEl);
    rootEl.appendChild(p1);
    // 面板二：输入数字与各格式储存结果
    const p2 = H('div', 'widget-panel');
    const irow = H('div', 'widget-row');
    irow.style.cssText = 'flex-wrap:wrap;gap:.5rem;align-items:center;margin-bottom:.5rem';
    const lab = H('label', null, '输入一个数字：');
    lab.style.cssText = 'font-size:.88rem;color:var(--fg)';
    inputEl = H('input');
    inputEl.type = 'number'; inputEl.step = 'any'; inputEl.value = '0.0001';
    inputEl.style.width = '9.5rem';
    inputEl.setAttribute('aria-label', '要测试的数字');
    lab.appendChild(inputEl);
    irow.appendChild(lab);
    [['0.0001', '小梯度'], ['70000', '大激活值'], ['1e-7', '极小梯度']].forEach(([v, t]) => {
      const b = H('button', null, v + '（' + t + '）');
      b.type = 'button';
      b.style.fontSize = '.8rem';
      b.addEventListener('click', () => { inputEl.value = v; state.x = parseFloat(v); drawTable(); drawLine(); });
      irow.appendChild(b);
    });
    p2.appendChild(irow);
    tableWrap = H('div');
    tableWrap.style.cssText = 'overflow-x:auto';
    p2.appendChild(tableWrap);
    noteEl = H('div');
    noteEl.style.cssText = 'margin-top:.5rem;font-size:.85rem;font-weight:600;color:var(--accent-2);border:1px solid var(--accent-2);border-radius:8px;padding:.45rem .65rem;background:var(--accent-soft)';
    noteEl.hidden = true;
    p2.appendChild(noteEl);
    rootEl.appendChild(p2);
    // 面板三：对数刻度数线
    const p3 = H('div', 'widget-panel');
    const t3 = H('div', null, '各格式可表示范围（对数刻度）');
    t3.style.cssText = 'font-size:.88rem;font-weight:700;color:var(--fg);margin-bottom:.4rem';
    p3.appendChild(t3);
    lineSvg = S('svg', { viewBox: '0 0 760 204', role: 'img', 'aria-label': '各浮点格式可表示范围的对数数线' });
    lineSvg.style.cssText = 'width:100%;height:auto;display:block';
    p3.appendChild(lineSvg);
    const legend = H('div', null, '深色带＝正规数范围；浅色带＝次正规数（精度更差）；虚线＝你输入的数字。BF16 的带和 FP32 几乎等长（范围相同、精度较粗），FP16 则窄得多（精度较细、范围有限）。');
    legend.style.cssText = 'margin-top:.45rem;font-size:.8rem;line-height:1.6;color:var(--fg-muted)';
    p3.appendChild(legend);
    rootEl.appendChild(p3);
    // 动态解读
    interpEl = H('div');
    interpEl.style.cssText = 'margin-top:.7rem;font-size:.86rem;line-height:1.75;color:var(--fg-muted);border-left:3px solid var(--accent);padding-left:.7rem';
    rootEl.appendChild(interpEl);
    inputEl.addEventListener('input', () => { state.x = parseFloat(inputEl.value); drawTable(); drawLine(); });
    refresh();
  }

  window.ChapterWidget = {
    title: '浮点格式探索器：FP32 / BF16 / FP16 / FP8',
    intro: '选一种浮点格式，看它的位元布局（符号／指数／尾数）与最大值、最小正规值、epsilon 等关键属性；输入一个数字，看它在各格式下实际被存成什么、误差多大、何时溢位或下溢；下方的对数数线则一眼比较各格式的可表示范围——理解为什么 FP16 需要损失缩放、BF16 不用、FP8 又难在哪。',
    render: render
  };
})();
