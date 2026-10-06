/* 附录 A3：计算／通信重叠条件计算器（公式取自翻译稿 appd.md） */
(function () {
  'use strict';

  var NVLINK = 450e9;      // 节点内 NVLink 带宽（bytes/s）
  var IB = 50e9;           // 跨节点 InfiniBand 带宽（bytes/s）
  var DPD = 8;             // 数据并行度（固定假设）
  var TPD = 8;             // 张量并行度（固定假设）
  var NUM_LAYERS = 32;     // DP 情境的层数（num_params = num_layers·16h²）
  var BUCKET = 25e6;       // DP 梯度桶大小 25 MB
  var PP_NEXT = 4;         // PP 下一阶段层数 num_layers_in_next_pp

  function sig(x) { return Number(x.toPrecision(3)).toString(); }
  function fmtTime(s) {
    if (!isFinite(s) || s <= 0) return '—';
    if (s >= 1) return sig(s) + ' s';
    if (s >= 1e-3) return sig(s * 1e3) + ' ms';
    if (s >= 1e-6) return sig(s * 1e6) + ' µs';
    return sig(s * 1e9) + ' ns';
  }
  function texTime(s) {
    if (!isFinite(s) || s <= 0) return '\\text{—}';
    if (s >= 1) return sig(s) + '\\,\\text{s}';
    if (s >= 1e-3) return sig(s * 1e3) + '\\,\\text{ms}';
    if (s >= 1e-6) return sig(s * 1e6) + '\\,\\mu\\text{s}';
    return sig(s * 1e9) + '\\,\\text{ns}';
  }
  function sci(x) {
    var e = Math.floor(Math.log10(x));
    return Number((x / Math.pow(10, e)).toPrecision(3)) + '{\\times}10^{' + e + '}';
  }
  function tex(elm, src) {
    if (window.katex) window.katex.render(src, elm, { throwOnError: false, displayMode: true });
    else elm.textContent = src;
  }
  function el(tag, cls, parent, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (parent) parent.appendChild(e);
    if (text != null) e.textContent = text;
    return e;
  }
  var SVGNS = 'http://www.w3.org/2000/svg';
  function svgEl(tag, attrs, parent) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  /* ── 四种情境：公式与计算皆对应翻译稿 appd.md ── */
  var SCEN = {
    dp: {
      label: 'DP（all-reduce 梯度）',
      note: '固定假设：DP = 8、bucket = 25 MB、num_layers = 32；num_params = num_layers·16h²、num_tokens = seq·mbs。依翻译稿，t_comm 取「单一桶」的 all-reduce 时间，t_compute 为整个反向传播。',
      texComm: 't_{comm} = t_{comm\\_bucket} = \\frac{bucket\\_size \\cdot 2(DP-1)}{DP \\cdot peak\\_bw}',
      texCompute: 't_{compute} = \\frac{4 \\cdot num\\_tokens \\cdot num\\_params}{peak\\_flops}',
      texCond: '\\frac{t_{comm}}{t_{compute}} = \\frac{num\\_params}{2 \\cdot num\\_tokens} \\cdot \\frac{DP-1}{DP} \\cdot \\frac{peak\\_flops}{peak\\_bw} \\leq 1',
      calc: function (p) {
        var params = NUM_LAYERS * 16 * p.h * p.h, tokens = p.seq * p.mbs;
        var tc = BUCKET * 2 * (DPD - 1) / (DPD * p.bw);
        var tx = 4 * tokens * params / p.fl;
        return { tc: tc, tx: tx,
          subC: 't_{comm} = \\frac{25{\\times}10^{6} \\cdot 2 \\cdot 7}{8 \\cdot ' + p.bwTex + '} \\approx ' + texTime(tc),
          subX: 't_{compute} = \\frac{4 \\cdot ' + tokens + ' \\cdot ' + sci(params) + '}{' + p.flTex + '} \\approx ' + texTime(tx) };
      }
    },
    zero3: {
      label: 'ZeRO-3（all-gather 参数）',
      note: '固定假设：DP = 8。每个 transformer 区块 16h² 位元组的参数需在前向时 all-gather；t_compute 为单一 decoder 层的前向计算。',
      texComm: 't_{comm} = 16h^2 \\cdot \\frac{DP-1}{DP \\cdot peak\\_bw}',
      texCompute: 't_{compute} = \\frac{32 \\cdot seq \\cdot mbs \\cdot h^2}{peak\\_flops}',
      texCond: '\\frac{t_{comm}}{t_{compute}} = \\frac{1}{2 \\cdot seq \\cdot mbs} \\cdot \\frac{DP-1}{DP} \\cdot \\frac{peak\\_flops}{peak\\_bw} \\leq 1',
      calc: function (p) {
        var tc = 16 * p.h * p.h * (DPD - 1) / (DPD * p.bw);
        var tx = 32 * p.seq * p.mbs * p.h * p.h / p.fl;
        return { tc: tc, tx: tx,
          subC: 't_{comm} = 16 \\cdot ' + p.h + '^2 \\cdot \\frac{7}{8 \\cdot ' + p.bwTex + '} \\approx ' + texTime(tc),
          subX: 't_{compute} = \\frac{32 \\cdot ' + p.seq + ' \\cdot ' + p.mbs + ' \\cdot ' + p.h + '^2}{' + p.flTex + '} \\approx ' + texTime(tx) };
      }
    },
    tp: {
      label: 'TP（all-gather 激活值）',
      note: '固定假设：TP = 8。分析某层的激活值 all-gather 能否藏进下一个线性层（参数量 h²）的计算。',
      texComm: 't_{comm} = \\frac{seq \\cdot mbs \\cdot h \\cdot (TP-1)}{TP \\cdot peak\\_bw}',
      texCompute: 't_{compute} = \\frac{2 \\cdot seq \\cdot mbs \\cdot h^2}{TP \\cdot peak\\_flops}',
      texCond: '\\frac{t_{comm}}{t_{compute}} = \\frac{TP-1}{2h} \\cdot \\frac{peak\\_flops}{peak\\_bw} \\leq 1',
      calc: function (p) {
        var tc = p.seq * p.mbs * p.h * (TPD - 1) / (TPD * p.bw);
        var tx = 2 * p.seq * p.mbs * p.h * p.h / (TPD * p.fl);
        return { tc: tc, tx: tx,
          subC: 't_{comm} = \\frac{' + p.seq + ' \\cdot ' + p.mbs + ' \\cdot ' + p.h + ' \\cdot 7}{8 \\cdot ' + p.bwTex + '} \\approx ' + texTime(tc),
          subX: 't_{compute} = \\frac{2 \\cdot ' + p.seq + ' \\cdot ' + p.mbs + ' \\cdot ' + p.h + '^2}{8 \\cdot ' + p.flTex + '} \\approx ' + texTime(tx) };
      }
    },
    pp: {
      label: 'PP（点对点激活值）',
      note: '固定假设：下一阶段层数 num_layers_next = 4。分析阶段间的 P2P 激活值传输能否藏进下一阶段 transformer 区块的计算。',
      texComm: 't_{comm} = \\frac{seq \\cdot mbs \\cdot h}{peak\\_bw}',
      texCompute: 't_{compute} = \\frac{32 \\cdot seq \\cdot mbs \\cdot h^2 \\cdot num\\_layers\\_next}{peak\\_flops}',
      texCond: '\\frac{t_{comm}}{t_{compute}} = \\frac{peak\\_flops}{32 \\cdot h \\cdot num\\_layers\\_next \\cdot peak\\_bw} \\leq 1',
      calc: function (p) {
        var tc = p.seq * p.mbs * p.h / p.bw;
        var tx = 32 * p.seq * p.mbs * p.h * p.h * PP_NEXT / p.fl;
        return { tc: tc, tx: tx,
          subC: 't_{comm} = \\frac{' + p.seq + ' \\cdot ' + p.mbs + ' \\cdot ' + p.h + '}{' + p.bwTex + '} \\approx ' + texTime(tc),
          subX: 't_{compute} = \\frac{32 \\cdot ' + p.seq + ' \\cdot ' + p.mbs + ' \\cdot ' + p.h + '^2 \\cdot 4}{' + p.flTex + '} \\approx ' + texTime(tx) };
      }
    }
  };

  function interpret(key, p, ratio) {
    var inter = p.bw === IB;
    if (key === 'tp') {
      if (ratio > 1 && inter) return '在跨节点 IB 带宽（50 GB/s）下，TP 的 all-gather 藏不进下一个线性层的计算——这就是实践中「TP 不出节点」的数学原因。比值 (TP−1)/2h · peak_flops/peak_bw 只取决于 h、TP 与硬件，调 seq、mbs 都救不了。按「切回 NVLink 看对比」看看节点内的情况。';
      if (ratio > 1) return '即使在 NVLink 节点内带宽下，比值仍大于 1：线性层计算量 ∝ h²、通信量 ∝ h，h 太小时计算不足以掩盖 all-gather。把 h 拉大可压低比值（h ≳ (TP−1)·peak_flops / (2·peak_bw) 时才可重叠）。';
      return '在 NVLink 带宽下，all-gather 能藏进下一个线性层的计算。注意：TP 的比值与 seq、mbs 完全无关——拖动这两支滑杆，两条时间等比缩放，比值不动；真正关键的是 h、TP 与 peak_flops/peak_bw。';
    }
    if (key === 'zero3') {
      if (ratio > 1) return (inter ? '跨节点 IB 带宽下，' : '') + '下一层参数的 all-gather 藏不进当前层计算。ZeRO-3 的比值与 h 无关（分子分母的 h² 相消），唯一解方是加大 seq×mbs（每 GPU 的 token 数）或换更快的互连——试著拉高 seq 或 mbs。';
      return '下一层参数的 all-gather（预取）能藏在当前层的计算背后。比值 ∝ 1/(seq·mbs)：每 GPU 处理的 token 越多越容易重叠，且与 h 无关——调 h 时两条时间等比缩放。';
    }
    if (key === 'pp') {
      if (ratio > 1) return 'P2P 传输藏不进下一阶段的计算——通常是 h 太小或下一阶段层数太少。实践中 PP 的通信量（seq·mbs·h）是四种并行中最小的，很少成为瓶颈。';
      return 'P2P 只需在阶段边界传一份激活值（seq·mbs·h），是四种并行中通信量最小的；即使在跨节点 IB 带宽下也能轻松重叠——这正是「PP、DP 跨节点，TP 留在节点内」布局的数学基础。与 TP 相同，比值与 seq、mbs 无关。';
    }
    if (ratio > 1) return '通信追不上计算：每 GPU 的 token 数（seq·mbs）太少，反向传播太快结束，梯度 all-reduce 来不及躲进去。增大 seq、mbs，或换更快的互连。';
    return '梯度在反向传播进行的同时，以 25 MB 的桶为单位逐桶 all-reduce；单桶通信时间远小于整个反向传播，几乎总能完全重叠——这是 DP 成为最容易扩展的并行方式的原因。但左侧条件式提醒：num_tokens 太小（模型大、批次小）时通信仍会浮出台面。';
  }

  function render(rootEl) {
    var state = { scen: 'tp', bw: IB, h: 8192, seqExp: 12, mbs: 1, tflops: 989 };

    /* ── 控制面板 ── */
    var ctrl = el('div', 'widget-panel', rootEl);
    var row1 = el('div', 'widget-row', ctrl);
    var selWrap = el('div', null, row1);
    selWrap.style.cssText = 'flex:1 1 220px;min-width:200px;';
    el('label', null, selWrap, '并行策略情境').style.cssText = 'display:block;font-size:.85rem;color:var(--fg-muted);margin-bottom:.25rem;';
    var sel = el('select', null, selWrap);
    sel.style.width = '100%';
    Object.keys(SCEN).forEach(function (k) {
      var o = el('option', null, sel, SCEN[k].label);
      o.value = k;
    });
    sel.value = state.scen;
    sel.addEventListener('change', function () { state.scen = sel.value; update(); });

    var bwWrap = el('div', null, row1);
    bwWrap.style.cssText = 'flex:1 1 260px;min-width:220px;';
    el('label', null, bwWrap, '互连带宽 peak_bw').style.cssText = 'display:block;font-size:.85rem;color:var(--fg-muted);margin-bottom:.25rem;';
    var bwRow = el('div', null, bwWrap);
    bwRow.style.cssText = 'display:flex;gap:.5rem;flex-wrap:wrap;';
    var btnNv = el('button', null, bwRow, '节点内 NVLink 450 GB/s');
    var btnIb = el('button', null, bwRow, '跨节点 IB 50 GB/s');
    function setBw(bw) {
      state.bw = bw;
      btnNv.className = bw === NVLINK ? '' : 'secondary';
      btnIb.className = bw === IB ? '' : 'secondary';
      update();
    }
    btnNv.addEventListener('click', function () { setBw(NVLINK); });
    btnIb.addEventListener('click', function () { setBw(IB); });

    var row2 = el('div', 'widget-row', ctrl);
    row2.style.marginTop = '.8rem';
    function slider(labelText, min, max, step, value, fmt, apply) {
      var w = el('div', null, row2);
      w.style.cssText = 'flex:1 1 150px;min-width:130px;';
      var lab = el('label', null, w);
      lab.style.cssText = 'display:flex;justify-content:space-between;gap:.5rem;font-size:.82rem;color:var(--fg-muted);';
      el('span', null, lab, labelText);
      var val = el('span', null, lab);
      val.style.cssText = 'font-family:var(--mono,monospace);color:var(--fg);';
      var input = el('input', null, w);
      input.type = 'range'; input.min = min; input.max = max; input.step = step; input.value = value;
      input.style.width = '100%';
      input.addEventListener('input', function () {
        apply(Number(input.value));
        val.textContent = fmt(Number(input.value));
        update();
      });
      val.textContent = fmt(value);
    }
    slider('隐藏维度 h', 1024, 16384, 128, state.h, function (v) { return String(v); }, function (v) { state.h = v; });
    slider('序列长度 seq', 11, 17, 1, state.seqExp, function (v) { return (Math.pow(2, v) / 1024) + 'k'; }, function (v) { state.seqExp = v; });
    slider('微批次 mbs', 1, 8, 1, state.mbs, function (v) { return String(v); }, function (v) { state.mbs = v; });
    slider('peak_flops（bf16）', 100, 2000, 1, state.tflops, function (v) { return v + ' TFLOPs'; }, function (v) { state.tflops = v; });

    /* ── 结果面板：双条 SVG 横条图 + 判定 + 解读 ── */
    var res = el('div', 'widget-panel', rootEl);
    res.style.marginTop = '1rem';
    function barRow(name) {
      var head = el('div', null, res);
      head.style.cssText = 'display:flex;justify-content:space-between;gap:.5rem;font-size:.85rem;margin-top:.3rem;';
      el('span', null, head, name).style.color = 'var(--fg-muted)';
      var val = el('span', null, head);
      val.style.cssText = 'font-family:var(--mono,monospace);color:var(--fg);';
      var svg = svgEl('svg', { viewBox: '0 0 100 10', preserveAspectRatio: 'none', 'aria-hidden': 'true' });
      svg.style.cssText = 'width:100%;height:18px;display:block;margin:.25rem 0 .5rem;';
      res.appendChild(svg);
      svgEl('rect', { x: 0, y: 0, width: 100, height: 10, rx: 1.2, fill: 'var(--border)', opacity: 0.35 }, svg);
      var bar = svgEl('rect', { x: 0, y: 0, width: 0, height: 10, rx: 1.2, fill: 'var(--accent)' }, svg);
      return { val: val, bar: bar };
    }
    var rowComm = barRow('通信时间 t_comm');
    var rowComp = barRow('计算时间 t_compute');

    var verdictRow = el('div', null, res);
    verdictRow.style.cssText = 'display:flex;align-items:center;gap:.7rem;flex-wrap:wrap;margin-top:.3rem;';
    var badge = el('span', null, verdictRow);
    badge.style.cssText = 'border:1.5px solid currentColor;border-radius:999px;padding:.18rem .75rem;font-size:.85rem;font-weight:600;';
    var flipBtn = el('button', 'secondary', verdictRow);
    flipBtn.addEventListener('click', function () { setBw(state.bw === IB ? NVLINK : IB); });
    var reading = el('p', null, res);
    reading.style.cssText = 'margin:.7rem 0 0;font-size:.88rem;line-height:1.7;color:var(--fg);';

    /* ── 公式面板：KaTeX 公式与代入结果 ── */
    var fpanel = el('div', 'widget-panel', rootEl);
    fpanel.style.marginTop = '1rem';
    el('div', null, fpanel, '公式（取自附录 A3）与代入结果').style.cssText = 'font-size:.85rem;font-weight:600;color:var(--fg-muted);margin-bottom:.3rem;';
    function texBlock() {
      var d = el('div', null, fpanel);
      d.style.cssText = 'overflow-x:auto;padding:.15rem 0;';
      return d;
    }
    var fComm = texBlock(), fCompute = texBlock(), fCond = texBlock(), fSub = texBlock();
    var noteEl = el('p', null, fpanel);
    noteEl.style.cssText = 'margin:.5rem 0 0;font-size:.78rem;line-height:1.6;color:var(--fg-muted);';

    function update() {
      var s = SCEN[state.scen];
      var p = {
        h: state.h, seq: Math.pow(2, state.seqExp), mbs: state.mbs,
        bw: state.bw, fl: state.tflops * 1e12,
        bwTex: (state.bw / 1e9) + '{\\times}10^{9}', flTex: state.tflops + '{\\times}10^{12}'
      };
      var r = s.calc(p);
      var ratio = r.tc / r.tx;
      var ok = ratio <= 1;
      var color = ok ? 'var(--accent)' : 'var(--accent-2)';

      var max = Math.max(r.tc, r.tx);
      rowComm.bar.setAttribute('width', Math.max(0.6, 100 * r.tc / max));
      rowComm.bar.setAttribute('fill', color);
      rowComm.val.textContent = '≈ ' + fmtTime(r.tc);
      rowComp.bar.setAttribute('width', Math.max(0.6, 100 * r.tx / max));
      rowComp.bar.setAttribute('fill', 'var(--fg-muted)');
      rowComp.val.textContent = '≈ ' + fmtTime(r.tx);

      badge.style.color = color;
      badge.textContent = ok
        ? '可重叠：t_comm / t_compute ≈ ' + sig(ratio) + ' ≤ 1'
        : '无法完全重叠——通信成为瓶颈（t_comm / t_compute ≈ ' + sig(ratio) + '）';
      flipBtn.textContent = state.bw === IB ? '切回 NVLink 看对比' : '切到跨节点 IB 看对比';
      reading.textContent = interpret(state.scen, p, ratio);

      tex(fComm, s.texComm);
      tex(fCompute, s.texCompute);
      tex(fCond, s.texCond);
      tex(fSub, r.subC + ',\\qquad ' + r.subX + ',\\qquad \\frac{t_{comm}}{t_{compute}} \\approx ' + sig(ratio));
      noteEl.textContent = s.note;
    }

    setBw(IB); // 预设：TP + 跨节点带宽（最有教学价值的警示案例），setBw 内含首次 update()
  }

  window.ChapterWidget = {
    title: '计算／通信重叠条件计算器',
    intro: '选择并行策略与硬件参数，即时比较 t_comm 与 t_compute：只有当比值 t_comm/t_compute ≤ 1，通信才能完全藏进计算——这条不等式决定了每种并行方式该部署在节点内还是跨节点。',
    render: render
  };
})();
