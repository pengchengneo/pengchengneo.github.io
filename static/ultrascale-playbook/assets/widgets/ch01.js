/* 第 1 章交互元件：单 GPU 训练内存计算器（呼应原书 predict-memory 工具） */
(function () {
  'use strict';

  // ---------- 模型预设与常数 ----------
  var MODELS = [
    { id: '1B',  name: 'Llama-3.2 1B',  L: 16, h: 2048, heads: 32 },
    { id: '3B',  name: 'Llama-3.2 3B',  L: 28, h: 3072, heads: 24 },
    { id: '8B',  name: 'Llama-3.1 8B',  L: 32, h: 4096, heads: 32 },
    { id: '70B', name: 'Llama-3.1 70B', L: 80, h: 8192, heads: 64 }
  ];
  var VOCAB = 128256;            // Llama 3 系列词汇表大小 v
  var CAP = 80;                  // H100 内存容量（GB）
  var SEGS = [
    { key: 'w', name: '权重',       fill: 'var(--accent)' },
    { key: 'g', name: '梯度',       fill: 'var(--link)' },
    { key: 'o', name: '优化器状态', fill: 'url(#ch1w-hatch)' },
    { key: 'a', name: '激活值',     fill: 'var(--accent-2)' }
  ];
  var SVG_NS = 'http://www.w3.org/2000/svg';

  // ---------- 小工具 ----------
  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }
  function svg(tag, attrs) {
    var e = document.createElementNS(SVG_NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  function tex(node, t) {
    if (window.katex) window.katex.render(t, node, { throwOnError: false, displayMode: true });
    else node.textContent = t;
  }
  function fmtGB(x) {
    if (x >= 1000) return (x / 1000).toFixed(x >= 10000 ? 0 : 1) + ' TB';
    if (x >= 100) return x.toFixed(0) + ' GB';
    if (x >= 10) return x.toFixed(1) + ' GB';
    return x.toFixed(2) + ' GB';
  }
  function fmtInt(n) { return Math.round(n).toLocaleString('en-US'); }
  function kLabel(seq) { return (seq / 1024) + 'k'; }
  function segPath(x, y, w, h, rl, rr) { // 左右可各自带圆角的长条
    rl = Math.min(rl, w / 2); rr = Math.min(rr, w / 2);
    return 'M' + (x + rl) + ' ' + y + 'H' + (x + w - rr) +
      (rr ? 'a' + rr + ' ' + rr + ' 0 0 1 ' + rr + ' ' + rr : '') + 'V' + (y + h - rr) +
      (rr ? 'a' + rr + ' ' + rr + ' 0 0 1 ' + -rr + ' ' + rr : '') + 'H' + (x + rl) +
      (rl ? 'a' + rl + ' ' + rl + ' 0 0 1 ' + -rl + ' ' + -rl : '') + 'V' + (y + rl) +
      (rl ? 'a' + rl + ' ' + rl + ' 0 0 1 ' + rl + ' ' + -rl : '') + 'Z';
  }

  // ---------- 内存公式（忠于翻译稿） ----------
  // N = h·v + L·(12h² + 13h) + 2h
  function paramCount(m) { return m.h * VOCAB + m.L * (12 * m.h * m.h + 13 * m.h) + 2 * m.h; }
  function compute(st) {
    var m = st.model, N = paramCount(m), GB = 1e9;
    var bf16 = st.prec === 'bf16';
    // bf16 混合精度：2N + 2N + 12N（优化器含 FP32 主权重 4N＋Adam 动量/变异数 8N）
    // fp32 全精度：  4N + 4N + 8N
    var w = (bf16 ? 2 : 4) * N, g = (bf16 ? 2 : 4) * N, o = (bf16 ? 12 : 8) * N;
    // m_act = L·seq·mbs·h·(34 + 5·n_heads·seq/h) bytes（混合精度；fp32 激活每值 4 位元组 → ×2）
    var base = m.L * st.seq * st.mbs * m.h, attn = 5 * m.heads * st.seq / m.h, mult = bf16 ? 1 : 2;
    var aNone = base * (34 + attn) * mult, aSel = base * 34 * mult, aFull = base * 2 * mult;
    var a = st.rc === 'none' ? aNone : st.rc === 'sel' ? aSel : aFull;
    return {
      N: N, attn: attn, wGB: w / GB, gGB: g / GB, oGB: o / GB, aGB: a / GB,
      aNoneGB: aNone / GB, aSelGB: aSel / GB, aFullGB: aFull / GB,
      staticGB: (w + g + o) / GB, totalGB: (w + g + o + a) / GB
    };
  }

  // ---------- 元件 ----------
  window.ChapterWidget = {
    title: '单 GPU 训练内存计算器',
    intro: '调整模型尺寸、序列长度、微批次与重算策略，观察权重、梯度、优化器状态与激活各占多少内存、什么时候会撑爆一颗 H100（80 GB）；再拉动梯度累积，体会「内存不变、有效批次变大」。',
    render: function (rootEl) {
      var st = { model: MODELS[2], seq: 4096, mbs: 1, rc: 'sel', prec: 'bf16', ga: 8 };

      // 隐藏 defs：优化器状态的斜线纹理（灰色低彩度，靠纹理辅助辨识）
      var defs = el('div');
      defs.innerHTML = '<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>' +
        '<pattern id="ch1w-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">' +
        '<rect width="5" height="5" fill="var(--fg-muted)"/><line x1="0" y1="0" x2="0" y2="5" stroke="var(--panel-2)" stroke-width="1.6"/>' +
        '</pattern></defs></svg>';
      rootEl.appendChild(defs);

      // --- 面板 1：控制项 ---
      var p1 = el('div', 'widget-panel');
      function group(labelHtml, ctrl) {
        var g = el('div');
        g.style.cssText = 'display:flex;flex-direction:column;gap:.3rem;min-width:130px;flex:1';
        var lb = el('label', '', labelHtml);
        lb.style.cssText = 'font-size:.82rem;color:var(--fg-muted)';
        g.appendChild(lb); g.appendChild(ctrl);
        return g;
      }
      var selModel = el('select');
      MODELS.forEach(function (m, i) {
        var o = el('option', '', m.name); o.value = i; selModel.appendChild(o);
      });
      selModel.value = 2;
      var selPrec = el('select');
      selPrec.innerHTML = '<option value="bf16">bf16 混合精度</option><option value="fp32">fp32 全精度</option>';
      var selRc = el('select');
      selRc.innerHTML = '<option value="none">无重算</option><option value="sel">选择性重算</option><option value="full">完整重算</option>';
      selRc.value = 'sel';
      var rgSeq = el('input'); rgSeq.type = 'range'; rgSeq.min = 0; rgSeq.max = 7; rgSeq.step = 1; rgSeq.value = 2; // 2^(10+i)
      var rgMbs = el('input'); rgMbs.type = 'range'; rgMbs.min = 1; rgMbs.max = 16; rgMbs.step = 1; rgMbs.value = 1;
      var lbSeq = '序列长度 seq：<b style="color:var(--fg)">4k</b>';
      var row1 = el('div', 'widget-row');
      row1.appendChild(group('模型', selModel));
      row1.appendChild(group('精度', selPrec));
      row1.appendChild(group('激活重算', selRc));
      var row2 = el('div', 'widget-row'); row2.style.marginTop = '.8rem';
      var gSeq = group(lbSeq, rgSeq), gMbs = group('微批次 mbs：<b style="color:var(--fg)">1</b>', rgMbs);
      row2.appendChild(gSeq); row2.appendChild(gMbs);
      var chip = el('div'); chip.style.cssText = 'margin-top:.7rem;font-size:.8rem;color:var(--fg-muted);font-family:var(--mono)';
      p1.appendChild(row1); p1.appendChild(row2); p1.appendChild(chip);
      rootEl.appendChild(p1);

      // --- 面板 2：堆叠横条图 + 图例 ---
      var p2 = el('div', 'widget-panel'); p2.style.marginTop = '12px';
      var wrap = el('div'); wrap.style.position = 'relative';
      var chart = svg('svg', { role: 'img', 'aria-label': '单 GPU 内存占用堆叠横条图' });
      chart.style.cssText = 'display:block;width:100%;height:auto';
      var tip = el('div');
      tip.style.cssText = 'display:none;position:absolute;pointer-events:none;z-index:5;background:var(--panel);' +
        'border:1px solid var(--border);border-radius:8px;padding:4px 9px;font-size:.78rem;color:var(--fg);box-shadow:var(--shadow);white-space:nowrap';
      wrap.appendChild(chart); wrap.appendChild(tip);
      var legend = el('div'); legend.style.cssText = 'display:flex;flex-wrap:wrap;gap:.4rem 1.2rem;margin-top:.6rem;font-size:.82rem';
      var legendVals = {};
      SEGS.forEach(function (s) {
        var item = el('span'); item.style.cssText = 'display:inline-flex;align-items:center;gap:.4rem;color:var(--fg-muted)';
        var sw = svg('svg', { width: 12, height: 12 });
        sw.appendChild(svg('rect', { width: 12, height: 12, rx: 3, fill: s.fill }));
        var val = el('b'); val.style.color = 'var(--fg)';
        item.appendChild(sw); item.appendChild(document.createTextNode(s.name + ' ')); item.appendChild(val);
        legend.appendChild(item); legendVals[s.key] = val;
      });
      p2.appendChild(wrap); p2.appendChild(legend);
      rootEl.appendChild(p2);

      // --- 面板 3：激活公式（KaTeX）＋动态解读 ---
      var p3 = el('div', 'widget-panel'); p3.style.marginTop = '12px';
      var fStatic = el('div'); fStatic.style.cssText = 'overflow-x:auto;font-size:.95rem';
      tex(fStatic, 'm_{act} = L \\cdot seq \\cdot mbs \\cdot h \\cdot \\Big(34 + \\frac{5 \\cdot n_{heads} \\cdot seq}{h}\\Big)\\ \\text{bytes}');
      var fDyn = el('div'); fDyn.style.cssText = 'overflow-x:auto;font-size:.9rem;color:var(--fg)';
      var fStat = el('div'); fStat.style.cssText = 'overflow-x:auto;font-size:.9rem';
      var noteP = el('p', '', '混合精度的静态部分与 fp32 同为 <b>16N</b>——混合精度本身不省总内存，赚到的是更快的低精度运算与砍半的激活内存。');
      noteP.style.cssText = 'font-size:.82rem;color:var(--fg-muted);margin:.6rem 0 0';
      var interp = el('ul'); interp.style.cssText = 'margin:.7rem 0 0;padding-left:1.2rem;font-size:.88rem;line-height:1.7';
      p3.appendChild(fStatic); p3.appendChild(fDyn); p3.appendChild(fStat); p3.appendChild(noteP); p3.appendChild(interp);
      rootEl.appendChild(p3);

      // --- 面板 4：梯度累积补充列 ---
      var p4 = el('div', 'widget-panel'); p4.style.marginTop = '12px';
      p4.appendChild(el('div', '', '<b>补充：梯度累积</b>')).style.marginBottom = '.5rem';
      var rgGa = el('input'); rgGa.type = 'range'; rgGa.min = 1; rgGa.max = 32; rgGa.step = 1; rgGa.value = st.ga;
      var rowGa = el('div', 'widget-row');
      var gGa = group('累积步数 grad_acc：<b style="color:var(--fg)">8</b>', rgGa);
      rowGa.appendChild(gGa);
      var gaTex = el('div'); gaTex.style.cssText = 'overflow-x:auto;font-size:.9rem';
      var gaNote = el('p'); gaNote.style.cssText = 'font-size:.82rem;color:var(--fg-muted);margin:.4rem 0 0';
      p4.appendChild(rowGa); p4.appendChild(gaTex); p4.appendChild(gaNote);
      rootEl.appendChild(p4);

      // ---------- 绘图 ----------
      function drawBar(r) {
        var W = Math.max(280, wrap.clientWidth || 640), H = 112;
        chart.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
        chart.setAttribute('width', W); chart.setAttribute('height', H);
        while (chart.firstChild) chart.removeChild(chart.firstChild);
        var pad = 4, innerW = W - pad * 2, axisMax = Math.max(CAP, r.totalGB) * 1.06;
        var x = function (gb) { return pad + gb / axisMax * innerW; };
        var barY = 40, barH = 32, over = r.totalGB > CAP;
        // 背景轨
        chart.appendChild(svg('rect', { x: pad, y: barY, width: innerW, height: barH, rx: 5, fill: 'var(--panel)', stroke: 'var(--border)' }));
        // 四色分段（段间留 2px 缝隙）
        var order = [r.wGB, r.gGB, r.oGB, r.aGB], cx = pad;
        var vis = order.map(function (gb) { return gb / axisMax * innerW; });
        var lastVis = -1;
        vis.forEach(function (wpx, i) { if (wpx >= 0.5) lastVis = i; });
        order.forEach(function (gb, i) {
          var wpx = vis[i];
          if (wpx < 0.5) { cx += wpx; return; }
          var gap = (i === lastVis) ? 0 : Math.min(2, wpx / 2);
          var seg = svg('path', {
            d: segPath(cx, barY, wpx - gap, barH, i === 0 ? 5 : 0, i === lastVis ? 4 : 0),
            fill: SEGS[i].fill
          });
          seg.style.cursor = 'default';
          (function (i2, gb2) {
            seg.addEventListener('mousemove', function (evt) {
              var b = wrap.getBoundingClientRect();
              tip.textContent = SEGS[i2].name + '：' + fmtGB(gb2) + '（' + (gb2 / r.totalGB * 100).toFixed(1) + '%）';
              tip.style.display = 'block';
              tip.style.left = Math.min(evt.clientX - b.left + 12, b.width - 150) + 'px';
              tip.style.top = (evt.clientY - b.top - 34) + 'px';
              seg.setAttribute('stroke', 'var(--fg)');
            });
            seg.addEventListener('mouseleave', function () { tip.style.display = 'none'; seg.removeAttribute('stroke'); });
          })(i, gb);
          chart.appendChild(seg);
          cx += wpx;
        });
        // H100 容量线
        var xc = x(CAP), lineCol = over ? 'var(--accent-2)' : 'var(--fg-muted)';
        chart.appendChild(svg('line', { x1: xc, y1: barY - 8, x2: xc, y2: barY + barH + 8, stroke: lineCol, 'stroke-width': 1.5, 'stroke-dasharray': '4 3' }));
        var capLb = svg('text', { x: xc > W - 84 ? xc - 5 : xc + 5, y: barY + barH + 18, 'text-anchor': xc > W - 84 ? 'end' : 'start', 'font-size': 11, fill: lineCol });
        capLb.textContent = 'H100 80 GB';
        chart.appendChild(capLb);
        // 总量标签
        var tot = svg('text', { x: pad, y: 18, 'font-size': 13.5, 'font-weight': 600, fill: 'var(--fg)' });
        tot.textContent = '总计 ' + fmtGB(r.totalGB);
        if (over) {
          var warn = svg('tspan', { fill: 'var(--accent-2)' });
          warn.textContent = '　⚠ 超出 H100 ' + fmtGB(r.totalGB - CAP);
          tot.appendChild(warn);
        }
        chart.appendChild(tot);
        // 轴端点
        var t0 = svg('text', { x: pad, y: H - 4, 'font-size': 10, fill: 'var(--fg-muted)' }); t0.textContent = '0';
        var t1 = svg('text', { x: W - pad, y: H - 4, 'text-anchor': 'end', 'font-size': 10, fill: 'var(--fg-muted)' }); t1.textContent = fmtGB(axisMax);
        chart.appendChild(t0); chart.appendChild(t1);
      }

      // ---------- 解读文字 ----------
      function interpret(r) {
        var items = [], m = st.model, share = r.aGB / r.totalGB * 100, sl = kLabel(st.seq);
        if (r.totalGB > CAP) {
          items.push('<b style="color:var(--accent-2)">爆内存</b>：总计 ' + fmtGB(r.totalGB) + '，超出 H100 的 80 GB。' +
            (st.rc === 'none' ? '先试试「选择性重算」。' : st.rc === 'sel' ? '可再试「完整重算」、调小 mbs／seq，或等下一章的数据并行与 ZeRO。' : '单靠重算已到极限——得缩小 mbs／seq，或动用多 GPU 并行技术。'));
        } else {
          items.push('总计 ' + fmtGB(r.totalGB) + '，放得进单颗 H100，剩约 ' + fmtGB(CAP - r.totalGB) + ' 余裕。');
        }
        if (r.staticGB > CAP) {
          items.push('光是权重＋梯度＋优化器状态（16N ≈ ' + fmtGB(r.staticGB) + '）就超过 80 GB——正如书中所说，模型一旦到 7B 级（！），单卡连静态部分都放不下。');
        }
        if (share >= 30) {
          items.push('seq=' + sl + ' 时激活占 ' + share.toFixed(0) + '%——激活随序列长度平方成长，这就是为什么长上下文需要重算与上下文并行（CP）。');
        }
        if (st.rc === 'none') {
          items.push('未使用重算：注意力项 5·n_heads·seq/h = ' + fmtInt(r.attn) + '，' + (r.attn > 34 ? '已压过固定项 34' : '与固定项 34 相当') + '；切到「选择性重算」可省 ' + ((1 - r.aSelGB / r.aNoneGB) * 100).toFixed(0) + '% 激活内存。');
        } else if (st.rc === 'sel') {
          items.push('选择性重算丢掉注意力激活：由 ' + fmtGB(r.aNoneGB) + ' 降到 ' + fmtGB(r.aSelGB) + '（−' + ((1 - r.aSelGB / r.aNoneGB) * 100).toFixed(0) + '%），GPT-3 上实测代价仅约 2.7% 计算——FlashAttention 使用者其实早已享有。');
        } else {
          items.push('完整重算只在层边界留检查点：激活压到 ' + fmtGB(r.aFullGB) + '（相对无重算 −' + ((1 - r.aFullGB / r.aNoneGB) * 100).toFixed(1) + '%），但反向传播要多付一次完整前向，计算时间多 30–40%。');
        }
        interp.innerHTML = items.map(function (t) { return '<li>' + t + '</li>'; }).join('');
      }

      // ---------- 更新 ----------
      function update() {
        st.model = MODELS[+selModel.value];
        st.seq = 1 << (10 + +rgSeq.value);
        st.mbs = +rgMbs.value;
        st.rc = selRc.value;
        st.prec = selPrec.value;
        st.ga = +rgGa.value;
        gSeq.firstChild.innerHTML = '序列长度 seq：<b style="color:var(--fg)">' + kLabel(st.seq) + '</b>';
        gMbs.firstChild.innerHTML = '微批次 mbs：<b style="color:var(--fg)">' + st.mbs + '</b>';
        gGa.firstChild.innerHTML = '累积步数 grad_acc：<b style="color:var(--fg)">' + st.ga + '</b>';
        var r = compute(st), m = st.model;
        chip.textContent = 'L=' + m.L + '　h=' + m.h + '　n_heads=' + m.heads + '　v=' + fmtInt(VOCAB) + '　→　N ≈ ' + (r.N / 1e9).toFixed(2) + 'B（依公式估计）';
        drawBar(r);
        legendVals.w.textContent = fmtGB(r.wGB); legendVals.g.textContent = fmtGB(r.gGB);
        legendVals.o.textContent = fmtGB(r.oGB); legendVals.a.textContent = fmtGB(r.aGB);
        // 代入值公式
        var x2 = st.prec === 'fp32' ? ' \\times 2' : '', gbTex = ' \\approx ' + fmtGB(r.aGB).replace(' ', '\\ \\text{') + '}';
        var head = m.L + ' \\cdot ' + st.seq + ' \\cdot ' + st.mbs + ' \\cdot ' + m.h;
        if (st.rc === 'none') tex(fDyn, 'm_{act} = ' + head + ' \\cdot (34 + ' + fmtInt(r.attn).replace(/,/g, '\\,') + ')' + x2 + gbTex);
        else if (st.rc === 'sel') tex(fDyn, 'm_{act} = ' + head + ' \\cdot (34 + \\cancel{' + fmtInt(r.attn).replace(/,/g, '\\,') + '})' + x2 + gbTex);
        else tex(fDyn, 'm_{act} \\approx 2 \\cdot ' + head + x2 + gbTex + '\\quad\\text{（只留层边界）}');
        // 静态部分公式
        var sTex = st.prec === 'bf16'
          ? 'm_{params} + m_{grad} + m_{opt} = (2 + 2 + 12) \\cdot N = 16N \\approx '
          : 'm_{params} + m_{grad} + m_{opt} = (4 + 4 + 8) \\cdot N = 16N \\approx ';
        tex(fStat, sTex + fmtGB(r.staticGB).replace(' ', '\\ \\text{') + '}');
        interpret(r);
        // 梯度累积列
        var gbs = st.mbs * st.ga;
        tex(gaTex, 'gbs = mbs \\times grad\\_acc = ' + st.mbs + ' \\times ' + st.ga + ' = ' + gbs +
          '\\ \\text{个样本} \\;\\Rightarrow\\; bst = ' + fmtInt(gbs * st.seq).replace(/,/g, '\\,') + '\\ \\text{词元}');
        gaNote.textContent = st.ga === 1
          ? '拉动滑杆：mbs 固定、上方内存长条完全不变，有效批次 gbs 却能任意放大。'
          : '上方内存长条完全不变——每次前向／反向只算一个微批次（mbs=' + st.mbs + '），梯度在优化器步骤前累积平均；代价是每个优化步骤要连跑 ' + st.ga + ' 次前向／反向。';
      }

      [selModel, selPrec, selRc].forEach(function (c) { c.addEventListener('change', update); });
      [rgSeq, rgMbs, rgGa].forEach(function (c) { c.addEventListener('input', update); });
      if (typeof ResizeObserver !== 'undefined') {
        var raf = 0;
        new ResizeObserver(function () {
          cancelAnimationFrame(raf);
          raf = requestAnimationFrame(function () { drawBar(compute(st)); });
        }).observe(wrap);
      }
      update();
    }
  };
})();
