/* 第 8 章交互元件：最佳配置三步骤精灵 */
(function () {
  'use strict';

  var CSS = [
    '.ch08-wiz .field { display: flex; flex-direction: column; gap: .35rem; min-width: 130px; flex: 1; }',
    '.ch08-wiz .field-wide { flex-basis: 100%; }',
    '.ch08-wiz .val { font-weight: 600; color: var(--accent); font-variant-numeric: tabular-nums; }',
    '.ch08-wiz .actions { margin-top: 1rem; display: flex; gap: .6rem; }',
    '.ch08-wiz .step-card { margin-top: 1rem; opacity: 0; transform: translateY(10px); transition: opacity .45s ease, transform .45s ease; }',
    '.ch08-wiz .step-card.show { opacity: 1; transform: none; }',
    '.ch08-wiz .step-head { display: flex; align-items: center; gap: .6rem; margin-bottom: .5rem; }',
    '.ch08-wiz .step-num { flex: none; width: 1.7rem; height: 1.7rem; border-radius: 50%; background: var(--accent); color: var(--bg); font: 700 .85rem/1.7rem sans-serif; text-align: center; }',
    '.ch08-wiz .step-title { font-weight: 700; color: var(--fg); }',
    '.ch08-wiz .combo { padding: .55rem .8rem; background: var(--accent-soft); border-left: 3px solid var(--accent); border-radius: 0 8px 8px 0; font-size: .92rem; color: var(--fg); margin: .5rem 0; }',
    '.ch08-wiz .warn { padding: .5rem .8rem; border: 1px dashed var(--accent-2); border-radius: 8px; font-size: .84rem; color: var(--fg); margin: .5rem 0; }',
    '.ch08-wiz .math { font-family: ui-monospace, monospace; font-size: .84rem; background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px; padding: .55rem .8rem; margin: .5rem 0; overflow-x: auto; white-space: nowrap; color: var(--fg); }',
    '.ch08-wiz ul.tips { margin: .4rem 0 .2rem 1.2rem; padding: 0; font-size: .88rem; color: var(--fg); }',
    '.ch08-wiz ul.tips li { margin: .3rem 0; }',
    '.ch08-wiz details { margin-top: .6rem; border: 1px solid var(--border); border-radius: 8px; background: var(--panel); }',
    '.ch08-wiz details summary { cursor: pointer; padding: .45rem .8rem; font-size: .82rem; font-weight: 600; color: var(--link); }',
    '.ch08-wiz details p { margin: 0; padding: .1rem .9rem .7rem; font-size: .84rem; color: var(--fg-muted); line-height: 1.7; }',
    '.ch08-wiz .foot { margin-top: .6rem; font-size: .76rem; color: var(--fg-muted); }',
    '.ch08-wiz .desc { font-size: .86rem; color: var(--fg-muted); margin: .1rem 0 .4rem; }'
  ].join('\n');

  var MODELS = { '1': '1B', '8': '8B', '70': '70B', '405': '405B' };
  var DEFAULTS = { model: '8', gpuExp: 6, gbs: '1048576', seq: '4096' };

  function fmt(n) { return n.toLocaleString('en-US'); }

  /* ---------- 步骤一：塞进内存（依本章规则） ---------- */
  function planStep1(p, gpus, seq) {
    var tp = 1, pp = 1, zero = 1, combo, reason, warn = null;
    if (p < 10) {
      if (gpus >= 512) {
        tp = 8; zero = 1;
        combo = 'TP=8（节点内）＋ DP（ZeRO-1）';
        reason = '模型小于 10B，本可用单卡＋ZeRO 或 TP=8 单一技术解决；但到了 512+ GPU 的规模，纯 DP／ZeRO-3 会因通信成本变得没有效率，故结合节点内 TP。';
      } else if (p <= 1) {
        tp = 1; zero = 3;
        combo = '单卡可容纳 ＋ 纯 DP（ZeRO-3，搭配完整重算）';
        reason = '10B 以下的模型可以只用单一并行化技术：搭配完整重算的 ZeRO-3/DP，或改在 8 张 GPU 上使用 TP=8。';
      } else {
        tp = 8; zero = 1;
        combo = 'TP=8（单节点）；或改用 ZeRO-3/DP ＋ 完整重算';
        reason = '10B 以下的模型可以只用单一并行化技术：在 8 张 GPU 上使用张量并行（TP=8），或搭配完整重算的 ZeRO-3/DP。';
      }
    } else if (p < 100) {
      if (gpus >= 16) {
        tp = 8; pp = 2; zero = 1;
        combo = 'TP=8 ＋ PP=2（每份模型实例占 16 张 GPU）';
        reason = '10B–100B 的模型需要超过 8 张 GPU：可选 TP=8＋PP、TP=8＋ZeRO-3，或纯 ZeRO-3。这里以 TP=8＋PP 为主要建议。';
      } else {
        tp = 8; zero = 3;
        combo = 'TP=8 ＋ ZeRO-3';
        reason = '10B–100B 的模型需要超过 8 张 GPU；GPU 不足 16 张时，以 TP=8 结合 ZeRO-3 分摊参数。';
        warn = '8 张 GPU 对这个大小相当吃紧（GPU 匮乏情境）：建议启用完整激活值重算、增加梯度累积，或增加 GPU。';
      }
    } else {
      if (gpus >= 64) {
        tp = 8; pp = 8; zero = 1;
        combo = 'TP=8 ＋ PP=8（每份模型实例占 64 张 GPU）';
        reason = '超过 100B 的模型，在 TP=8 之外需要加大 PP 深度来分摊参数与内存。';
      } else {
        tp = 8; pp = Math.max(1, gpus / 8); zero = 3;
        combo = 'TP=8 ＋ PP=' + pp + ' ＋ ZeRO-3（勉强尝试）';
        reason = '超过 100B 的模型建议 TP=8 并加大 PP。';
        warn = '目前 GPU 数不足以妥善容纳 405B（建议至少 64 张：TP=8 × PP=8）。请启用完整重算并增加 GPU。';
      }
    }
    if (gpus >= 1024 && pp === 1) {
      pp = 2; zero = 2;
      combo += '；1024+ GPU 规模下再加上 PP（建议设定：TP=8 ＋ ZeRO-2 ＋ PP）';
    }
    var cp = 1, cpNote = null;
    if (seq >= 32768 && gpus / (tp * pp) >= 2) {
      cp = 2;
      cpNote = '序列长度达 ' + fmt(seq) + '：建议跨节点加上上下文并行（CP=2）以分摊长序列的激活值内存。';
    }
    return { tp: tp, pp: pp, cp: cp, zero: zero, combo: combo, reason: reason, warn: warn, cpNote: cpNote };
  }

  /* ---------- 步骤二：达到目标 gbs（token 帐） ---------- */
  function planStep2(s1, gpus, gbsTok, seq) {
    var dp = gpus / (s1.tp * s1.pp * s1.cp);
    var samples = gbsTok / seq;                      // 目标 gbs 换算成序列数
    var mbs = s1.tp * s1.pp === 1 ? 4 : (s1.pp >= 8 ? 1 : 2); // 依模型占用粗估
    if (seq >= 8192) mbs = Math.max(1, mbs >> 1);
    if (seq >= 32768) mbs = 1;
    var warn = null, ga;
    if (dp > samples) {
      mbs = 1; ga = 1;
      warn = 'DP=' + dp + ' 已超过目标 gbs 所需的序列数（' + fmt(samples) + '）：实际 gbs 会是 ' + fmt(dp * seq) +
        ' tokens，超出目标。依本章建议「缩减数据并行、改采其他并行化策略」（加大 TP/PP，长序列则调整 CP），或改选较大的目标 gbs。';
    } else {
      var perRank = samples / dp;                    // 每个 DP rank 每步要吃的序列数
      if (mbs > perRank) mbs = perRank;
      ga = Math.max(1, Math.round(perRank / mbs));
    }
    var actual = dp * mbs * ga * seq;
    return { dp: dp, mbs: mbs, ga: ga, samples: samples, actual: actual, hit: actual === gbsTok, warn: warn };
  }

  /* ---------- 步骤三：吞吐量提示 ---------- */
  function planStep3(s1, s2, gpus) {
    var tips = [];
    tips.push(s1.tp < 8
      ? '将张量并行从 TP=' + s1.tp + ' 扩大到节点内上限（TP=8），利用节点内 NVLink 高速带宽，减少对其他并行化的需求。'
      : 'TP 已达节点大小（TP=8）：跨节点的 TP 通信昂贵，不建议再放大，改调整其他维度。');
    tips.push('尝试多种微批次大小：从 mbs=' + s2.mbs + ' 逐步加大直到逼近 OOM，摊薄每步开销；mbs 加倍时把梯度累积减半（' + s2.ga + ' → ' + Math.max(1, s2.ga >> 1) + '）即可维持相同 gbs。');
    if (s1.zero === 3) {
      tips.push('在维持目标 gbs 的前提下，增加使用 ZeRO-3 的数据并行；当 DP 通信开始成为瓶颈时，转而使用流水线并行（PP）。');
    } else {
      tips.push('DP 扩展的通信代价：目前 DP=' + s2.dp + '，梯度同步／ZeRO 通信量随 DP 上升' + (gpus >= 512 ? '——在 512+ GPU 规模下尤其明显' : '') + '；当 DP 通信成为瓶颈，改把资源投入 PP。');
    }
    tips.push('逐一尝试扩大各种并行化，在 gbs、模型大小、计算与通信之间找最佳平衡——最终仍以实测吞吐量（如 MFU）为准。');
    return tips;
  }

  /* ---------- 卡片渲染 ---------- */
  function card(num, title, bodyHTML, whyHTML) {
    var div = document.createElement('div');
    div.className = 'widget-panel step-card';
    div.innerHTML =
      '<div class="step-head"><span class="step-num">' + num + '</span><span class="step-title">' + title + '</span></div>' +
      bodyHTML +
      '<details><summary>为什么？（本章的取舍）</summary><p>' + whyHTML + '</p></details>' +
      '<div class="foot">⚠️ 以上为粗略指南，实际仍需基准测试（见本章「对数千种配置进行基准测试」）。</div>';
    return div;
  }

  function render(rootEl) {
    var style = document.createElement('style');
    style.textContent = CSS;
    rootEl.appendChild(style);

    var wiz = document.createElement('div');
    wiz.className = 'ch08-wiz';
    wiz.innerHTML =
      '<div class="widget-panel">' +
        '<div class="desc">设定你的模型与集群条件，依本章「步骤一 → 二 → 三」的决策流程产生一份起步配置建议。</div>' +
        '<div class="widget-row">' +
          '<label class="field">模型大小' +
            '<select data-k="model"><option value="1">1B</option><option value="8" selected>8B</option><option value="70">70B</option><option value="405">405B</option></select></label>' +
          '<label class="field">目标 gbs（tokens）' +
            '<select data-k="gbs"><option value="1048576" selected>1M（1,048,576）</option><option value="4194304">4M（4,194,304）</option></select></label>' +
          '<label class="field">序列长度' +
            '<select data-k="seq"><option value="4096" selected>4k（4,096）</option><option value="8192">8k（8,192）</option><option value="32768">32k（32,768）</option></select></label>' +
          '<label class="field field-wide">GPU 数（2 的幂）：<span class="val" data-k="gpuVal">64</span>' +
            '<input type="range" min="3" max="10" step="1" value="6" data-k="gpuExp"></label>' +
        '</div>' +
        '<div class="actions"><button data-k="go">产生建议</button><button class="secondary" data-k="reset">重设</button></div>' +
      '</div>' +
      '<div data-k="out"></div>';
    rootEl.appendChild(wiz);

    var $ = function (k) { return wiz.querySelector('[data-k="' + k + '"]'); };
    var out = $('out');

    $('gpuExp').addEventListener('input', function () {
      $('gpuVal').textContent = fmt(Math.pow(2, +this.value));
    });

    function run() {
      var p = +$('model').value;
      var gpus = Math.pow(2, +$('gpuExp').value);
      var gbsTok = +$('gbs').value;
      var seq = +$('seq').value;

      var s1 = planStep1(p, gpus, seq);
      var s2 = planStep2(s1, gpus, gbsTok, seq);
      var tips = planStep3(s1, s2, gpus);

      out.innerHTML = '';
      var body1 =
        '<div class="combo">建议组合：<strong>' + s1.combo + '</strong>（ZeRO-' + s1.zero + (s1.cp > 1 ? '，CP=' + s1.cp : '') + '）</div>' +
        '<div class="desc">' + s1.reason + '</div>' +
        (s1.cpNote ? '<div class="combo">' + s1.cpNote + '</div>' : '') +
        (s1.warn ? '<div class="warn">😭 ' + s1.warn + '</div>' : '');
      var why1 = '先确保「一份完整的模型实例」放得进 GPU：GPU 充裕时依模型大小挑并行化组合；GPU 匮乏时可用完整激活值重算以计算换内存（训练稍慢），或增加梯度累积在有限内存下处理更大批次。本章亦提醒：混合专家（MoE）架构可跨节点使用专家并行（EP）。';

      var acct = 'gbs = dp × mbs × grad_acc × seq = ' + s2.dp + ' × ' + s2.mbs + ' × ' + s2.ga + ' × ' + fmt(seq) + ' = ' + fmt(s2.actual) + ' tokens';
      var body2 =
        '<div class="combo">建议：<strong>mbs = ' + s2.mbs + '，grad_acc = ' + s2.ga + '</strong>（DP=' + s2.dp + '）</div>' +
        '<div class="math">' + acct + '</div>' +
        '<div class="desc">目标 gbs = ' + fmt(gbsTok) + ' tokens（即每步 ' + fmt(s2.samples) + ' 条长度 ' + fmt(seq) + ' 的序列）' +
        (s2.hit ? ' → ✅ 帐面刚好吻合。' : ' → 与实际值有出入，见下方提醒。') + '</div>' +
        (s2.warn ? '<div class="warn">' + s2.warn + '</div>' : '');
      var why2 = '步骤一结束后的 mbs 与 DP 未必刚好凑出目标批次大小。要「加大」gbs：扩大数据并行或增加梯度累积步数，长序列则可利用上下文并行；要「缩小」gbs：缩减数据并行、改采其他并行化策略，或降低上下文并行的程度。';

      var body3 = '<ul class="tips">' + tips.map(function (t) { return '<li>' + t + '</li>'; }).join('') + '</ul>';
      var why3 = '模型与批次大小的大方向配置跑起来之后，剩下的问题是「用最快的方式训练」：只要内存与通信还不是瓶颈，就优先吃满节点内高速带宽（TP 靠近节点大小）、增加 ZeRO-3 的 DP、DP 通信成瓶颈时转 PP，并尝试多种 mbs。本章实测也显示性能高度取决于实现品质（TP 与 PP 的相对快慢曾因代码优化而互换）。';

      var cards = [
        card(1, '塞进内存（Fitting in Memory）', body1, why1),
        card(2, '达到目标全局批次大小（gbs）', body2, why2),
        card(3, '优化训练吞吐量', body3, why3)
      ];
      cards.forEach(function (c, i) {
        out.appendChild(c);
        setTimeout(function () { c.classList.add('show'); }, 120 + i * 260);
      });
    }

    function reset() {
      $('model').value = DEFAULTS.model;
      $('gbs').value = DEFAULTS.gbs;
      $('seq').value = DEFAULTS.seq;
      $('gpuExp').value = DEFAULTS.gpuExp;
      $('gpuVal').textContent = fmt(Math.pow(2, DEFAULTS.gpuExp));
      out.innerHTML = '';
    }

    $('go').addEventListener('click', run);
    $('reset').addEventListener('click', reset);
  }

  window.ChapterWidget = {
    title: '最佳配置三步骤精灵',
    intro: '依本章的决策流程——步骤一「塞进内存」、步骤二「达到目标 gbs」、步骤三「优化吞吐量」——输入模型大小、GPU 数、目标批次与序列长度，产生一份可作为起点的并行化配置建议。',
    render: render
  };
})();
