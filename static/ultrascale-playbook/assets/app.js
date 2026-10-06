/* RLHF 中文版 — 章节页渲染器：markdown → HTML、KaTeX 数学、侧栏目录、交互元件挂载 */
(function () {
  const mdEl = document.getElementById('chapter-md');
  const content = document.getElementById('content');
  if (!mdEl || !content) return;

  let md = mdEl.textContent
    .replaceAll('<\\/script', '</script')
    .replaceAll('../webapp/assets/', '../assets/');

  // 先把数学片段抽出，避免 marked 把 LaTeX 的 _ * 当成 markdown 语法
  const mathStore = [];
  const stash = (s) => { mathStore.push(s); return '⦀M' + (mathStore.length - 1) + '⦀'; };
  // 跳过代码区块，仅在一般文字中抽取数学
  const segments = md.split(/(```[\s\S]*?```|`[^`\n]*`)/g);
  md = segments.map((seg, i) => {
    if (i % 2 === 1) return seg; // code 片段原样保留
    return seg
      .replace(/\$\$([\s\S]+?)\$\$/g, (m) => stash(m))
      .replace(/\$(?!\s)((?:\\.|[^$\n\\])+?)(?<!\s)\$/g, (m) => stash(m));
  }).join('');

  let html = marked.parse(md, { mangle: false });
  html = html.replace(/⦀M(\d+)⦀/g, (_, i) => mathStore[+i]
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'));
  content.innerHTML = html;

  if (window.renderMathInElement) {
    renderMathInElement(content, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false },
      ],
      throwOnError: false,
    });
  }

  // 侧栏目录（h2 / h3）＋卷动高亮
  const toc = document.getElementById('toc');
  const heads = content.querySelectorAll('h2, h3');
  if (toc && heads.length) {
    const ol = document.createElement('ol');
    heads.forEach((h, i) => {
      h.id = 'sec-' + i;
      const li = document.createElement('li');
      li.className = 'toc-' + h.tagName.toLowerCase();
      const a = document.createElement('a');
      a.href = '#sec-' + i;
      a.textContent = h.textContent.replace(/（[^）]*）$/, '');
      li.appendChild(a);
      ol.appendChild(li);
    });
    toc.appendChild(ol);
    const links = toc.querySelectorAll('a');
    const spy = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        const idx = [...heads].indexOf(e.target);
        links.forEach((l, i) => l.classList.toggle('active', i === idx));
      });
    }, { rootMargin: '-10% 0px -75% 0px' });
    heads.forEach((h) => spy.observe(h));
  }

  // 交互元件挂载
  const lab = document.getElementById('lab');
  const w = window.ChapterWidget;
  if (lab && w && typeof w.render === 'function') {
    lab.hidden = false;
    if (w.title) lab.querySelector('.lab-title').textContent = '🧪 交互实验室 · ' + w.title;
    if (w.intro) {
      const introEl = document.getElementById('lab-intro');
      introEl.className = 'lab-intro';
      introEl.textContent = w.intro;
    }
    // 内文顶部放一张前往实验室的提示卡
    const banner = document.getElementById('lab-banner');
    if (banner) {
      banner.className = 'lab-banner';
      banner.innerHTML = '🧪 本章附有交互实验：<a href="#lab">' + (w.title || '前往实验室') + ' ↓</a>';
    }
    try {
      w.render(document.getElementById('lab-root'));
    } catch (err) {
      console.error('widget render failed:', err);
      lab.hidden = true;
    }
  }
})();
