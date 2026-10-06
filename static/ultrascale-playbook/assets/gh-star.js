/* GitHub Star 按钮：抓取 repo star 数，localStorage 缓存 1 小时 */
(function () {
  document.querySelectorAll('[data-gh-repo]').forEach(function (el) {
    var repo = el.getAttribute('data-gh-repo');
    var key = 'ghstars:' + repo;
    var chip = el.querySelector('.gh-count');
    if (!chip) return;
    function show(n) {
      chip.textContent = n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n);
      chip.hidden = false;
    }
    try {
      var cached = JSON.parse(localStorage.getItem(key) || 'null');
      if (cached && Date.now() - cached.t < 3600 * 1000) { show(cached.n); return; }
    } catch (e) { /* 缓存损毁就直接重抓 */ }
    fetch('https://api.github.com/repos/' + repo)
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d || typeof d.stargazers_count !== 'number') return;
        show(d.stargazers_count);
        try { localStorage.setItem(key, JSON.stringify({ n: d.stargazers_count, t: Date.now() })); } catch (e) {}
      })
      .catch(function () { /* API 失败就只显示 Star 字样 */ });
  });
})();
