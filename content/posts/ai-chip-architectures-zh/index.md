---
title: "AI 芯片架构全景：GPU、TPU、Trainium、WSE 与 LPU"
date: 2026-08-24
draft: false
showTableOfContents: true
wideContent: true
manualTableOfContents:
  - id: "the-problem"
    title: "问题：AI 计算与内存墙"
  - id: "nvidia-gpu"
    title: "NVIDIA GPU"
  - id: "google-tpu"
    title: "Google TPU"
  - id: "amd-gpu"
    title: "AMD GPU"
  - id: "cerebras-wse"
    title: "Cerebras WSE"
  - id: "aws-trainium"
    title: "AWS Trainium"
  - id: "groq-lpu"
    title: "Groq LPU"
  - id: "comparison"
    title: "架构对比"
tags: ["AI 芯片", "GPU", "TPU", "Trainium", "Cerebras", "Groq"]
categories: ["AI Infra"]
summary: "系统比较 NVIDIA GPU、Google TPU、AMD GPU、Cerebras WSE、AWS Trainium 与 Groq LPU 的设计理念、计算与内存架构、扩展方式和软件栈。"
---

<style>
.article-body a:has(.katex),
.article-body a:has(.katex):hover {
  border-bottom: none;
}
.article-body code {
  font-family: "SF Mono", "Fira Code", "Fira Mono", Menlo, Consolas, monospace;
  font-size: 0.84em;
  color: #3a3a3a;
  background: #f0efeb;
  padding: 2px 6px;
  border-radius: 3px;
  letter-spacing: -0.01em;
}
.article-body pre {
  background: #f0efeb;
  padding: 16px 20px;
  border-radius: 4px;
  overflow-x: auto;
  margin-bottom: 20px;
}
.article-body pre code {
  background: none;
  padding: 0;
  border-radius: 0;
  font-size: 0.82em;
  line-height: 1.6;
}
.article-body pre code.hljs { color: #3a3a3a; background: none; }
.article-body .hljs-comment,
.article-body .hljs-quote { color: #5a9a5a; font-style: italic; }
.article-body .hljs-keyword,
.article-body .hljs-selector-tag,
.article-body .hljs-literal,
.article-body .hljs-section,
.article-body .hljs-doctag,
.article-body .hljs-name { color: #7b5ea7; }
.article-body .hljs-type,
.article-body .hljs-class .hljs-title,
.article-body .hljs-built_in { color: #2a8a6a; }
.article-body .hljs-title,
.article-body .hljs-title.function_,
.article-body .hljs-function .hljs-title { color: #2a6ab5; }
.article-body .hljs-string,
.article-body .hljs-symbol,
.article-body .hljs-bullet,
.article-body .hljs-link { color: #2a8a6a; }
.article-body .hljs-number,
.article-body .hljs-variable.constant_ { color: #c04040; }
.article-body .hljs-meta,
.article-body .hljs-meta .hljs-keyword,
.article-body .hljs-meta .hljs-string { color: #b8702a; }
.article-body .hljs-attr,
.article-body .hljs-attribute,
.article-body .hljs-variable,
.article-body .hljs-template-variable,
.article-body .hljs-params { color: #3a3a3a; }
.article-body .hljs-operator,
.article-body .hljs-punctuation { color: #6a6a6a; }
.article-body .hljs-emphasis { font-style: italic; }
.article-body .hljs-strong { font-weight: 600; }
.article-body .katex-display {
  margin: 24px 0;
  overflow-x: auto;
  text-align: center;
}
.article-body .katex-display > .katex > .katex-html {
  display: inline-block;
  padding: 8px 16px;
}
.article-body .katex-html {
  background: #f0efeb;
  padding: 4px 5px;
  border-radius: 3px;
  display: inline-block;
  vertical-align: middle;
}
.article-body .katex {
  font-size: 1.05em;
  margin: 0 1px;
  position: relative;
  top: -1px;
}
.article-body p {
  margin-bottom: 20px;
}
.article-body hr {
  border: none;
  border-top: 1px solid #e0e0e0;
  margin: 28px 0;
}
.article-body h3 {
  font-weight: 700;
  font-style: normal;
  font-size: 26px;
  line-height: 1.35;
  color: #1a1a1a;
  margin-top: 52px;
  margin-bottom: 18px;
}
.article-body h3 .heading-logo {
  display: inline-block;
  height: 18px;
  width: 18px;
  vertical-align: -4px;
  margin: 0 8px 0 0;
  border-radius: 3px;
  border-bottom: none;
  object-fit: contain;
  object-position: center;
}
.article-body h3 .heading-logo.is-tight {
  height: 13px;
  width: 18px;
  vertical-align: -2px;
}
.article-body h3 .heading-logo.is-mid {
  height: 15px;
  width: 18px;
  vertical-align: -3px;
}
[data-theme="dark"] .article-body h3 .heading-logo.is-mono {
  filter: invert(1);
}
.article-body h4 {
  font-weight: 700;
  font-size: 21px;
  line-height: 1.4;
  color: #292524;
  text-transform: none;
  letter-spacing: 0;
  margin-top: 40px;
  margin-bottom: 14px;
}
.article-body h4 a[data-popup] {
  color: inherit;
  border-bottom: none;
  text-decoration: none;
  cursor: pointer;
  transition: color 0.15s ease;
}
.article-body h4 a[data-popup]:hover {
  color: #4a4a4a;
  border-bottom: none;
}
.article-body h5 {
  font-weight: 700;
  font-size: 17px;
  line-height: 1.45;
  font-style: normal;
  color: #44403c;
  margin-top: 32px;
  margin-bottom: 10px;
}
.article-body .mesi-key {
  text-align: center;
  font-size: 11.5px;
  color: #4a4a4a;
  margin: -8px auto 4px;
  max-width: 660px;
}
.article-body .mesi-key .key-rdhit  { color: #5a8a4a; font-size: 14px; vertical-align: -1px; }
.article-body .mesi-key .key-rdmiss { color: #4a7a9a; font-size: 14px; vertical-align: -1px; }
.article-body .mesi-key .key-wrhit  { color: #c8a058; font-size: 14px; vertical-align: -1px; }
.article-body .mesi-key .key-wrmiss { color: #c4503a; font-size: 14px; vertical-align: -1px; }
.article-body .mesi-key .key-snoop  { color: #8a847a; font-size: 14px; vertical-align: -1px; letter-spacing: 1px; }
.article-body .mesi-key-sub {
  text-align: center;
  font-size: 10.5px;
  font-style: italic;
  color: #8a847a;
  margin: 0 auto 28px;
  max-width: 660px;
}
[data-theme="dark"] .article-body .mesi-key {
  color: #c8c2b6;
}
[data-theme="dark"] .article-body .mesi-key .key-rdhit  { color: #7aaa6a; }
[data-theme="dark"] .article-body .mesi-key .key-rdmiss { color: #6a9aba; }
[data-theme="dark"] .article-body .mesi-key .key-wrhit  { color: #d8b878; }
[data-theme="dark"] .article-body .mesi-key .key-wrmiss { color: #d8704a; }
[data-theme="dark"] .article-body .mesi-key .key-snoop  { color: #a09a8e; }
[data-theme="dark"] .article-body .mesi-key-sub {
  color: #6a6660;
}
.article-body .philosophy {
  background: #f4f2ea;
  border-radius: 10px;
  padding: 20px 24px;
  margin: 18px 0;
}
.article-body .genealogy {
  margin: 20px 0 28px;
  position: relative;
}
.article-body .genealogy::before {
  content: "";
  position: absolute;
  left: 54px;
  top: 10px;
  bottom: 10px;
  width: 2px;
  background: #e0ddd3;
}
.article-body .gen-row {
  display: flex;
  align-items: flex-start;
  margin-bottom: 18px;
  position: relative;
}
.article-body .gen-row:last-child {
  margin-bottom: 0;
}
.article-body .gen-year {
  flex: 0 0 46px;
  text-align: right;
  padding-right: 18px;
  padding-top: 3px;
  font-family: "Mundo Serif", Georgia, serif;
  font-size: 11.5px;
  font-weight: 700;
  color: #999;
  letter-spacing: 0.04em;
  white-space: nowrap;
}
.article-body .gen-row::before {
  content: "";
  position: absolute;
  left: 50px;
  top: 7px;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #b5b0a2;
  border: 2px solid #ffffff;
  box-sizing: content-box;
}
.article-body .gen-body {
  flex: 1 1 auto;
  padding-left: 22px;
  min-width: 0;
}
.article-body .gen-head {
  font-size: 14.5px;
  color: #1a1a1a;
  margin-bottom: 3px;
  line-height: 1.3;
}
.article-body .gen-head a {
  border-bottom-color: transparent;
}
.article-body .gen-head a:hover {
  border-bottom-color: #4a4a4a;
}
.article-body .gen-chip {
  color: #8a8a8a;
  font-size: 13px;
  font-weight: 400;
  font-style: normal;
  margin-left: 10px;
  font-family: "Mundo Serif", Georgia, serif;
  letter-spacing: 0.01em;
}
.article-body .gen-desc {
  font-size: 13.5px;
  color: #4a4a4a;
  line-height: 1.55;
}
.article-body .philosophy::before {
  content: "设计理念";
  display: block;
  font-size: 10.5px;
  font-weight: 700;
  color: #8b8676;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  margin-bottom: 10px;
}
.article-body .philosophy p:last-child {
  margin-bottom: 0;
}
.article-body section.architecture {
  display: block;
}
@media (min-width: 1680px){.article-body section.architecture:has(.genealogy-block) {
    display: grid;
    grid-template-columns: minmax(0, 720px) 380px;
    column-gap: 48px;
    align-items: start;
    /* extend beyond article-body's 720px column so the rail sits to the
       right of the page edge (rail 380 + gap 48 = 428 of overflow) */
    width: calc(100% + 428px);
  }.article-body section.architecture:has(.genealogy-block) > * {
    grid-column: 1;
    min-width: 0;
  }.article-body section.architecture > .genealogy-block {
    grid-column: 2;
    /* row 2 = the row containing the philosophy in column 1. the rail's
       natural top therefore sits at the same y as the philosophy's top in
       every section, regardless of philosophy length — giving consistent
       entry alignment across architectures. */
    grid-row: 2 / span 999;
    position: sticky;
    top: 24px;
    align-self: start;
    /* block fills the viewport vertically; content inside is flex-centred.
       this preserves the simple sticky logic (engage at viewport top, exit
       at section end) while displaying the genealogy content at viewport
       centre throughout. */
    height: calc(100vh - 48px);
    display: flex;
    flex-direction: column;
    justify-content: center;
    overflow-y: auto;
    padding-right: 8px;
    font-size: 0.92em;
    scrollbar-width: thin;
    scrollbar-color: transparent transparent;
  }.article-body section.architecture > .genealogy-block:hover {
    scrollbar-color: #ddd transparent;
  }.article-body section.architecture > .genealogy-block > h4 {
    margin-top: 0;
    margin-bottom: 14px;
    font-size: 10.5px;
    font-weight: 700;
    color: #8b8676;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    border-bottom: none;
    padding-bottom: 0;
  }.article-body section.architecture > .genealogy-block > .genealogy {
    margin: 0;
  }.article-body section.architecture > .genealogy-block .gen-row {
    margin-bottom: 10px;
  }.article-body section.architecture > .genealogy-block .gen-desc {
    line-height: 1.45;
  }.article-body section.architecture > .genealogy-block .gen-head {
    margin-bottom: 1px;
  }}
.article-body .definitions {
  display: flex;
  gap: 14px;
  margin: 18px 0;
  flex-wrap: wrap;
}
.article-body p:has(+ .definitions) {
  margin-bottom: 0;
}
.article-body .definition {
  flex: 1 1 240px;
  background: #f7f5ef;
  border: 1px dotted #bbb;
  border-radius: 8px;
  padding: 14px 18px 16px;
}
.article-body .definition-term {
  font-family: "Mundo Serif", Georgia, serif;
  font-weight: 700;
  font-style: italic;
  font-size: 14px;
  color: #1a1a1a;
  margin-bottom: 6px;
}
.article-body .definition-body {
  font-size: 13px;
  line-height: 1.55;
  color: #4a4a4a;
}
.article-body .definitions:has(.analogy-key) {
  margin: 18px 0 36px;
}
.article-body .definition.analogy-key {
  flex: 1 1 100%;
}
.article-body .definition.analogy-key .definition-term {
  margin-bottom: 14px;
}
.article-body .analogy-key table.analogy {
  width: auto;
  margin: 0;
  font-size: 13px;
  line-height: 1.5;
  border-collapse: collapse;
}
.article-body .analogy-key table.analogy th {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #b0b0b0;
  text-align: left;
  padding: 0 24px 6px 0;
  border-bottom: 1px solid #d4cfbd;
  white-space: nowrap;
}
.article-body .analogy-key table.analogy td {
  padding: 4px 24px 4px 0;
  border-bottom: none;
  vertical-align: top;
}
.article-body .analogy-key table.analogy td:first-child {
  font-weight: 600;
  color: #1a1a1a;
  white-space: nowrap;
}
.article-body .analogy-key table.analogy td:last-child {
  color: #4a4a4a;
}
.article-body .analogy-key table.analogy tbody tr:hover {
  background: transparent;
}
.article-body .analogy-key table.analogy tbody tr:first-child td {
  padding-top: 8px;
}
[data-theme="dark"] .article-body .analogy-key table.analogy th {
  color: #6a6660;
  border-bottom-color: #4a4640;
}
[data-theme="dark"] .article-body .analogy-key table.analogy td:first-child {
  color: #e8e4d8;
}
[data-theme="dark"] .article-body .analogy-key table.analogy td:last-child {
  color: #b0aa98;
}
.article-body ol {
  padding-left: 20px;
  margin-bottom: 20px;
}
.article-body ol li {
  margin-bottom: 4px;
}
.article-body ul {
  padding-left: 20px;
  margin-bottom: 20px;
}
.article-body ul li {
  margin-bottom: 8px;
}
.article-body table {
  width: 100%;
  border-collapse: collapse;
  margin: 20px 0 28px;
  font-size: 12.5px;
  line-height: 1.5;
}
.article-body table th,
.article-body table td {
  text-align: left;
  padding: 8px 10px;
  border-bottom: 1px solid #ececec;
  vertical-align: top;
}
.article-body table th {
  font-weight: 600;
  color: #1a1a1a;
  border-bottom: 1px solid #c8c8c8;
  white-space: nowrap;
}
.article-body table tbody tr:hover {
  background: #faf8f0;
}
[data-theme="dark"] .article-body table th,
[data-theme="dark"] .article-body table td {
  border-bottom-color: #2e2c27;
}
[data-theme="dark"] .article-body table th {
  color: #e8e4d8;
  border-bottom-color: #4a4640;
}
[data-theme="dark"] .article-body table tbody tr:hover {
  background: #1f1d18;
}
.article-body .comparison-wrap {
  max-width: 100%;
  overflow-x: auto;
  margin: 20px 0 40px;
  padding-bottom: 6px;
  scrollbar-width: thin;
  scrollbar-color: #cfc9b8 transparent;
}
.article-body .comparison-wrap::-webkit-scrollbar {
  height: 8px;
}
.article-body .comparison-wrap::-webkit-scrollbar-track {
  background: transparent;
}
.article-body .comparison-wrap::-webkit-scrollbar-thumb {
  background: #cfc9b8;
  border-radius: 4px;
}
.article-body .comparison-wrap::-webkit-scrollbar-thumb:hover {
  background: #b0aa98;
}
[data-theme="dark"] .article-body .comparison-wrap {
  scrollbar-color: #4a4640 transparent;
}
[data-theme="dark"] .article-body .comparison-wrap::-webkit-scrollbar-thumb {
  background: #4a4640;
}
[data-theme="dark"] .article-body .comparison-wrap::-webkit-scrollbar-thumb:hover {
  background: #6a6660;
}
.article-body .comparison-wrap > table.comparison-table {
  margin: 0;
}
.article-body .comparison-caption {
  margin: -20px 0 40px;
  color: #6c675d;
  font-size: 0.78rem;
  line-height: 1.5;
}
[data-theme="dark"] .article-body .comparison-caption {
  color: #aaa59a;
}
.article-body table.comparison-table tbody tr td {
  border-bottom: none;
}
.article-body table.comparison-table tbody:not(:last-child) tr:last-child td {
  border-bottom: 1px solid #c8c8c8;
}
.article-body table.comparison-table td.company {
  background: #efece4;
  vertical-align: middle;
  text-align: center;
  white-space: nowrap;
  width: 36px;
}
.article-body table.comparison-table td.company .company-logo {
  display: inline-block;
  width: 22px;
  height: 22px;
  border-radius: 4px;
  object-fit: contain;
  border-bottom: none;
  margin: 0;
}
.article-body table.comparison-table td.company .company-logo.is-tight {
  height: 14px;
}
[data-theme="dark"] .article-body table.comparison-table tbody:not(:last-child) tr:last-child td {
  border-bottom-color: #4a4640;
}
[data-theme="dark"] .article-body table.comparison-table td.company {
  background: #2a2823;
  color: #e8e4d8;
}
.article-body img {
  display: block;
  max-width: 100%;
  margin: 28px auto;
}
.article-body img.img-small {
  max-width: 460px;
}
[data-theme="dark"] .section-title,
[data-theme="dark"] .article-header .article-title,
[data-theme="dark"] .article-body h3 {
  color: #e8e4d8;
}
[data-theme="dark"] .divider,
[data-theme="dark"] .article-body hr {
  border-top-color: #2a2823;
}
[data-theme="dark"] .article-body code {
  background: #181612;
  color: #e0dccf;
}
[data-theme="dark"] .article-body pre {
  background: #14120e;
}
[data-theme="dark"] .article-body pre code {
  color: #e0dccf;
}
[data-theme="dark"] .article-body pre code.hljs { color: #e0dccf; background: none; }
[data-theme="dark"] .article-body .hljs-comment,
[data-theme="dark"] .article-body .hljs-quote { color: #6fa86f; font-style: italic; }
[data-theme="dark"] .article-body .hljs-keyword,
[data-theme="dark"] .article-body .hljs-selector-tag,
[data-theme="dark"] .article-body .hljs-literal,
[data-theme="dark"] .article-body .hljs-section,
[data-theme="dark"] .article-body .hljs-doctag,
[data-theme="dark"] .article-body .hljs-name { color: #b48ce0; }
[data-theme="dark"] .article-body .hljs-type,
[data-theme="dark"] .article-body .hljs-class .hljs-title,
[data-theme="dark"] .article-body .hljs-built_in { color: #6cc8a4; }
[data-theme="dark"] .article-body .hljs-title,
[data-theme="dark"] .article-body .hljs-title.function_,
[data-theme="dark"] .article-body .hljs-function .hljs-title { color: #6aa6e8; }
[data-theme="dark"] .article-body .hljs-string,
[data-theme="dark"] .article-body .hljs-symbol,
[data-theme="dark"] .article-body .hljs-bullet,
[data-theme="dark"] .article-body .hljs-link { color: #6cc8a4; }
[data-theme="dark"] .article-body .hljs-number,
[data-theme="dark"] .article-body .hljs-variable.constant_ { color: #e07a7a; }
[data-theme="dark"] .article-body .hljs-meta,
[data-theme="dark"] .article-body .hljs-meta .hljs-keyword,
[data-theme="dark"] .article-body .hljs-meta .hljs-string { color: #d6a060; }
[data-theme="dark"] .article-body .hljs-attr,
[data-theme="dark"] .article-body .hljs-attribute,
[data-theme="dark"] .article-body .hljs-variable,
[data-theme="dark"] .article-body .hljs-template-variable,
[data-theme="dark"] .article-body .hljs-params { color: #e0dccf; }
[data-theme="dark"] .article-body .hljs-operator,
[data-theme="dark"] .article-body .hljs-punctuation { color: #a09a8e; }
[data-theme="dark"] .article-body h4 {
  color: #a09a8e;
}
[data-theme="dark"] .article-body h4 a[data-popup]:hover {
  color: #e0dccf;
}
[data-theme="dark"] .article-body h5 {
  color: #c8c2b6;
}
[data-theme="dark"] .article-body .philosophy {
  background: #14120e;
}
[data-theme="dark"] .article-body .philosophy::before {
  color: #a09a8e;
}
[data-theme="dark"] .article-body .definition {
  background: #14120e;
  border-color: #555048;
}
[data-theme="dark"] .article-body .definition-term {
  color: #e8e4d8;
}
[data-theme="dark"] .article-body .definition-body {
  color: #d0cabe;
}
[data-theme="dark"] .article-body .genealogy::before {
  background: #2e2c27;
}
[data-theme="dark"] .article-body .gen-row::before {
  background: #6a6660;
  border-color: #080706;
}
[data-theme="dark"] .article-body .gen-year {
  color: #a09a8e;
}
[data-theme="dark"] .article-body .gen-head {
  color: #e8e4d8;
}
[data-theme="dark"] .article-body .gen-chip {
  color: #a09a8e;
}
[data-theme="dark"] .article-body .gen-desc {
  color: #d0cabe;
}
[data-theme="dark"] .article-body .katex-html {
  background: #14120e;
}
@media (max-width: 700px){.article-body h3 {
    scroll-margin-top: 14px;
  }}
.article-body { max-width: 64rem; margin: 0 auto; line-height: 1.75; }
.article-body img:not(.company-logo):not(.heading-logo) { display: block; width: 100%; height: auto; margin: 24px auto 8px; }
.article-body p:has(> img) { margin-bottom: 20px; color: #666; font-size: 13px; line-height: 1.55; }
.translation-notice { max-width: 64rem; margin: 0 auto 28px; padding: 14px 18px; border-left: 3px solid #b88c4a; background: #f7f5ef; font-size: 14px; line-height: 1.65; }
@media (prefers-color-scheme: dark) {
  .translation-notice { background: rgba(255,255,255,.06); }
}
.dark .article-body h3,
.dark .article-body h4,
.dark .article-body h5,
.dark .article-body .gen-head,
.dark .article-body .definition-term,
.dark .article-body table th { color: #e8e4d8; }
.dark .article-body .gen-desc,
.dark .article-body .definition-body,
.dark .article-body table td { color: #d0cabe; }
.dark .article-body .philosophy,
.dark .article-body .definition { background: #292620; }
.dark .article-body .definition { border-color: #625d52; }
.dark .article-body code,
.dark .article-body pre,
.dark .article-body .katex-html { color: #e8e4d8; background: #14120e; }
.dark .article-body hr,
.dark .article-body table th,
.dark .article-body table td { border-color: #4a4640; }
.dark .article-body p:has(> img) { color: #b0aa98; }
.article-body em,
.article-body h3,
.article-body h4,
.article-body h5,
.article-body .definition-term,
.translation-notice em { font-style: normal !important; }

</style>

<div class="translation-notice">
本文翻译自 Jacob Peake 的 <a href="https://www.jacobpeake.com/ai-chip-architectures"><em>AI Chip Architectures</em></a>。原文作者：Jacob Peake；译者：Pengcheng。
</div>
<div class="article-body" data-cta="standard-machines" data-prerendered="true" data-src="/posts/ai-chip-architectures.md"><p>在 2018 <a data-popup="isca">计算机体系结构国际研讨会上</a>， <em><strong><a href="https://en.wikipedia.org/wiki/John_L._Hennessy">John Hennessy</a></strong></em> 和 <em><strong><a href="https://en.wikipedia.org/wiki/David_Patterson_(computer_scientist)">David Patterson</a></strong></em> 发表了他们的 <a data-popup="turing-award">Turing</a> 讲座： <em><strong><a href="https://dl.acm.org/doi/10.1145/3282307">“计算机架构的新黄金时代”</a></strong></em>。 </p>
<p>在 20 世纪 80 年代， <em><strong>Hennessy</strong></em> 和 <em><strong>Patterson</strong></em> 进行了他们的图灵奖获奖研究，<br/> 单线程 CPU 性能每年增长 52%。到 2018 年，随着 <em><strong><a href="https://en.wikipedia.org/wiki/Moore%27s_law">摩尔定律</a></strong></em> 和 <em><strong><a href="https://en.wikipedia.org/wiki/Dennard_scaling">登纳德缩放</a></strong></em>的结束，该比率为 3%。</p>
<p>于是，Domain-Specific Architecture（DSA）成为重要方向。Hennessy 和 Patterson 以 Google 已投入生产的 <strong><a href="https://en.wikipedia.org/wiki/Tensor_Processing_Unit">TPU v1</a></strong> 为例：它的神经网络推理吞吐量是 CPU 的 29 倍，能效则高出 80 倍。最后，他们预测：<strong>“未来十年将迎来计算机架构的寒武纪大爆发。”</strong></p>
<p>这一预测成真了。如今，已有数十种架构进入严肃开发阶段： <em><strong>GPU</strong></em>、 <em><strong>TPU</strong></em>、 <em><strong>LPU</strong></em>、 <em><strong>NPU</strong></em>、 <em><strong>DPU</strong></em>、 <em><strong>ASIC</strong></em>、 <em><strong>晶圆级引擎</strong></em>、 <em><strong>可重构数据流</strong></em>、 <em><strong>神经形态</strong></em>、 <em><strong>光子计算</strong></em>与 <em><strong>模拟计算</strong></em>。其中相当一部分聚焦于 AI 计算。</p>
<p>迄今为止已获得实际部署的架构： <em><strong>GPU</strong></em> （NVIDIA、AMD）、 <em><strong>脉动阵列加速器</strong></em> （TPU、Trainium）、 <em><strong>Cerebras 晶圆级引擎</strong></em>和 <em><strong>Groq LPU</strong></em>。</p>
<p><em><strong>NVIDIA</strong></em> 是明确的领先者； <em><strong>AMD</strong></em> 紧随其后，并分别获得 <a href="https://openai.com/index/openai-amd-strategic-partnership/">OpenAI</a> 和 <a href="https://www.amd.com/en/newsroom/press-releases/2026-2-24-amd-and-meta-announce-expanded-strategic-partnersh.html">Meta</a> 各 6 GW 的部署承诺。 <em><strong>TPU</strong></em> 用于训练 Gemini，并将 <a href="https://www.anthropic.com/news/expanding-our-use-of-google-cloud-tpus-and-services">以多达 100 万颗芯片为 Anthropic 提供服务</a>；Anthropic 还在 <a href="https://techcrunch.com/2026/03/22/an-exclusive-tour-of-amazons-trainium-lab-the-chip-thats-won-over-anthropic-openai-even-apple/">超过一百万颗 <em><strong>Trainium</strong></em> 芯片</a>上运行 Claude。 <em><strong>Cerebras</strong></em> <a href="https://openai.com/index/cerebras-partnership/">现已承载 OpenAI 推理服务</a>； <em><strong>Groq LPU</strong></em> 则通过一笔 <a href="https://www.datacenterdynamics.com/en/news/nvidia-builds-out-lpu-chip-team-following-20bn-groq-acquihire-announcement-rumored-for-gtc/">价值 200 亿美元的 acquihire 交易</a>并入 NVIDIA。</p>
<p>本文将系统梳理这些不同路线的 <em>设计理念</em>、 <em>芯片架构</em>、 <em>扩展方式（纵向扩展与横向扩展）</em>以及 <em>软件栈（如何对芯片编程）</em>。</p>
<hr/>
<h3 id="the-problem">问题</h3>
<p>AI 计算以 <em><strong>矩阵乘法</strong></em>为主。 Transformer 是一系列 matmuls： <em><strong>Q/K/V 投影</strong></em>, <em><strong>attention</strong></em>, <em><strong>输出投影</strong></em>, <em><strong>FFN</strong></em> - 与 element-wise ops 交错：归一化、激活、残差添加。 <a data-popup="training">训练</a> 前沿模型执行 <span class="katex notranslate" translate="no"><span class="katex-mathml"><math class="notranslate" translate="no" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msup><mn>10</mn><mn>25</mn></msup></mrow><annotation encoding="application/x-tex">10^{25}</annotation></semantics></math></span><span aria-hidden="true" class="katex-html"><span class="base"><span class="strut" style="height:0.8141em;"></span><span class="mord">1</span><span class="mord"><span class="mord">0</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height:0.8141em;"><span style="top:-3.063em;margin-right:0.05em;"><span class="pstrut" style="height:2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight">25</span></span></span></span></span></span></span></span></span></span></span></span> 乘法累加运算（matmuls 是一系列乘法累加）。</p>
<p>matmul shape 取决于 workload。训练阶段会让一批 sequence 依次通过所有 layer，再做 backpropagation 和 optimizer update；同一块 weight matrix 往往同时处理数千个 token。<strong><a data-popup="prefill">Prefill</a></strong> 是 inference 中处理完整 prompt 的阶段：模型在生成第一个 output token 之前，一次性计算整段输入。training 和 prefill 都能把大量 token 堆叠在同一块 weight matrix 上，因此主要计算是 arithmetic intensity 很高、通常 compute-bound 的 GEMM。<strong><a data-popup="decode">Decode</a></strong> 则是 autoregressive 的：必须生成 token N 后才能开始 token N+1，每一步的 matmul 因而更接近 GEMV。生成一个 token 需要读取整套 model weight，并在 attention 中读取 KV Cache，所以 decode 的 arithmetic intensity 比 prefill 低几个数量级。</p>
<p>推理系统通过 batching 把这些 GEMV 重新组织成 GEMM 来提高 arithmetic intensity： <em><strong><a data-popup="continuous-batching">continuous batching</a></strong></em> 堆叠许多用户的 <a data-popup="decode">decode</a> 步骤， <em><strong><a data-popup="speculative-decoding">speculative decoding</a></strong></em> 为每个请求生成 K 个 draft token，再一次性完成验证，并且 <em><strong><a data-popup="multi-token-prediction">multi-token prediction</a></strong></em> 把同一思路直接内置到模型中。这实现了 matmul 单元的更高利用率，并提高了 Ops/B。对于 continuous batching，每个用户的请求仍然读取自己的 <a data-popup="kv-cache">KV Cache</a>，因此长上下文 decode 从 weight-bandwidth-bound 转变为 KV-bandwidth-bound。</p>
<p>这里的架构问题是 <em><strong>足够快地把数据</strong></em> 送到 matmul 单元。这被称为 <em><strong><a data-popup="memory-wall">内存墙</a></strong></em>：计算量呈指数级增长，但内存带宽却没有。</p>
<p>每种架构都提出了不同的策略来赢得 data movement 竞争。理解一颗 AI chip 可以归结为四个问题： <em>数据 <strong>在哪里</strong>，它如何 <strong>移动</strong> 到计算单元， <strong>计算单元</strong> 的外观，以及芯片如何在 <strong>规模</strong></em>上相互通信。</p>
<hr/>
<h3 id="nvidia-gpu">NVIDIA GPU</h3>
<div class="philosophy">
<p>NVIDIA GPU 的核心定位是 <strong>massively parallel processor</strong>：一颗可编程芯片提供数千个 thread，由 host CPU 负责 orchestration，并通过 CUDA 暴露给 developer。每一代都会在 Streaming Multiprocessor 上加入新的 acceleration primitive，但保留相同的 programming model。正因如此，同一颗 GPU 既能训练 Transformer、执行 inference，也能用于 graphics rendering 和 scientific computing。</p>
</div>
<h4 id="genealogy">演进路线</h4>
<div class="genealogy">
<div class="gen-row"><div class="gen-year">2006</div><div class="gen-body"><div class="gen-head"><strong><em><a data-no-preview="" href="https://en.wikipedia.org/wiki/Tesla_(microarchitecture)">Tesla</a></em></strong><span class="gen-chip"><a data-popup="g80">G80</a></span></div><div class="gen-desc">首款支持 CUDA 的 GPU； <a data-popup="unified-shaders">统一着色器</a> 和 <a data-popup="simt-execution-model">SIMT 执行模型</a>。</div></div></div>
<div class="gen-row"><div class="gen-year">2010</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://www.nvidia.com/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf">Fermi</a></em></strong><span class="gen-chip"><a data-popup="gf100">GF100</a></span></div><div class="gen-desc">第一个真正的计算架构：统一的 L1/L2 缓存、双 <a data-popup="warp-schedulers">warp scheduler</a>，IEEE-754 <a data-popup="fp64">FP64</a>。</div></div></div>
<div class="gen-row"><div class="gen-year">2012</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/NVIDIA-Kepler-GK110-GK210-Architecture-Whitepaper.pdf">Kepler</a></em></strong><span class="gen-chip"><a data-popup="k20">K20</a>， <a data-popup="k40">K40</a></span></div><div class="gen-desc"><a data-popup="smx">SMX</a>、 <a data-popup="dynamic-parallelism">动态并行</a>、 <a data-popup="hyper-q">Hyper-Q</a>; GPU 可以启动自己的工作。</div></div></div>
<div class="gen-row"><div class="gen-year">2014</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://developer.nvidia.com/maxwell-compute-architecture">Maxwell</a></em></strong><span class="gen-chip"><a data-popup="m40">M40</a></span></div><div class="gen-desc">重新设计了 SM，约 2×Kepler 的每瓦性能。</div></div></div>
<div class="gen-row"><div class="gen-year">2016</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf">Pascal</a></em></strong><span class="gen-chip"><a data-popup="p100">P100</a></span></div><div class="gen-desc"><a data-popup="nvlink">NVLink</a> 1.0， <a data-popup="hbm2">HBM2</a>，原生 <a data-popup="fp16">FP16</a> 吞吐量；第一款专为深度学习而设计的 GPU。</div></div></div>
<div class="gen-row"><div class="gen-year">2017</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf">Volta</a></em></strong><span class="gen-chip"><a data-popup="v100">V100</a></span></div><div class="gen-desc"><strong>首款 <a data-popup="tensor-cores">Tensor Core</a></strong>； <a data-popup="independent-thread-scheduling">独立 thread 调度</a>。</div></div></div>
<div class="gen-row"><div class="gen-year">2018</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf">Turing</a></em></strong><span class="gen-chip"><a data-popup="t4">T4</a></span></div><div class="gen-desc">第二代 <a data-popup="tensor-cores">Tensor Core</a> ，带有 <a data-popup="int8">INT8</a>/<a data-popup="int4">INT4</a>;第一个 <a data-popup="rt-cores">RT Core</a>。</div></div></div>
<div class="gen-row"><div class="gen-year">2020</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf">Ampere</a></em></strong><span class="gen-chip"><a data-popup="a100">A100</a></span></div><div class="gen-desc">第三代 <a data-popup="tensor-cores">Tensor Core</a> 与 <a data-popup="tf32">TF32</a> 和 <a data-popup="structured-sparsity">structured sparsity</a>； <a data-popup="multi-instance-gpu">多实例 GPU</a> 分区。</div></div></div>
<div class="gen-row"><div class="gen-year">2022</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c">Hopper</a></em></strong><span class="gen-chip"><a data-popup="h100">H100</a>、 <a data-popup="h200">H200</a>、 <a data-popup="gh200">GH200</a></span></div><div class="gen-desc">第四代 <a data-popup="tensor-cores">Tensor Core</a>， <a data-popup="fp8">FP8</a>, <a data-popup="transformer-engine">Transformer 引擎</a>; <a data-popup="hbm3">HBM3</a>、 <a data-popup="tma">TMA</a>、thread-block cluster、异步 <a data-popup="wgmma"><code class="notranslate" translate="no">wgmma</code></a>。</div></div></div>
<div class="gen-row"><div class="gen-year">2024</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-architecture-technical-brief">Blackwell</a></em></strong><span class="gen-chip"><a data-popup="b100">B100</a>、 <a data-popup="b200">B200</a>、 <a data-popup="gb200">GB200</a></span></div><div class="gen-desc">第 5 代 <a data-popup="tensor-cores">Tensor Core</a> 与 <a data-popup="fp4">FP4</a>、 <a data-popup="tensor-memory">Tensor Memory (TMEM)</a>、 <a data-popup="two-die-chiplet-gpu">两芯片小芯片 GPU</a>、 <a data-popup="nvlink-5">NVLink 5</a>。</div></div></div>
<div class="gen-row"><div class="gen-year">2025</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://www.nvidia.com/en-us/data-center/gb300-nvl72/">Blackwell Ultra</a></em></strong><span class="gen-chip"><a data-popup="b300">B300</a>, <a data-popup="gb300">GB300</a></span></div><div class="gen-desc">周期中刷新：~1.5× <a data-popup="fp4">FP4</a> 吞吐量，288 GB <a data-popup="hbm3e">HBM3e</a>。针对长上下文推理进行了调整。</div></div></div>
<div class="gen-row"><div class="gen-year">2026</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference">Rubin</a></em></strong><span class="gen-chip"><a data-popup="rubin">Rubin</a>， <a data-popup="vr200">VR200</a>、 <a data-popup="rubin-cpx">Rubin CPX</a></span></div><div class="gen-desc"><a data-popup="hbm4">HBM4</a>、第三代 <a data-popup="transformer-engine">Transformer 引擎</a>、 <a data-popup="vera-cpu">Vera CPU</a> 配对，分解 <a data-popup="prefill">prefill</a> 通过 <a data-popup="rubin-cpx">Rubin CPX</a>。</div></div></div>
<div class="gen-row"><div class="gen-year">2027</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/">Rubin Ultra</a></em></strong><span class="gen-chip"><a data-popup="rubin-ultra">Rubin Ultra</a></span></div><div class="gen-desc">4 芯片 GPU 封装，每个封装 1 TB <a data-popup="hbm4">HBM4</a>e。部署在 600 kW NVL576 Kyber 机架中，每 GPU 速度为 100 PetaFLOPS <a data-popup="fp4">FP4</a> 。</div></div></div>
</div>
<h4 id="architecture">架构</h4>
<p>NVIDIA GPU 由大量面向吞吐的 <strong>Streaming Multiprocessor（SM）</strong> 组成，并配套多级内存层次和硬件调度器，让数千个 thread 保持 in flight。SM 的数量从 V100 的 80 个、A100 的 108 个、H100 的 132 个，增长到 B200 的 148 个、B300 的 160 个和 Rubin 的 224 个。每个 SM 包含四个 SM sub-partition；每个 sub-partition 都有独立的 warp scheduler、dispatch unit、16K×32-bit register file、CUDA Core lanes、SFU，以及访问 Tensor Core 的端口。四个 sub-partition 共享 L1/SMEM 和 TMA。thread 以 32 个为一组组成 warp，并按 SIMT 模式锁步执行；调度器在多个 resident warp 之间切换，以隐藏 memory stall 和 arithmetic stall。</p>
<p><img alt="Blackwell B200 单芯片布局图 — GigaThread 引擎在中间运行，将芯片分成左右两半；每一半都有自己的 L2 缓存带，两侧是 GPC 集群； HBM3e 堆栈通过内存控制器排列在外边缘。 NVLink 和一个小型 PCIe Gen 6 主机链路位于顶部；底部的 NV-HBI 桥是与完成封装的镜像第二芯片的接缝。" loading="lazy" src="images/nvidia-gpu-die.png"/></p>
<p><img alt="放大到一个流 Multi-chip — 四个子分区，每个分区都有自己的 warp scheduler、调度、register file 和 Tensor Memory，利用共享的 L1/SMEM 和下面的 TMA。" loading="lazy" src="images/nvidia-sm.png"/></p>
<h5 id="compute">计算</h5>
<p><em><strong><a data-popup="cuda-cores">CUDA Core</a></strong></em> 是原始的计算吞吐量，对于人工智能来说，它们仍然拥有除 matmul 以外的所有内容：激活、残差加法、归一化、地址算术。但是，Transformer 块的 matmul FLOP 数约为 99%，因此压倒性的计算吞吐量来自 <em><strong><a data-popup="tensor-cores">Tensor Core</a></strong></em>。</p>
<p>这些核心执行 <em><strong>融合矩阵乘法累加</strong></em> 在小型 matrix tile 上， <span class="katex notranslate" translate="no"><span class="katex-mathml"><math class="notranslate" translate="no" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>D</mi><mo>=</mo><mi>A</mi><mo>⋅</mo><mi>B</mi><mo>+</mo><mi>C</mi></mrow><annotation encoding="application/x-tex">D = A \cdot B + C</annotation></semantics></math></span><span aria-hidden="true" class="katex-html"><span class="base"><span class="strut" style="height:0.6833em;"></span><span class="mord mathnormal" style="margin-right:0.0278em;">D</span><span class="mspace" style="margin-right:0.2778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right:0.2778em;"></span></span><span class="base"><span class="strut" style="height:0.6833em;"></span><span class="mord mathnormal">A</span><span class="mspace" style="margin-right:0.2222em;"></span><span class="mbin">⋅</span><span class="mspace" style="margin-right:0.2222em;"></span></span><span class="base"><span class="strut" style="height:0.7667em;vertical-align:-0.0833em;"></span><span class="mord mathnormal" style="margin-right:0.0502em;">B</span><span class="mspace" style="margin-right:0.2222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right:0.2222em;"></span></span><span class="base"><span class="strut" style="height:0.6833em;"></span><span class="mord mathnormal" style="margin-right:0.0715em;">C</span></span></span></span> 完整的 matmul 被分解为输出 tile：为了生成一个输出 tile，内核遍历共享的内部维度 <span class="katex notranslate" translate="no"><span class="katex-mathml"><math class="notranslate" translate="no" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>K</mi></mrow><annotation encoding="application/x-tex">K</annotation></semantics></math></span><span aria-hidden="true" class="katex-html"><span class="base"><span class="strut" style="height:0.6833em;"></span><span class="mord mathnormal" style="margin-right:0.0715em;">K</span></span></span></span>，读取左矩阵 A 的行块 <span class="katex notranslate" translate="no"><span class="katex-mathml"><math class="notranslate" translate="no" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>A</mi></mrow><annotation encoding="application/x-tex">A</annotation></semantics></math></span><span aria-hidden="true" class="katex-html"><span class="base"><span class="strut" style="height:0.6833em;"></span><span class="mord mathnormal">A</span></span></span></span> ，并读取右矩阵 B 的列块 <span class="katex notranslate" translate="no"><span class="katex-mathml"><math class="notranslate" translate="no" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>B</mi></mrow><annotation encoding="application/x-tex">B</annotation></semantics></math></span><span aria-hidden="true" class="katex-html"><span class="base"><span class="strut" style="height:0.6833em;"></span><span class="mord mathnormal" style="margin-right:0.0502em;">B</span></span></span></span> ，并将每个部分乘积累加到当前 accumulator 中。 <span class="katex notranslate" translate="no"><span class="katex-mathml"><math class="notranslate" translate="no" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>C</mi></mrow><annotation encoding="application/x-tex">C</annotation></semantics></math></span><span aria-hidden="true" class="katex-html"><span class="base"><span class="strut" style="height:0.6833em;"></span><span class="mord mathnormal" style="margin-right:0.0715em;">C</span></span></span></span> 保存到目前为止的部分和， <span class="katex notranslate" translate="no"><span class="katex-mathml"><math class="notranslate" translate="no" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>D</mi></mrow><annotation encoding="application/x-tex">D</annotation></semantics></math></span><span aria-hidden="true" class="katex-html"><span class="base"><span class="strut" style="height:0.6833em;"></span><span class="mord mathnormal" style="margin-right:0.0278em;">D</span></span></span></span> 是带入下一步的更新值。内循环完成后， <span class="katex notranslate" translate="no"><span class="katex-mathml"><math class="notranslate" translate="no" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>D</mi></mrow><annotation encoding="application/x-tex">D</annotation></semantics></math></span><span aria-hidden="true" class="katex-html"><span class="base"><span class="strut" style="height:0.6833em;"></span><span class="mord mathnormal" style="margin-right:0.0278em;">D</span></span></span></span> 就是完整输出矩阵中的一个最终 tile；整个 matmul 由许多这样的 tile <a data-popup="mma">MMA</a>构建而成。</p>
<p>tile shape 记为 <strong>M × N × K</strong>， <span class="katex notranslate" translate="no"><span class="katex-mathml"><math class="notranslate" translate="no" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>M</mi><mo>×</mo><mi>N</mi></mrow><annotation encoding="application/x-tex">M \times N</annotation></semantics></math></span><span aria-hidden="true" class="katex-html"><span class="base"><span class="strut" style="height:0.7667em;vertical-align:-0.0833em;"></span><span class="mord mathnormal" style="margin-right:0.109em;">M</span><span class="mspace" style="margin-right:0.2222em;"></span><span class="mbin">×</span><span class="mspace" style="margin-right:0.2222em;"></span></span><span class="base"><span class="strut" style="height:0.6833em;"></span><span class="mord mathnormal" style="margin-right:0.109em;">N</span></span></span></span> 是输出 tile 大小， <span class="katex notranslate" translate="no"><span class="katex-mathml"><math class="notranslate" translate="no" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>K</mi></mrow><annotation encoding="application/x-tex">K</annotation></semantics></math></span><span aria-hidden="true" class="katex-html"><span class="base"><span class="strut" style="height:0.6833em;"></span><span class="mord mathnormal" style="margin-right:0.0715em;">K</span></span></span></span> 表示单条指令沿 reduction dimension 处理的长度； matmul 的 <span class="katex notranslate" translate="no"><span class="katex-mathml"><math class="notranslate" translate="no" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>K</mi></mrow><annotation encoding="application/x-tex">K</annotation></semantics></math></span><span aria-hidden="true" class="katex-html"><span class="base"><span class="strut" style="height:0.6833em;"></span><span class="mord mathnormal" style="margin-right:0.0715em;">K</span></span></span></span> 轴的其余部分由内核的内部循环遍历。accumulator 会在整个 K-loop 中持续复用：每个 MMA 的输出 <span class="katex notranslate" translate="no"><span class="katex-mathml"><math class="notranslate" translate="no" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>D</mi></mrow><annotation encoding="application/x-tex">D</annotation></semantics></math></span><span aria-hidden="true" class="katex-html"><span class="base"><span class="strut" style="height:0.6833em;"></span><span class="mord mathnormal" style="margin-right:0.0278em;">D</span></span></span></span> 成为下一个 MMA 的输入 <span class="katex notranslate" translate="no"><span class="katex-mathml"><math class="notranslate" translate="no" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>C</mi></mrow><annotation encoding="application/x-tex">C</annotation></semantics></math></span><span aria-hidden="true" class="katex-html"><span class="base"><span class="strut" style="height:0.6833em;"></span><span class="mord mathnormal" style="margin-right:0.0715em;">C</span></span></span></span>，因此实际执行的是原地累加 <span class="katex notranslate" translate="no"><span class="katex-mathml"><math class="notranslate" translate="no" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>C</mi><mo>←</mo><mi>A</mi><mo>⋅</mo><mi>B</mi><mo>+</mo><mi>C</mi></mrow><annotation encoding="application/x-tex">C \leftarrow A \cdot B + C</annotation></semantics></math></span><span aria-hidden="true" class="katex-html"><span class="base"><span class="strut" style="height:0.6833em;"></span><span class="mord mathnormal" style="margin-right:0.0715em;">C</span><span class="mspace" style="margin-right:0.2778em;"></span><span class="mrel">←</span><span class="mspace" style="margin-right:0.2778em;"></span></span><span class="base"><span class="strut" style="height:0.6833em;"></span><span class="mord mathnormal">A</span><span class="mspace" style="margin-right:0.2222em;"></span><span class="mbin">⋅</span><span class="mspace" style="margin-right:0.2222em;"></span></span><span class="base"><span class="strut" style="height:0.7667em;vertical-align:-0.0833em;"></span><span class="mord mathnormal" style="margin-right:0.0502em;">B</span><span class="mspace" style="margin-right:0.2222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right:0.2222em;"></span></span><span class="base"><span class="strut" style="height:0.6833em;"></span><span class="mord mathnormal" style="margin-right:0.0715em;">C</span></span></span></span> 就位：连续的指令将其部分乘积折叠到同一存储中，直到 K 轴遍历完成。</p>
<p><a data-popup="v100">V100</a> 的第一代 Tensor Core 每个 SM 有 8 个，执行 warp-level 16×16×16 FP16 MMA。A100 的第三代 Tensor Core 加入 TF32、BF16、FP64 matmul 和 2:4 structured sparsity。H100 的第四代 Tensor Core 支持原生 FP8，并把 MMA 的协作范围从单个 warp 扩展到 128-thread warp group：<code class="notranslate" translate="no">wgmma</code> 以 64×256×16 tile shape 异步执行，发射它的 warp group 可以同时加载下一块 tile。B200 的第五代 Tensor Core 进一步支持 256×256×16 two-SM MMA、原生 FP4，并为每个 SM 增加 256 KB Tensor Memory（TMEM）来保存 accumulator tile，减轻 register pressure。Rubin 的第六代 Tensor Core 增加原生 FP6，并配合第三代 Transformer Engine 在硬件中完成自适应 NVFP4 micro-block scaling，使 quantization metadata 留在 Tensor Core datapath 上。</p>
<p>在所有六代中保持不变的是，matmul 位于 <em><strong>thread/warp 层次结构中</strong></em>，但是 <em>issue</em> 的 thread 数量已经减少，并且 issue 过程也逐渐与执行解耦。 <a data-popup="v100">Volta</a>的 <code class="notranslate" translate="no">mma.sync</code> 是 warp-<a data-popup="collective-instruction">collective</a> 和同步： <a data-popup="warp">warp 中的所有 32 个 thread</a> 一起执行，每个 lane 保存 A、B、 <strong>和 accumulator D</strong>的 register fragment，并且整个 warp 会阻塞到指令完成。 <a data-popup="h100">Hopper</a>的 <a data-popup="wgmma"><code class="notranslate" translate="no">wgmma.mma_async</code></a> 把 issuer 扩展为 <a data-popup="warp-group">warp-group</a> 128 个 thread，将 B 移动到 <a data-popup="shared-memory-descriptor">shared-memory descriptor</a> （A 变为可选：register <em>或</em> descriptor，内核选择），以及 <strong>立即返回</strong>：matmul 在后台运行，同时 warp-group 对下一个 tile 进行排队，并通过 <code class="notranslate" translate="no">wgmma.commit_group</code> / <code class="notranslate" translate="no">wgmma.wait_group</code>跟踪完成情况。</p>
<p>Blackwell 的 <code class="notranslate" translate="no">tcgen05.mma</code> 完成了这一迁移：A、B 都可以由 shared-memory descriptor 提供，accumulator D 则写入 TMEM，不再占用 per-thread register fragment。由于 operand 和 accumulator 都脱离了 lane，一条 MMA 可以由单个 thread 发射并立即返回；consumer warp 通过 <code class="notranslate" translate="no">mbarrier</code> 等待完成。CTA-pair 版本把同样的机制扩展到两个 SM：每个 SM 各有一个 thread 协同 issue MMA，共享 operand，并通过 cluster-level barrier 保持同步。</p>
<p>同时发出 thread 上的 matmul 变得更大、更轻：开始为 32 lanes 锁步运行的指令现在更接近于单个 <a data-popup="descriptor-driven-command">描述符驱动命令</a>，仍从 warp model 内 issue，但不再由其执行。</p>
<p>这种解耦使得 Transformer attention kernel 在 GPU 上高效运行。在 matmul 运行时，warp 可以运行 softmax、应用掩模或预加载下一个 tile； matmul 和周围的 element-wise 工作的重叠是每个现代 attention kernel 的结构（<a data-popup="flashattention-versions">FlashAttention-3</a>，FA4），并且它取决于不阻塞 warp 的矩阵指令。</p>
<h5 id="memory">内存</h5>
<p>片上层次结构是 <em><strong>各个级别的硬件管理缓存，软件提示位于顶层</strong></em>。片外为 <em><strong><a data-popup="hbm">HBM</a></strong></em>：32 GB <a data-popup="hbm2">HBM2</a> V100，80 GB <a data-popup="hbm3">HBM3</a> 在 H100 上，192 GB <a data-popup="hbm3e">HBM3e</a> 在 B200 上，288 GB B300，288 GB <a data-popup="hbm4">HBM4</a> ，Rubin。芯片级 <em><strong>L2 缓存</strong></em> 位于 HBM 和 SM 之间：V100 上为 6 MB，A100 上为 40 MB，H100 上为 50 MB，B200 上为 60 MB（分为两个 30 MB 存储体） <a data-popup="two-die-chiplet-gpu">双芯片封装</a>，具有位置感知 <a data-popup="l2-residency-controls">驻留控制</a> ，以便热块可以固定到附近的芯片上）。在每个 SM 内部，256 KB 的统一 <em><strong><a data-popup="l1-shared-memory">L1/SMEM</a></strong></em> 在 kernel launch 时在硬件管理的 L1 和程序员控制的 scratchpad 之间进行分区。每个 SM 的 register file 大约为 256 KB，在分区上分为四个方向。</p>
<p>Blackwell 添加了第五层： <em><strong><a data-popup="tensor-memory">TMEM</a></strong></em>，每个 SM 256 KB 专用于 <a data-popup="mma">MMA</a> accumulator，仅由 Tensor Core 寻址，将 operand 驻留压力从通用 register file 中拉出。</p>
<p>memory movement 也在逐代摆脱 warp。Ampere 之前，global memory 到 shared memory 的 tile load 需要 thread 先把数据读入 register，再写入 SMEM，整个 warp 在 load 完成前都会阻塞。Ampere 的 <code class="notranslate" translate="no">cp.async</code> 允许每个 thread 发起 HBM→SMEM async copy，并绕过 register。Hopper 又把这件事交给 TMA：单个 thread 提交包含 base address、stride 和 swizzle 的 multidimensional tile descriptor，DMA engine 自动完成 address generation，把数据写入 SMEM，再通过 <code class="notranslate" translate="no">mbarrier</code> 通知 consumer。TMA 还支持 cluster multicast，让一次 HBM read 同时送达 thread-block cluster 中的多个 SM。Blackwell 进一步支持直接向 TMEM 搬运数据。整体趋势很清楚：每一代都让 warp 少参与一些 data movement 和 address calculation。</p>
<h5 id="warp-specialisation">Warp 专用化</h5>
<p>Hopper 之后常见的编程方式是 <strong><a data-popup="warp-specialisation">Warp Specialization</a></strong>：同一个 thread block 内，一组 warp 作为 producer，连续发出 TMA load；另一组 warp 作为 consumer，在 tile 到达后执行 <code class="notranslate" translate="no">wgmma</code>。二者不再依赖整个 block 的 <code class="notranslate" translate="no">__syncthreads()</code>，而是通过 <code class="notranslate" translate="no">mbarrier</code> 和 TMA completion barrier 做细粒度握手。FlashAttention-3、CUTLASS ping-pong GEMM 和 Blackwell FA4 都采用类似结构：TMA producer pipeline 经 SMEM/TMEM 向 wgmma consumer pipeline 供数，thread-block cluster 再把多个 SM 组织成一个协作单元。</p>
<h5 id="numerics">数值格式</h5>
<p><a data-popup="fp32">FP32</a> 是历史默认值； Volta 带来了 <em><strong><a data-popup="fp16">FP16</a></strong></em> FP32 累积和 <a data-popup="loss-scaling">loss scaling</a> 使其可训练的技巧；添加了 Ampere <em><strong><a data-popup="tf32">TF32</a></strong></em> （FP32 范围、FP16 尾数、FP32 matmul 的插入）、 <em><strong><a data-popup="bf16">BF16</a></strong></em>和 2:4 <em><strong><a data-popup="structured-sparsity">structured sparsity</a></strong></em> ，使修剪权重的有效吞吐量加倍。 Hopper 引入原生 <em><strong><a data-popup="fp8">FP8</a></strong></em> 在 <a data-popup="e4m3">E4M3</a> 和 <a data-popup="e5m2">E5M2 中</a>，与 <em><strong><a data-popup="transformer-engine">Transformer Engine</a></strong></em> 配对，可逐层自动缩放激活以将其保持在 FP8 动态范围内。 Blackwell 通过 <em><strong><a data-popup="fp4">FP4</a></strong></em> 再次将精度减半，并推出了 <em><strong><a data-popup="microscaling-formats">microscaling MX 格式</a></strong></em> （恢复最多的块级共享指数） FP4 的准确性损失），以及将自动缩放管道重新定位到 FP4 的第二代 Transformer 引擎。 Rubin 的第三代 Transformer 引擎添加了 <em><strong><a data-popup="nvfp4">NVFP4</a></strong></em> （NVIDIA 的强化 FP4 变体）和原生 <em><strong>FP6</strong></em> 具有更激进的稀疏性。芯片布局本身现在已成为数字故事的一部分：B100/B200/B300 是 <em><strong><a data-popup="two-die-chiplet-gpu">两个十字线限制芯片</a></strong></em> 以约 10 TB/s 的速度缝合 <em><strong><a data-popup="nv-hbi">NV-HBI</a></strong></em> 链接并作为一个逻辑 GPU 呈现给软件，封装上有 8 个 HBM 堆栈； Rubin 将小芯片配方扩展到具有 8 个 HBM4 堆栈的约 336 B 晶体管的双芯片。每一代通过将位数减少一半并通过更细粒度的缩放方案恢复精度，以及越来越多地通过将更多的硅粘合到封装中来购买大约 2 倍的每瓦吞吐量。</p>
<h5 id="bets">设计取舍</h5>
<ul>
<li><em><strong>可编程性。</strong></em> 工作负载是一个移动目标（注意力变体、新颖的模型架构），因此请保持每个模块的可编程性并让开发人员编写 <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/">CUDA</a>。连专业单位都暴露了 <em>通过</em> 该模型而不是固定功能块。</li>
<li><em><strong>通过大规模多 thread 隐藏延迟。</strong></em> 延迟是不可预测的且依赖于数据，因此不要使用静态计划而是使用大量 thread 来隐藏它过度使用，每个 SM 最多 64 个常驻 warp，硬件 <a data-popup="warp-schedulers">warp 调度程序</a> 每个周期都会选择一个就绪的 warp。</li>
<li><em><strong>warp 包裹的 Matmul。</strong></em> 矩阵单元具有压倒性的计算吞吐量，但它必须存在于其他所有东西使用的相同 warp/thread 抽象后面，因此将其包裹在 <code class="notranslate" translate="no">mma.sync</code> → <code class="notranslate" translate="no">wgmma</code> → <code class="notranslate" translate="no">tcgen05.mma</code> - 而不是将其公开为固定功能管道。这使得单个 core 能够一次性融合 matmul、softmax 和 element-wise 运算。</li>
<li><em><strong>异步内存层次结构。</strong></em> 使内存层次结构 <a data-popup="explicit-hierarchy">显式</a> 和 <a data-popup="programmer-managed">程序员管理</a> 而不是 <a data-popup="implicit-hierarchy">隐式</a> 和 <a data-popup="compiler-scheduled">编译器调度</a>。保留 <a data-popup="l2-cache">二级缓存</a>，但公开 <a data-popup="l1-shared-memory">SMEM</a> 和 <a data-popup="tensor-memory">TMEM</a> 作为命名 scratchpad，并在顶部分层异步机制： <a data-popup="tma">TMA</a> 用于批量复制， <a data-popup="tensor-memory">TMEM</a> 用于 matmul accumulator， <a data-popup="mbarrier"><code class="notranslate" translate="no">mbarrier</code></a> 用于生产者/消费者握手。层次结构位于可编程内核内 <a data-popup="software-pipelined">软件流水线</a> ，而不是由编译器针对已知延迟 scratchpad 进行静态调度。</li>
<li><em><strong>摊销 SIMT 税。</strong></em> 用于 warp 调度程序、register file 或一致性缓存的每个晶体管都是未用于 <a data-popup="mac">MAC</a>的晶体管；接受税收，并以两种方式支付：Tensor Core 现在足够大，SIMT 机器可以在更大的 MAC 数量上摊销，像 TMEM 这样的单元会牺牲一些通用灵活性来换取 MAC 密度。</li>
</ul>
<h4 id="scaling">扩展</h4>
<p>有两种缩放机制： <em><strong><a data-popup="scale-up">纵向扩展</a></strong></em> 和 <em><strong><a data-popup="scale-out">向外扩展</a></strong></em>。</p>
<div class="definitions">
<div class="definition"><div class="definition-term">纵向扩展</div><div class="definition-body">将多个 GPU 绑定到一个一致的内存域中。任何 GPU 都可以直接通过 <a data-popup="hbm">NVLink</a> 以纳秒延迟加载或存储任何其他 GPU 的 <a data-popup="nvlink">HBM</a> ：一个地址空间，无需显式传输。</div></div>
<div class="definition"><div class="definition-term">横向扩展</div><div class="definition-body">在 <a data-popup="rack">机架上将这些域联网</a> 和 <a data-popup="cluster">集群</a> 级别。数据通过显式 <a data-popup="rdma">RDMA</a> 以微秒延迟进行传输：独立的地址空间，但每个 <a data-popup="cluster">集群有数万个芯片</a>。</div></div>
</div>
<p>人工智能基础设施同时使用：带宽需求 <a data-popup="collective">collective</a> （<a data-popup="tensor-parallelism">tensor parallel</a>、 <a data-popup="moe-expert-routing">MoE 专家路由</a>）留在扩展域内； <a data-popup="data-parallelism">数据并行</a> 和 <a data-popup="pipeline-parallelism">管道并行</a> 跨横向扩展结构。</p>
<h5 id="scale-up">纵向扩展</h5>
<p>扩展堆栈为 <em><strong><a data-popup="nvlink">NVLink</a></strong></em> 加上 <em><strong><a data-popup="nvswitch">NVSwitch</a></strong></em>。 <a data-popup="nvlink">NVLink</a> 实现 <em><strong>缓存一致性结构</strong></em> GPU 之间，因此一个 GPU 上的加载或存储可以通过硬件处理地址转换和一致性来针对另一个 GPU 的 <a data-popup="hbm">HBM</a> 。但 <a data-popup="nvlink">NVLink</a> 本身是点对点的：一条链路恰好连接两个芯片。 <a data-popup="nvswitch">NVSwitch</a> 是专用的 <em><strong><a data-popup="crossbar">crossbar</a></strong></em> 每个 GPU 连接的芯片，路由流量，以便每个 GPU 可以同时以完整 <a data-popup="nvlink">NVLink</a> 带宽进行通信， <a data-popup="non-blocking">非阻塞</a> 和 <a data-popup="all-to-all">all-to-all</a>。 </p>
<p>他们共同定义了 <em><strong><a data-popup="hgx">HGX</a></strong></em> 8-GPU 基板，配对八个 <a data-popup="h100">H100</a> <a data-popup="sxm">SXM</a> 模块，带有 <a data-popup="x86">x86</a> 主机（<a data-popup="epyc">AMD EPYC</a> 或 <a data-popup="xeon">Intel Xeon</a>）超过 <a data-popup="pcie-gen5">PCIe Gen5</a>。 Hopper 还提供了 <a data-popup="grace">Grace</a>配对形式： <em><strong><a data-popup="gh200">GH200 Grace Hopper Superchip</a></strong></em> 通过 <a data-popup="grace">Grace</a> ARM CPU 到一个 <a data-popup="h100">H100</a> 超过 <em><strong><a data-popup="nvlink-c2c">NVLink-C2C</a></strong></em> ，速度为 900 GB/s，消除了 <a data-popup="pcie">PCIe</a> 主机设备跃点。模块可扩展为 <em><strong><a data-popup="gh200">GH200</a> NVL2</strong></em> 对和机架级 <em><strong><a data-popup="gh200">GH200</a> NVL32</strong></em>。 Blackwell 将配对设置为默认值。 <em><strong><a data-popup="gb200">GB200</a></strong></em> 模块将一台 <a data-popup="grace">Grace</a> 与两台 B200 融合在一起 <a data-popup="nvlink-c2c">NVLink-C2C</a>和 <em><strong>NVL72</strong></em> 将其中的 36 个连接成单个液冷扩展域中： 72 个 GPU、36 个 <a data-popup="grace">Grace</a> CPU、13.5 TB <a data-popup="hbm">HBM</a> 和 17 TB <a data-popup="lpddr5x">LPDDR5X</a> 作为一个平坦、一致的地址空间。 Rubin 将其分为两部分。 <em><strong>NVL144</strong></em> 将于 2026 年作为 Rubin 一代的更新产品在同一个 <a data-popup="oberon">Oberon 中发货</a>级机架：72 个 Rubin 封装，根据 NVIDIA 新的芯片计数约定标记为 144 个 GPU，具有 <a data-popup="hbm4">HBM4</a> 和 NVLink 6 将每个封装带宽加倍。实际的机架规模跳跃是 2027 年的 Rubin Ultra： <em><strong>NVL576</strong></em> 将 144 个四芯片 Rubin Ultra 封装装入新的 <em><strong>Kyber</strong></em> 用于 576 GPU 芯片的机箱位于一个相干域中。</p>
<p><img alt="NVL72 — 72 个 Blackwell GPU 位于一排 NVSwitch ASIC 下方，形成一个无阻塞 crossbar，因此任何 GPU 都可以在完整的 NVLink 带宽下处理任何其他 GPU 的 HBM。整个结构在无源铜背板上运行：与光纤等效器件相比，约 5,184 条电缆盲插、约 130 TB/s 的全对全带宽、约 20 kW 的收发器功率节省。" loading="lazy" src="images/nvidia-scale-up.png"/></p>
<p>该密度由 <em><strong><a data-popup="passive-copper-backplane">无源铜</a></strong></em>保持在一起。 NVL72 的 NVLink 结构通过背板盲插运行超过 5,184 条电缆（每个机架约 2 英里的布线，没有 <a data-popup="in-cable-retimer">in-cable retimer</a>， <a data-popup="serdes">SerDes</a> 运行在 GPU 和 <a data-popup="switch-asic">交换机 ASIC</a> 上），承载约 130 TB/s 的 <a data-popup="all-to-all">所有</a> 跨 72 个 GPU 的带宽。 NVIDIA 估计，与每个链路上都需要 <a data-popup="pluggable-transceiver">可插拔收发器</a> 的光学等效方案相比，选择铜缆可以为每个机架节省大约 20 kW 的电量。铜使得 <em>机架作为一个 GPU</em> 经济实用：在低于 2 米的运行中，它仍然在功耗、成本和信号完整性方面胜出。除此之外，这些位必须安装在玻璃上。</p>
<p>NVL144 留在 Oberon 内部，铜继续工作，因为封装数量 (72) 与 NVL72 相同；布线不必加长，只需在第 6 代 SerDes 上传输速度更快即可。 <a data-popup="rubin">Rubin Ultra</a>的 NVL576 通过重塑机架来保持相同的铜线：新的 <em><strong>Kyber</strong></em> 外形尺寸大约是 Oberon 高度的两倍，并将所有 576 个 GPU 芯片封装到一个外壳中，其尺寸经过专门设计，因此即使在 144 个四芯片封装和数万根电缆的情况下，每个 NVLink 路径也能保持在无源铜缆范围内。</p>
<h5 id="scale-out">横向扩展</h5>
<p>横向扩展堆栈来自他们对 <a data-popup="mellanox">Mellanox</a>的收购。与 <a data-popup="nvlink">NVLink</a>不同，横向扩展结构 <em><strong>不连贯</strong></em>：节点保留单独的地址空间，数据仅通过显式交叉 <em><strong><a data-popup="rdma">RDMA</a></strong></em> 由软件发起，通常封装在 <em><strong><a data-popup="nccl">NCCL</a></strong></em> 集合中，例如 <a data-popup="all-reduce">all-reduce</a> 或 <a data-popup="all-to-all">all-to-all</a>。参考 <a data-popup="cluster">集群</a> 是 <em><strong>DGX SuperPOD</strong></em>：八个 NVL72 机架缝合在一起 <a data-popup="quantum-x800">Quantum-X800</a> <a data-popup="infiniband">InfiniBand</a> 在单个调度程序下产生 576 个 Blackwell GPU，训练集群通过平铺 SuperPOD 进一步扩展。 2026 年的 Rubin SuperPOD 与 NVL144 保持相同的 8 机架模式（每个 SuperPOD 产生 1,152 个 GPU，而不是 576 个）。 Rubin Ultra 将于 2027 年将这一方案扩大一个数量级：Kyber 机架每个包含 576 个 GPU 芯片，通过 <a data-popup="quantum-x-photonics">Quantum-X Photonics</a> <a data-popup="cpo">CPO</a>缝合在一起，将数千个一个调度程序下的 GPU。</p>
<p><img alt="DGX SuperPOD — 8 个 NVL72 机架（总共 576 个 GPU）位于 Quantum-X800 InfiniBand 主干下方。每 GPU 横向扩展为 800 Gbps 的 ConnectX-8 NIC；机架间跳跃跨越 OSFP-RHS 可插拔光收发器，支付微秒级延迟，而不是上述机架内 NVLink 结构的纳秒级延迟。" loading="lazy" src="images/nvidia-scale-out.png"/></p>
<p>每个 GPU 在该结构中都有自己的 <a data-popup="connectx-nics">ConnectX</a> <a data-popup="nic">NIC</a> 。 Blackwell 节点以每个 GPU 800 Gbps 的速度运行 ConnectX-8，比每个 GPU <a data-popup="nvlink">NVLink</a>的带宽低一个数量级，并且延迟从纳秒攀升至微秒。 Rubin 迁移到 ConnectX-9，每 GPU 速度为 1.6 Tbps，随着每机架纵向扩展域从 72 个 GPU 增长到 576 个 GPU，每 GPU 横向扩展带宽加倍。每个 NIC 旁边都有一个 <a data-popup="bluefield-dpus">BlueField DPU</a>，添加 ARM 内核和加速器以减轻主机 CPU 的存储、网络和安全负担。对于更喜欢以太网而不是 <a data-popup="infiniband">InfiniBand</a>的客户， <em><strong><a data-popup="spectrum-x">Spectrum-X</a></strong></em> 是专为人工智能而优化的无损以太网替代方案流量。</p>
<p>从铜缆到玻璃的交叉发生在机架边界。 NVL72 内部的脊柱是铜的；一旦链路必须以 800 Gbps 的速度跨机架，它就是 <em><strong>光纤</strong></em>。无源铜质 DAC 在 200 G/lane 时的最高长度约为 1.5–2 米，远低于跨机架范围，因此今天的 SuperPOD 主干采用 <em><strong><a data-popup="osfp">OSFP-RHS</a></strong></em> <a data-popup="pluggable-transceiver">可插拔收发器</a>，每个模块都带有自己的激光器、调制器、光电探测器和 DSP。从光学角度来看，分散到数千个 GPU 的 SuperPOD 主干相当于数万个 <a data-popup="pluggable-transceiver">可插拔</a> 仅在收发器激光器上就消耗了数十千瓦的功率。</p>
<p>到了 Rubin，光学器件被直接集成进 <a data-popup="switch-asic">交换机 ASIC</a>。<em><strong><a data-popup="quantum-x-photonics">Quantum-X Photonics</a></strong></em>（<a data-popup="infiniband">InfiniBand</a>）和 <em><strong><a data-popup="spectrum-x-photonics">Spectrum-X Photonics</a></strong></em>（<a data-popup="spectrum-x">以太网</a>）不再使用可插拔光模块，而是采用 <em><strong><a data-popup="cpo">共封装光学（CPO）</a></strong></em>：通过 TSMC COUPE 工艺，把激光器、调制器和光电探测器与交换芯片封装在一起。NVIDIA 表示，与功能相当的 OSFP 可插拔方案相比，这种设计可将激光器数量减少约 4 倍，并将链路功耗降低约 3.5 倍。此前，NVIDIA 已通过 Chiplet 封装把多个 GPU 裸片和 HBM 集成在同一封装内；如今，相同的集成思路进一步延伸到网络层，把计算、内存和光子器件集成到同一系统中。</p>
<p><em><strong><a data-popup="nvlink">NVLink</a> Fusion</strong></em> 最近开放了扩展结构本身：第三方 CPU 和 <a data-popup="xpus">XPU</a> 现在可以加入 <a data-popup="nvlink">NVLink</a> 域，让超大规模厂商可以围绕 NVIDIA 互连构建半定制机架，而无需从头开始设计自己的相干结构。</p>
<h4 id="software">软件栈</h4>
<p><strong><a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/">CUDA</a></strong> 是 GPU 的 natural programming model：开发者编写 kernel，由大量 thread 以 block 和 warp 的层次执行，并显式决定 data sharing、synchronization 和 work partition。这个 abstraction 十八年来基本稳定，所以从 2007 年开始编写的 CUDA kernel，今天仍能在 Blackwell 上编译运行。</p>
<p>这种连续性既是护城河也是约束。每一代都会引入新硬件（<a data-popup="tensor-cores">Tensor Core</a>, <a data-popup="tma">TMA</a>, <a data-popup="tensor-memory">TMEM</a>）到相同的 kernel-and-warps 模型上，在 <em><strong><a href="https://docs.nvidia.com/cuda/parallel-thread-execution/">PTX</a></strong></em> 和 <em><strong><a href="https://docs.nvidia.com/cuda/cuda-binary-utilities/">SASS</a></strong></em>中作为内在函数公开： <code class="notranslate" translate="no">mma.sync</code>、 <code class="notranslate" translate="no">wgmma.mma_async</code>等。 NVIDIA 无法从根本上重新考虑 SM，因为太多代码依赖于它；作为回报，对 CUDA 软件的每一项投资都会在几代人之间复合。</p>
<p>PTX 之上是一个经过二十年构建的堆栈。 <em><strong><a href="https://docs.nvidia.com/cuda/cublas/">cuBLAS</a></strong></em> 和 <em><strong><a href="https://developer.nvidia.com/cudnn">cuDNN</a></strong></em> 用于数学和 DNN 基元； <em><strong><a href="https://github.com/NVIDIA/cutlass">CUTLASS</a></strong></em>，编码了数十年 GEMM 在模板化 C++ 方面的专业知识； <em><strong><a href="https://github.com/NVIDIA/TensorRT-LLM">TensorRT-LLM</a></strong></em> 用于分页注意力、运行中批处理和 speculative decoding；通过 <em><strong><a href="https://pytorch.org/">PyTorch</a></strong></em>进行框架绑定， <em><strong><a href="https://triton-lang.org/">Triton</a></strong></em>和 <em><strong><a href="https://github.com/jax-ml/jax">JAX</a></strong></em>。</p>
<p><em><strong><a href="https://arxiv.org/abs/2205.14135">FlashAttention</a></strong></em>是现代人工智能中最重要的算法重写之一，它会集中注意力以避免实现 <span class="katex notranslate" translate="no"><span class="katex-mathml"><math class="notranslate" translate="no" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>O</mi><mo stretchy="false">(</mo><msup><mi>N</mi><mn>2</mn></msup><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">O(N^2)</annotation></semantics></math></span><span aria-hidden="true" class="katex-html"><span class="base"><span class="strut" style="height:1.0641em;vertical-align:-0.25em;"></span><span class="mord mathnormal" style="margin-right:0.0278em;">O</span><span class="mopen">(</span><span class="mord"><span class="mord mathnormal" style="margin-right:0.109em;">N</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height:0.8141em;"><span style="top:-3.063em;margin-right:0.05em;"><span class="pstrut" style="height:2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">2</span></span></span></span></span></span></span></span><span class="mclose">)</span></span></span></span> 矩阵。它的四代产品（FA1 到 FA4）均针对最新的 NVIDIA 芯片进行了手工优化（FA3 适用于 Hopper 的异步管道，FA4 适用于 Blackwell），与其他硬件的端口相隔数月或数年。</p>
<p>CUDA 的 moat 不只是 API 本身，更是二十多年积累的 third-party kernel、library、tooling，以及已经掌握这套 programming model 的大量 developer。很多关键 software 并非 NVIDIA 内部完成，但它们共同放大了 CUDA ecosystem 的价值。</p>
<p>NVIDIA 同时提供大量 engineering support。它会把 engineer 直接嵌入 frontier lab 和 hyperscaler 团队，为新 model architecture 编写 kernel，并针对最新 silicon 调优。因此，切换离开 NVIDIA 不只是重写 library 和 kernel，还意味着重新建立团队的 performance mental model，并失去这部分现场 engineering support。</p>
<hr/>
<h3 id="google-tpu">Google TPU</h3>
<div class="philosophy">
<p><strong><a href="https://en.wikipedia.org/wiki/Tensor_Processing_Unit">TPU</a></strong> 是一台以 matrix multiplication 为中心的 domain-specific machine。它不追求像 GPU 那样覆盖所有 massively parallel workload，而是把主要 silicon 投入大型 systolic array，再让 XLA 在编译期安排每个 cycle 和每一块 memory。系统没有 hardware thread scheduler 或 cache hierarchy；多个 TPU 通过 ICI 组成 pod，由 compiler 统一生成 SPMD program 和 collective。TPU 的目标很明确：以更高 performance-per-watt 运行 Google 自身的 search、recommendation、translation 和 Gemini workload。</p>
</div>
<h4 id="genealogy">演进路线</h4>
<div class="genealogy">
<div class="gen-row"><div class="gen-year">2015</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://arxiv.org/abs/1704.04760">TPU v1</a></em></strong><span class="gen-chip"><a data-popup="tpu-v1">v1</a></span></div><div class="gen-desc">首款量产深度学习 ASIC； <a data-popup="int8">INT8</a> 仅进行推理 <a data-popup="pcie">PCIe</a>。</div></div></div>
<div class="gen-row"><div class="gen-year">2017</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://cacm.acm.org/research/a-domain-specific-supercomputer-for-training-deep-neural-networks/">TPU v2</a></em></strong><span class="gen-chip"><a data-popup="tpu-v2">v2</a></span></div><div class="gen-desc">第一个具有训练功能的 TPU；将 <a data-popup="mxu">MXU</a> 从 <a data-popup="int8">INT8</a> 切换为 <strong><a data-popup="bf16">BF16</a></strong>，建立双<a data-popup="tensorcore">TensorCore</a> + <a data-popup="hbm">HBM</a></div></div></div>
<div class="gen-row"><div class="gen-year">2018</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://cacm.acm.org/research/a-domain-specific-supercomputer-for-training-deep-neural-networks/">TPU v3</a></em></strong><span class="gen-chip"><a data-popup="tpu-v3">v3</a></span></div><div class="gen-desc">首款液冷 TPU；与 v2 相比， <a data-popup="mxu">MXU</a> 和 <a data-popup="hbm">HBM</a> 翻倍； 1,024 芯片 Pod。</div></div></div>
<div class="gen-row"><div class="gen-year">2020</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://arxiv.org/abs/2304.01433">TPU v4</a></em></strong><span class="gen-chip"><a data-popup="tpu-v4">v4</a>， <a data-popup="tpu-v4i">v4i</a></span></div><div class="gen-desc">首款可重新配置 <a data-popup="ocs">Optical Circuit Switch（OCS）</a> (<a data-popup="palomar">Palomar</a>); <a data-popup="sparsecore">SparseCores</a>;两者 <a data-popup="bf16">BF16</a> &amp; <strong><a data-popup="int8">INT8</a></strong>; 4,096 芯片 Pod。</div></div></div>
<div class="gen-row"><div class="gen-year">2023</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-tpu-v5p-and-ai-hypercomputer">TPU v5</a></em></strong><span class="gen-chip"><a data-popup="tpu-v5e">v5e</a>， <a data-popup="tpu-v5p">v5p</a></span></div><div class="gen-desc">v5e 提高效率，v5p 提高性能； v5p 具有 3.3× INT8 FLOP 和 2.2× v4、8,960 芯片 Pod 的 HBM 带宽。</div></div></div>
<div class="gen-row"><div class="gen-year">2024</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus">Trillium</a></em></strong><span class="gen-chip"><a data-popup="tpu-v6e">v6e</a></span></div><div class="gen-desc">第一个 256×256 <a data-popup="mxu">MXU</a>；相似功率下 4.7× v5e 峰值 FLOPS；训练有素 <strong>Gemini 2.0</strong>。</div></div></div>
<div class="gen-row"><div class="gen-year">2025</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/ironwood-tpu-age-of-inference/">Ironwood</a></em></strong><span class="gen-chip"><a data-popup="tpu-v7">v7</a></span></div><div class="gen-desc">面向 reasoning model inference；添加原生 <strong>FP8</strong>； 9,216 chip Superpod， <strong>42.5 ExaFLOPS FP8</strong>。</div></div></div>
<div class="gen-row"><div class="gen-year">2026</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/">TPU v8</a></em></strong><span class="gen-chip"><a data-popup="tpu-8t">8t</a>、 <a data-popup="tpu-8i">8i</a></span></div><div class="gen-desc"> 8t 用于训练，8i 用于推理；添加原生 <strong>FP4</strong>； 9,600 个 chip Superpod， <strong>121 ExaFLOPS FP4</strong> (8t)。</div></div></div>
</div>
<h4 id="architecture">架构</h4>
<p>TPU 的核心是一台 matmul engine，外围逻辑的主要任务就是持续为它搬运数据。基本计算单元是 <strong><a data-popup="tensorcore">TensorCore</a></strong>：v2 之后的旗舰型号每个封装包含两个 TensorCore，v4i、v5e、v6e 等面向能效优化的型号则包含一个。TensorCore 内部由五类单元组成：负责矩阵运算的 <a data-popup="mxu">MXU</a>、负责 element-wise ops 的 <a data-popup="vpu">VPU</a>、控制执行流程的 <a data-popup="scalar-unit">Scalar Unit</a>、执行 cross-lane reduction 的 <a data-popup="xlu">XLU</a>，以及 Transpose/Permute Unit；accumulator queue 负责向 MXU 输入和排出数据。从 v4 开始，芯片还增加了独立的 <a data-popup="sparsecore">SparseCore</a>，专门处理 embedding lookup 这类不适合 systolic array 的 workload。所有单元连接在同一个 VLIW issue plane 上，由 <a data-popup="core-sequencer">Core Sequencer</a> 每周期发出一个 322-bit bundle。这里没有 I-cache miss、warp scheduler、out-of-order engine 或 branch predictor：compiler 本身就是 scheduler，省下的硅面积用于放置更多 MAC。</p>
<p><img alt="TPU Ironwood / v8t 单封装布局图 — 两个计算小芯片并排放置在芯片到芯片桥上；每个小芯片顶部都有一个 TensorCore 和两个 SparseCore 数据流引擎，两侧是 HBM3e 堆栈。 ICI 端口贯穿 3D 环面的顶部和底部，右上角有一个用于横向扩展的小型 DCN NIC。" loading="lazy" src="images/google-tpu-chip.png"/></p>
<p><img alt="Zoom 中的一个 TensorCore — 顶部的 Scalar Unit 在每个周期将 322 位 VLIW 捆绑发射到 8 个功能槽中：VPU 通过其 2D vector lane 运行 element-wise ops 运算； XLU 和 Transpose/Permute 单元处理 cross-lane reduction 和 layout shuffle；四个 256×256 MXU 执行收缩 matmul。accumulator 队列将 partial sum 放入 VMEM，这是一个 software-managed scratchpad，为阵列提供数据和耗尽数据。" loading="lazy" src="images/google-tpu-tensorcore.png"/></p>
<h5 id="tensorcore">TensorCore</h5>
<p> <em><strong><a data-popup="mxu">MXU</a></strong></em> 是脉动阵列。 <a data-popup="tpu-v1">v1</a> 发货 1 个 256×256 INT8 推理阵列； <a data-popup="tpu-v2">v2</a> 是第一个具有训练功能的 TPU，并引入了 128×128 个单元， <a data-popup="bf16">BF16</a> 与 <a data-popup="fp32">FP32</a> 累积（INT8 在 <a data-popup="tpu-v4">v4</a> 以同等吞吐量返回到 MXU）。每个 TensorCore 的单元数量从此开始增长： <a data-popup="tpu-v2">v2</a> 上有 1 个 MXU→ <a data-popup="tpu-v3">v3</a> 上有 2 个→ 4 上 <a data-popup="tpu-v4">v4</a>/<a data-popup="tpu-v5e">v5e</a>/<a data-popup="tpu-v5p">v5p</a>。 <a data-popup="tpu-v6e">Trillium</a> 回到 256×256（每个周期每个阵列 65,536 个乘法累加单元），并且 <a data-popup="tpu-v7">Ironwood</a>、 <a data-popup="tpu-8t">8t</a>和 <a data-popup="tpu-8i">8i</a> 都保留了 256×256 形状。</p>
<p>计算矩阵乘法时，矩阵 B 会预先装入 systolic array 的 weight cell，这就是 <strong><a data-popup="weight-stationary">weight-stationary dataflow</a></strong>。activation 从阵列左侧进入，每周期向右传播一列，与 cell 中驻留的 weight 相乘；partial sum 则向下流入 accumulator queue。数据一旦进入阵列，就不再访问 memory：weight 会被所有经过的 activation 复用，activation 也会沿一整行复用。矩阵乘法本身只消耗几 pJ，访问 memory 的能耗通常高出两个到三个数量级，因此 systolic array 的价值就是把 data reuse 直接做进连线。代价是 array underfill：在 256×256 array 上运行 128×128 matmul 时，75% 的计算单元处于空闲状态，所以 XLA 会把 tile、padding 和 schedule 对齐到 array dimension。</p>
<p><strong><a data-popup="vpu">VPU</a></strong> 的算力不如 MXU 醒目，但它是 TPU 微架构中很关键的一部分：TPU 的 VPU 是 2D vector machine，而不是普通的 1D SIMD。VPU register file 保存二维 <a data-popup="vreg">VREG</a>；以 v4/v5p 为例，其 shape 为 <code class="notranslate" translate="no">(8, 128)</code>，即 128 个 lane、8 个 sublane。lane axis 与 MXU 的输入宽度对齐，sublane axis 则决定 VPU 向 MXU 流式提供 tile 的节奏。现代 TPU kernel 的性能很大程度来自 <strong><a data-popup="vpu-mxu-overlap">VPU/MXU overlap</a></strong>：MXU 执行 matmul 的同时，VPU 并行完成 quantization、LayerNorm、Softmax、activation 和 bias add。cross-lane reduction 对 2D vector ISA 很难高效实现，因此交给 <strong><a data-popup="xlu">XLU</a></strong>；它延迟高、代价大，也是 compiler 中常见的热点。与二维布局不一致的变换则由 Transpose/Permute Unit 处理，避免数据绕行内存。</p>
<p> <em><strong><a data-popup="scalar-unit">Scalar Unit</a></strong></em> 是最小的块，并且可以说是最重要的：一个单 thread、双发出整数 ALU，具有 32 个 32 位寄存器和 4 KiB 的 <em><strong><a data-popup="smem">SMEM</a></strong></em> 用于控制状态，与保存程序的 Imem 配对。它是唯一执行指令提取的块；每个周期它都会提取一个 322 位 VLIW 包，在本地执行自己的两个 scalar 槽（地址算术、循环计数器、分支、同步寄存器检查），并将剩余的 6 个槽分派到芯片的其余部分：2 个 vector ALU (VPU)、2 个 vector 加载/存储 (HBM↔VMEM <a data-popup="dma">DMA</a>)，2 个矩阵（推送/弹出 MXU 队列）。块之间的同步是显式的： <em><strong><a data-popup="sync-flags">同步标志</a></strong></em> 在 MXU 和 VPU 管道繁忙时进行跟踪，并且编译器插入屏障检查而不是硬件跟踪依赖项。Scalar Unit 使 TensorCore 的其余部分看起来像固定功能数据流：每个周期，由同一个控制点决定八个 functional slot 的行为，并且没有动态 <a data-popup="reorder-buffer">重新排序缓冲区</a> 来撤消错误的决定。</p>
<h5 id="memory">内存</h5>
<p>片上内存层次结构与计算端的思路相同： <em><strong>没有缓存，每个级别均由软件管理</strong></em>。片外为 <em><strong><a data-popup="hbm">HBM</a></strong></em> （v2/v5e 上为 16 GB，v3/v4/v6e 上为 32 GB，v5p 上为 95 GB，Ironwood 上为 192 GB，v8 代上为 216–288 GB），片上是手工堆叠的可明确寻址的 scratchpad 层。最接近计算的是 <em><strong><a data-popup="vmem">VMEM</a></strong></em>，为 VPU 和 MXU 输入队列提供数据的 vector scratchpad，在 v4 上大小为 32 MiB，在 v5e 上大小为 128 MiB，在经过推理调整的 <a data-popup="tpu-8i">v8i 上扩展到 384 MiB</a> 精确地在芯片上保存整个 <a data-popup="kv-cache">KV 缓存</a> 。其上方是 <em><strong><a data-popup="cmem">CMEM</a></strong></em>，随 <a data-popup="tpu-v4">v4</a> 引入，大小为 128 MiB：HBM 和 VMEM 之间较慢、较大的 SRAM 暂存区域，可吸收 <a data-popup="op-fusion">融合操作</a> 中间体。 <a data-popup="scalar-unit">Scalar Unit</a> 有自己的 <em><strong><a data-popup="smem">SMEM</a></strong></em> （v4 上的控制状态约为 10 MiB）和一个微小的 scalar register file。程序中的每个张量都会在编译期绑定到确定的存储层级； XLA 的 <a data-popup="buffer-assignment">buffer assignment</a> 跨层调度 <a data-popup="dma">DMA</a> ，以便数据恰好在循环之前到达消耗它。硬件不进行预取，不逐出，不存在 <a data-popup="cache-coherence">一致性</a>；schedule 正确时，数组永远不会停止；schedule 出错时，没有后备路径。</p>
<h5 id="sparsecore">SparseCore</h5>
<p>TensorCore 之外最特别的单元是 <strong><a data-popup="sparsecore">SparseCore</a></strong>。recommendation/ranking model 的 embedding lookup 具有 irregular、indirect、all-to-all 的访问模式，和 dense matmul 完全不同，并不适合 256×256 systolic array。SparseCore 是专用 dataflow processor，包含 16 个 compute block 和独立的 SPMEM scratchpad，负责 scatter、gather、segmented reduction 以及 embedding sharding 带来的 data-dependent all-to-all。Google 报告它能以约 5% 的 die area 和 power，为 embedding-heavy model 带来 5–7× speedup。v4、v5p 和 Ironwood 每颗芯片有 4 个 SparseCore，Trillium 为 2 个；面向 inference 的 v8i 则移除了 SparseCore，改为在 I/O chiplet 上加入 CAE（Collective Acceleration Engine），针对 decode 中的 collective reduction。</p>
<h5 id="numerics">数值格式</h5>
<p>TPU <a data-popup="tpu-v1">v1</a> 为 <a data-popup="int8">INT8</a>-仅限推理； <a data-popup="tpu-v2">v2</a> 将此切换为 <em><strong><a data-popup="bf16">BF16</a></strong></em> 作为规范训练格式：与 <a data-popup="fp32">FP32</a>相同的动态范围，一半的内存，无 <a data-popup="loss-scaling">loss scaling</a> 技巧。 <a data-popup="tpu-v4">v4</a> 重新引入了原生 INT8 支持。 <a data-popup="tpu-v7">Ironwood</a> 然后添加原生 <em><strong><a data-popup="fp8">FP8</a></strong></em> 支持（E4M3 和 E5M2），吞吐量约为 2 倍 BF16 在同一地区。 v8 添加原生 <em><strong><a data-popup="fp4">FP4</a></strong></em> 加上 MXU 本身内部的 <em><strong><a data-popup="block-scale-multiplication">块级乘法</a></strong></em> ，这会删除 Ironwood 仍需支付的 VPU 反量化开销。 <em><strong><a data-popup="stochastic-rounding">stochastic rounding</a></strong></em> 在每个现代 TensorCore 上都受到硬件支持：由作为概率的较低尾数位做出的舍入决策，这保留了长时间训练运行中低精度累积的预期值，并且是让 BF16/FP8 缩小与 FP32 精度差距的小细节之一。</p>
<p>位于芯片边界 <em><strong><a data-popup="ici">ICI</a></strong></em> 自行移植（ <a data-popup="2d-torus">2D-torus</a> 芯片上有 4 个端口 v2/v3/v5e/v6e、 <a data-popup="3d-torus">3D-torus</a> 旗舰版 v4/v5p/v7/8t 上的 6 个）以及 <a data-popup="dcn">DCN</a> NIC 用于横向扩展。从芯片级的角度来看，ICI 端口看起来只是另一组 <a data-popup="dma">DMA</a> 引擎 <a data-popup="core-sequencer">核心定序器</a> 可以在 VLIW 包中定位：远程张量发送与 VMEM 到 HBM 传输是相同的指令类，并且编译器处理 <a data-popup="collective">collective</a> 作为同一总体 <a data-popup="schedule">计划的一部分</a> 它为计算和本地内存构建。</p>
<h5 id="bets">设计取舍</h5>
<ul>
<li><em><strong>脉动阵列。</strong></em> Matmul 主导工作负载，因此将芯片用于脉动阵列。</li>
<li><em><strong>软件 scratchpad。</strong></em> 计算成本低廉，内存昂贵，因此重用阵列线路中的数据，并用 software-managed scratchpad 替换缓存。</li>
<li><em><strong>编译器调度。</strong></em> 工作负载是静态可预测的，因此将调度移至编译器中：VLIW 问题，否 <a data-popup="speculation">推测</a>，无乱序，无 <a data-popup="dynamic-scheduler">动态调度器</a>。</li>
<li><em><strong>押注 4：仅 MAC 芯片。</strong></em> 功率比峰值更重要，因此删除每个不 <a data-popup="mac">乘法添加</a>的晶体管：每个缓存标记、每个分支预测器、每个重新排序缓冲区。</li>
<li><em><strong>专用的数组外引擎。</strong></em> 密集的 matmul 数组对于某些实际工作负载来说是错误的形状（<a data-popup="embeddings">嵌入</a>、 <a data-popup="collective">collective</a>），因此请开发小型专用引擎（SparseCore、CAE），而不是 warp 主核心以适应它们。</li>
</ul>
<h4 id="scaling">扩展</h4>
<p>TPU 的 scale-up 路线与 NVIDIA 不同。NVLink/NVSwitch 提供的是硬件管理的 coherent address space，一颗 GPU 可以直接访问另一颗 GPU 的 HBM；ICI 则是 message-passing fabric，没有 remote load/store semantics，也不提供 cache coherence。所有跨芯片操作都表现为由 XLA 编译的显式 collective。TPU 通过 ICI 把芯片直接连成 2D/3D torus，机架边界再由 Optical Circuit Switch（OCS）重构连接关系，而不是依赖传统 crossbar。</p>
<div class="definitions">
<div class="definition"><div class="definition-term">纵向扩展</div><div class="definition-body">通过 <a data-popup="ici">ICI</a> 将芯片直接连接成 <a data-popup="2d-torus">2D</a> 或 <a data-popup="3d-torus">3D 环面</a>。XLA 发出 <a data-popup="spmd">SPMD</a> collective communication，将数千颗 TPU 紧密编排成一台机器。它不提供缓存一致性，但能以低延迟提供巨大的 <a data-popup="bisection-bandwidth">二分带宽</a>。</div></div>
<div class="definition"><div class="definition-term">横向扩展</div><div class="definition-body">网络 <a data-popup="pod">pod</a> 通过数据中心结构一起：在一个 <a data-popup="ici">ICI</a> 域中容纳的芯片数量要多得多，且每芯片带宽较低。今天： <a data-popup="virgo">Virgo</a> 处理东西向 TPU 流量 (v8t+)， <a data-popup="jupiter">Jupiter</a> 处理南北。 <a data-popup="multislice">Multislice</a> + <a data-popup="pathways">Pathways</a> 编排 <a data-popup="spmd">SPMD</a> 跨 pod。</div></div>
</div>
<h5 id="scale-up">纵向扩展</h5>
<p><a data-popup="ici">ICI</a> 链接直接来自 TPU 芯片：高速 <a data-popup="serial-lanes">串行 lane</a>， <a data-popup="dac">直连铜</a> 位于 64 芯片立方体（ 4×4×4 排列，位于一个液冷机架中），光学立方体之间。每芯片聚合 <a data-popup="ici">ICI</a> 带宽已从 <a data-popup="tpu-v2">v2</a> 上的约 250 GB/s 扩展到 <em><strong>1.2 TB/s 双向 <a data-popup="tpu-v7">Ironwood</a></strong></em>和 <em><strong>2×</strong></em> 在 <em><strong><a data-popup="tpu-8t">v8t</a></strong></em>上。拓扑按代交替： <a data-popup="2d-torus">2D 环面</a> ，位于效率调整芯片上（<a data-popup="tpu-v2">v2</a>， <a data-popup="tpu-v3">v3</a>、 <a data-popup="tpu-v5e">v5e</a>、 <a data-popup="tpu-v6e">v6e</a>)、 <a data-popup="3d-torus">3D 环面</a> 在旗舰产品上（<a data-popup="tpu-v4">v4</a>, <a data-popup="tpu-v5p">v5p</a>, <a data-popup="tpu-v7">v7</a>、 <a data-popup="tpu-8t">v8t</a>)。</p>
<p> <em><strong><a data-popup="palomar">Palomar OCS</a></strong></em>没有 NVIDIA 类似产品： <em><strong><a data-popup="3d-mems">3D-MEMS</a> <a data-popup="ocs">Optical Circuit Switch（OCS）</a></strong></em> 位于立方体之间。微小的镜子通过物理旋转将任何输入光纤映射到任何输出。 <a data-popup="tpu-v4">v4</a> <a data-popup="superpod">superpod</a> 使用 48 个 Palomar 开关将 64 个立方体（4,096 个芯片）连接到一个 3D 环面； <a data-popup="tpu-v5p">v5p</a> 和 <a data-popup="tpu-v7">Ironwood</a> 扩展了相同的方案。重新配置是毫秒级的，而不是纳秒级的，但这没关系，因为 <a data-popup="ocs">OCS</a> 是 <em><strong>电路交换</strong></em>：在作业开始时选择一个拓扑，运行一周，然后为下一个工作负载重新配置。三个问题归结为一个组件：每个工作负载的拓扑重新配置（<a data-popup="twisted-torus">warptorus</a> 优化高达 70% <a data-popup="bisection-bandwidth">二分</a>）、子 pod <a data-popup="slice">按需切片</a> 以及 <em><strong>容错</strong></em> （当芯片失效时， <a data-popup="ocs">OCS</a> 光学交换到备用立方体中，并且运行继续，而不会丢失 <a data-popup="ici">ICI</a> 域）。</p>
<p><img alt="TPU Ironwood 超级 Pod — 左：一个由 64 个芯片 (4×4×4) 组成的立方体，连接在 3D torus 中，在最近的邻居之间直接连接铜，并在每个面上包边。右图：Palomar OCS 将 144 个立方体拼接成一个相干 ICI 域，Palomar OCS 是一种 3D-MEMS Optical Circuit Switch，可根据工作负载重新配置拓扑。" loading="lazy" src="images/google-tpu-scale-up.png"/></p>
<p>这使得 <em><strong><a data-popup="superpod">超级 Pod</a></strong></em> 扩展单位：相当于 NVIDIA 的 NVL72，大两个数量级。 <a data-popup="tpu-v4">v4</a> 为 4,096 个芯片； <a data-popup="tpu-v5p">v5p</a>, 8,960; <em><strong><a data-popup="tpu-v7">Ironwood (TPU v7)</a></strong></em> 是 9,216 个芯片，排列为 144 个 64 的立方体，呈现 <em><strong>1.77 PB 的 <a data-popup="hbm">HBM</a> （~68 PB/s）和 42.5 ExaFLOPS <a data-popup="fp8">FP8</a></strong></em> 作为一个连贯的 <a data-popup="ici">ICI</a> 域。</p>
<p><em><strong><a data-popup="tpu-8t">TPU 8t (Sunfish)</a></strong></em> 将其拉伸至 <em><strong>9,600 个芯片、2 PB 的 <a data-popup="hbm">HBM</a> （约 62 PB/s）和 121 ExaFLOPS FP4</strong></em>。 <em><strong><a data-popup="tpu-8i">TPU 8i (Zebrafish)</a></strong></em> 拥有 <em><strong>1,024 个芯片，约 295 TB 的 <a data-popup="hbm">HBM</a> (8.8 PB/s) 和 ~10 ExaFLOPS FP4</strong></em>。 8i 用新的分层 <a data-popup="high-radix">高基</a> 拓扑取代了 torus，称为 <em><strong><a data-popup="boardfly">Boardfly</a></strong></em> （4 片环 → 8 片组 → 通过 <a data-popup="ocs">OCS</a>连接最多 36 个组）、切割 <a data-popup="all-to-all">all-to-all</a> 延迟减半。这是专为 <a data-popup="moe">MoE</a> 推理而设计的。当 <a data-popup="collective">collective</a> 是最近邻时，3D 环面表现出色（<a data-popup="ring-all-reduce">环 all-reduce</a> 使用每个链接每个周期），但 <a data-popup="moe-expert-routing">MoE 专家路由</a> 是相反的模式， <a data-popup="all-to-all">all-to-all</a>：每个芯片都向每个芯片发送独特的片段，往返延迟受最长跳对的限制。 1,024 芯片 3D 环面的直径为 16 跳； <a data-popup="boardfly">Boardfly</a>的环 → 组 → OCS 层次结构将其压缩为 7。</p>
<h5 id="scale-out">横向扩展</h5>
<p>通过 <a data-popup="tpu-v7">TPU v7</a>，横向扩展在单个结构上运行： <em><strong><a data-popup="jupiter">Jupiter</a></strong></em>、 <em><strong>自 2022 年起在脊柱上采用全光学</strong></em> 通过 <em><strong><a data-popup="apollo">Apollo OCS</a></strong></em>，与 <a data-popup="palomar">Palomar</a>相同的 3D-MEMS 系列，在整个建筑中扩展。Google 在从机架到数据中心骨干的每一层都使用相同的原语（光电路交换）；这是其他人没有的建筑特征。 <a data-popup="jupiter">Jupiter</a> 今天每颗 <a data-popup="bisection-bandwidth">二分</a> 承载着 13 Pb/s</p>
<p>使用 <em><strong><a data-popup="tpu-8t">TPU 8t</a></strong></em>，横向拆分为两个结构。东西向 TPU 到 TPU 流量移至 <em><strong><a data-popup="virgo">Virgo</a></strong></em>，专用加速器结构； <a data-popup="jupiter">Jupiter</a> 保留了 <em><strong>南北</strong></em> 角色：存储访问、通用计算、和站点间缩放。 <a data-popup="virgo">Virgo</a> 是 <em><strong><a data-popup="flat-topology">扁平</a>， <a data-popup="two-layer">两层</a>、 <a data-popup="non-blocking">非阻塞</a></strong></em> 拓扑构建于 <em><strong><a data-popup="high-radix">high-radix</a></strong></em> 开关：每个 TPU 最多有两个开关。一个 <a data-popup="virgo">Virgo</a> 集群链接 134,000+ <a data-popup="tpu-8t">TPU 8t</a>s，47 Pb/s 的 <a data-popup="bisection-bandwidth">二分</a> （每芯片带宽的 4 倍，降低 40% <a data-popup="unloaded-latency">卸载延迟</a> 比上一代 <a data-popup="dcn">DCN</a> 一代）， <a data-popup="multi-planar-fault-isolation">多平面故障隔离</a> 和 <a data-popup="sub-millisecond-telemetry">亚毫秒遥测</a> 让调度程序在破坏步骤之前杀死落后者。架构的回报是，每一层现在都可以独立发展：纵向扩展、东西横向扩展，前端可以以不同的节奏进行迭代，而无需重新连接其他层。</p>
<p><img alt="TPU 8t 横向扩展 — 东西向 TPU 到 TPU 流量穿过 Virgo，这是一种扁平的两层无阻塞高基数交换机结构，可将任何 TPU 置于任何其他 TPU 的两个交换机跃点内（134,000 多个 TPU，47 Pb/s 对分）。南北流量（存储、通用计算、站点间）保留在 Jupiter 上，自 2022 年以来，Jupiter 的骨干一直通过 Apollo OCS 实现全光纤化。" loading="lazy" src="images/google-tpu-scale-out.png"/></p>
<p>每芯片横向扩展带宽在 <em><strong>Ironwood <a data-popup="tpu-v7">上约为</a></strong></em>100 Gbps， <em><strong>4×</strong></em> 在 <em><strong><a data-popup="tpu-8t">v8t</a></strong></em>上，但仍然比每芯片 <a data-popup="ici">ICI</a>小两个数量级。这种带宽差距决定了分区： <a data-popup="tensor-parallelism">tensor parallel 性</a> 和 <a data-popup="moe-expert-routing">MoE 专家路由</a> 留在内部 <a data-popup="ici">ICI</a>; <a data-popup="data-parallelism">数据并行</a> 和 <a data-popup="pipeline-parallelism">管道并行性</a> 跨横向扩展结构。</p>
<p>Google 的 <em><strong><a data-popup="multislice">Multislice</a></strong></em> 框架，深入 <em><strong><a href="https://openxla.org/xla">XLA</a></strong></em>，让单个 <a data-popup="spmd">SPMD</a> 程序跨越多个 <a data-popup="slice">在不同的</a> pod <a data-popup="pod">中切片</a>；编译器在每个切片内 issue hierarchical <a data-popup="collective">collective</a> （<a data-popup="ring-all-reduce">ring all-reduce</a> ， <a data-popup="higher-level-reduce">higher-level reduction</a> ）。该结构正是隐藏 <a data-popup="ici">ICI</a>/<a data-popup="dcn">DCN 的技巧</a> 带宽差距：在快速 ICI 上，尽可能多的工作保留在片内，仅留下跨片剩余部分来支付慢速结构成本。</p>
<p>更上层是 <strong><a data-popup="pathways">Pathways</a></strong>。传统 NCCL + Slurm + Megatron 系统通常由多个 controller 驱动 SPMD job；Pathways 则由单个 client 统一控制，并把多个拥有独立 ICI domain 的 pod 虚拟化成一台逻辑机器。它负责 gang scheduling、elastic training 和跨 region orchestration：某个 slice 失败后，OCS 可以重构 topology，Pathways 再从 checkpoint 在新 topology 上恢复。Gemini Ultra 是首批跨多个数据中心训练的 frontier model，Pathways 将这些数据中心组织成一个同步 SPMD job。</p>
<p>理念： <em><strong>编译器是调度器，环面是拓扑，光开关是通用可重构器底层</strong></em>，位于从机架到数据中心的每一层。</p>
<h4 id="software">软件栈</h4>
<p>TPU software stack 是 compiler-driven，而 CUDA 更偏 kernel-driven。GPU developer 通常直接编写或调用 kernel，compiler 主要做局部优化；TPU developer 在 JAX 中描述数值程序，XLA 则负责下层大部分决策：op fusion、tensor residency、2D VREG layout、HBM↔VMEM DMA、VLIW bundle schedule，以及跨数千颗 chip 的 sharding。TPU 没有 warp scheduler、cache 或 out-of-order engine 来掩盖坏 schedule，compiler 本身就是 system。它的优势是自动化程度高，更容易接近整体 performance ceiling；代价是当自动生成结果仍有差距时，手工补齐最后一段性能更困难。</p>
<p>compile path 是 <strong>JAX → JAXpr → StableHLO → HLO → LLO → VLIW bundle</strong>。JAX 把 Python function trace 成 typed functional IR（JAXpr），再 lower 到 StableHLO；XLA ingest StableHLO/HLO 后执行 op fusion、layout assignment、buffer assignment 和 SPMD partitioning。随后 TPU backend lower 到 LLO，并生成最终 VLIW stream。一个高质量 schedule 会在同一时间窗口中 overlap MXU matmul、VPU element-wise ops 和 HBM↔VMEM DMA，从而让 compute engine 与 memory pipeline 同时保持 busy。</p>
<p>多芯片执行是 <em><strong><a data-popup="spmd">SPMD</a></strong></em>：一个程序、分片数据、分层 <a data-popup="collective">集体</a>、 <a data-popup="emit">发布</a> 由 <em><strong><a data-popup="gspmd">GSPMD</a></strong></em> （现已被 <em><strong><a data-popup="shardy">Shardy</a></strong></em>取代， <a data-popup="mlir">MLIR</a>- 原生后继者将于 2026 年初作为默认版本）。用户以声明方式表达分片 <a data-popup="declarative"></a> 与 <em><strong><a data-popup="mesh">网格</a></strong></em> + <em><strong><a data-popup="partitionspec">PartitionSpec</a></strong></em> 几个关键张量的注释； <a data-popup="compiler">编译器</a> 通过 <a data-popup="graph">图的其余部分传播分片</a> 并插入 <a data-popup="all-reduce">全部归约</a>、 <a data-popup="all-gather">全部聚集</a>和 <a data-popup="reduce-scatter">reduce-scatters</a> 布局发生变化的地方。当 <a data-popup="compiler">编译器</a> 选择了错误的集合时， <em><strong><a data-popup="shard-map">shard_map</a></strong></em> 会将用户放入 <em><strong>手动 SPMD</strong></em> （具有显式本地形状和显式 <a data-popup="collective">collective</a>的每设备代码），可在内部组合 <code class="notranslate" translate="no">jit</code> 因此可以对单个 core 进行手动分区，而无需在其他地方放弃自动分区。这是 PyTorch 习惯用法的反面： <em><strong><a data-popup="fsdp">FSDP</a></strong></em> 和 <em><strong><a data-popup="deepspeed">DeepSpeed</a></strong></em> 将模型包装在运行时中模块边界处的问题 <a data-popup="collective">collective</a> ； GSPMD/Shardy 将整个 <a data-popup="graph">图</a> 划分为 <a data-popup="compiler">编译器</a> 问题。</p>
<p><strong><a data-popup="pallas">Pallas</a></strong> 是 JAX 在 TPU 上的 kernel escape hatch，作用大致对应 GPU 上的 Triton。Pallas kernel 用 JAX 风格的 Python 编写，经基于 MLIR 的 Mosaic lower 到 LLO，再作为 custom op 嵌回 HLO。它解决的是 XLA 不擅长自动生成的 schedule-sensitive kernel，例如新的 attention variant、fused MoE schedule，以及需要手动控制 VMEM tiling 和 DMA 的 FlashAttention-class 优化。Mosaic GPU 使用同一套前端面向 H100/Blackwell，因此 kernel author 可以复用编程模型。更上层仍是 JAX 原生生态：Flax NNX、Optax、Orbax、Grain、Tunix、Qwix，以及用于 LLM 的 MaxText 和用于 diffusion model 的 MaxDiffusion。</p>
<p>PyTorch 路径是真实的，但是二流的。 <em><strong><a data-popup="torch-xla">torch_xla</a></strong></em> 使用 <em><strong><a data-popup="lazy-tensor">LazyTensor</a></strong></em> 机制：每个 PyTorch 操作记录到一个 HLO <a data-popup="graph">图</a> 中，并在下一个 PyTorch 操作中进行编译屏障，编译后的工件由图形形状哈希缓存。 PyTorch/XLA 2.x 添加了 <em><strong>GSPMD 风格的分片注释</strong></em>， <em><strong><code class="notranslate" translate="no">torch.compile</code> 通过 XLA 后端集成</strong></em> 、 <em><strong>JAX 桥</strong></em>和 (PyTorch/XLA 2.7) C++11-ABI 构建，跟踪速度明显加快。与 JAX 的差距是真实存在的（JAX 的原语更清晰地映射到 <a data-popup="stablehlo">StableHLO</a>，并且更好地涵盖了复杂的并行策略），这就是为什么 <em><strong><a data-popup="vllm-tpu">vLLM TPU</a></strong></em> （由 Cloud Next 2025 上宣布的 <em><strong><a data-popup="tpu-inference">tpu-inference</a></strong></em> 插件提供支持）降低 <em><strong>每</strong></em> 模型，JAX 定义或 PyTorch 定义，通过 <em><strong>统一 JAX→XLA 路径</strong></em>。 <em><strong><a data-popup="torchtpu">TorchTPU</a></strong></em>于 2026 年 4 月宣布，Google 的回应是：采用 Eager 模式的原生 PyTorch 体验、 <code class="notranslate" translate="no">torch.distributed</code>和 XLA 上的 <code class="notranslate" translate="no">torch.compile</code> ，有望取代 torch_xla。</p>
<p>与 CUDA 相比，TPU ecosystem 更集中：XLA、JAX、Flax、Optax、Pallas、MaxText、Pathways、Shardy、Mosaic 等关键组件大多由 Google 主导，并与 hardware 同步演进。它缺少 CUDA 数十年积累的大量 third-party kernel，因此遇到非典型 workload 时灵活性较弱；但当 workload 与 Google 自身模型相似时，chip、ICI、OCS、compiler、runtime 和 model stack 可以一起 co-design。Ironwood 所谓的“co-designed AI stack”指的正是这种整体交付方式。Triton 与 <code class="notranslate" translate="no">torch.compile</code> 正在缩小不同平台的编程差异，但抽象边界仍很清楚：在 TPU 上 compiler 是主要接口；在 GPU 上，compiler 只是众多接口之一。</p>
<hr/>
<h3 id="amd-gpu">AMD GPU</h3>
<div class="philosophy">
<p>AMD Instinct 与 NVIDIA 的关注点不同。NVIDIA 持续在每一代 SM 内加入新的 Tensor Core primitive、async mechanism 和 memory tier；AMD 则让 CU microarchitecture 相对稳定，把更多 engineering effort 投向 package、memory capacity 和 open ecosystem。CDNA 3 很早就采用 XCD/IOD chiplet 与 3D stacking，MI300A 又把 CPU 与 GPU 放入 coherent HBM address space；HBM capacity 也长期是 Instinct 的优势。software 侧，AMD 选择 ROCm、HIP、Triton、OCP MX 和 UALink 等开放接口，试图用 open stack 缩小 CUDA ecosystem 的差距。</p>
</div>
<h4 id="genealogy">演进路线</h4>
<div class="genealogy">
<div class="gen-row"><div class="gen-year">2018</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/specs/radeon-instinct-mi50-data-sheet.pdf">Vega 20</a></em></strong><span class="gen-chip"><a data-popup="mi50">MI50</a>、 <a data-popup="mi60">MI60</a></span></div><div class="gen-desc">首款 7 nm GPU； 1:2 <a data-popup="fp64">FP64</a> vector 吞吐量。 <a data-popup="cdna">CDNA</a> / RDNA 之前的最后一个 GCN 系列本能。</div></div></div>
<div class="gen-row"><div class="gen-year">2020</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna-white-paper.pdf">CDNA</a></em></strong><span class="gen-chip"><a data-popup="mi100">MI100</a></span></div><div class="gen-desc">第一个 <a data-popup="mfma">MFMA</a> Matrix Core；图形固定功能硅完全下降。原生 <a data-popup="bf16">BF16</a>.</div></div></div>
<div class="gen-row"><div class="gen-year">2021</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf">CDNA 2</a></em></strong><span class="gen-chip"><a data-popup="mi210">MI210</a>， <a data-popup="mi250">MI250</a>、 <a data-popup="mi250x">MI250X</a></span></div><div class="gen-desc">第一个通过双 GCD 包的 MCM Instinct；全速率 <a data-popup="fp64">FP64</a> 矩阵</div></div></div>
<div class="gen-row"><div class="gen-year">2023</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf">CDNA 3</a></em></strong><span class="gen-chip"><a data-popup="mi300a">MI300A</a>、 <a data-popup="mi300x">MI300X</a></span></div><div class="gen-desc">首款 3D 堆叠小芯片 GPU： <a data-popup="xcd">XCD</a> 混合键合到 <a data-popup="iod">IOD</a> 通过 <a data-popup="tsv">TSV</a>; <a data-popup="fp8">FP8</a>; <a data-popup="infinity-cache">Infinity Cache</a>; MI300A 上一致的 CPU+GPU <a data-popup="apu">APU</a> ；动力 <a data-popup="el-capitan">El Capitan</a>。</div></div></div>
<div class="gen-row"><div class="gen-year">2024</div><div class="gen-body"><div class="gen-head"><strong><em>CDNA 3 刷新</em></strong><span class="gen-chip"><a data-popup="mi325x">MI325X</a></span></div><div class="gen-desc">相同计算， <a data-popup="hbm3e">HBM3E</a> 刷新：256 GB（6.0） TB/s。</div></div></div>
<div class="gen-row"><div class="gen-year">2025</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://www.amd.com/en/products/accelerators/instinct/mi350.html">CDNA 4</a></em></strong><span class="gen-chip"><a data-popup="mi350x">MI350X</a>, <a data-popup="mi355x">MI355X</a></span></div><div class="gen-desc">原生 <strong><a data-popup="fp4">FP4</a></strong> / FP6，具有 OCP MX microscaling 功能；每个 CU <a data-popup="fp64">FP64</a> 大约减少一半；第一代倾向于 AI 密度而非 HPC。</div></div></div>
<div class="gen-row"><div class="gen-year">2026</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://www.amd.com/en/blogs/2025/amd-advancing-ai-2025-mi400-helios-rack-scale-ai-platform.html">CDNA Next</a></em></strong><span class="gen-chip"><a data-popup="mi430x">MI430X</a>, <a data-popup="mi440x">MI440X</a>, <a data-popup="mi455x">MI455X</a></span></div><div class="gen-desc"><a data-popup="hbm4">HBM4</a>; <a data-popup="helios">Helios</a> 机架（72-GPU <a data-popup="mi455x">MI455X</a> 旗舰版 <a data-popup="ualoe">UALoE</a> 发布时，原生 <a data-popup="ualink">UALink</a> 从 2027 年开始）：AMD 对 NVL72 的第一个答案。</div></div></div>
</div>
<h4 id="architecture">架构</h4>
<div class="definitions">
<div class="definition analogy-key">
<div class="definition-term">术语</div>
<div class="definition-body">
<table class="analogy">
<thead>
<tr><th>AMD</th><th>NVIDIA</th></tr>
</thead>
<tbody>
<tr><td>计算单元(CU)</td><td>Streaming Multiprocessor（SM） (SM)</td></tr>
<tr><td>SIMD</td><td>SM 子分区</td></tr>
<tr><td>SIMD lane</td><td>CUDA Core (FP32 ALU)</td></tr>
<tr><td>波前 (wave64)</td><td>warp (warp32)</td></tr>
<tr><td>Matrix Core</td><td>Tensor Core</td></tr>
<tr><td>MFMA</td><td>mma.sync / wgmma / tcgen05.mma</td></tr>
<tr><td>VGPR / SGPR</td><td>register file</td></tr>
<tr><td>LDS（本地数据共享）</td><td>SMEM（共享内存）</td></tr>
<tr><td>Infinity Fabric</td><td>NVLink</td></tr>
</tbody>
</table>
</div>
</div>
</div>
<p>NVIDIA 的主要创新集中在 SM 内部：每一代都会增加新的 Tensor Core primitive、async data movement 机制和 operand storage。AMD CDNA 的重点则更多位于 CU 之间——如何把更多 CU、HBM 和 I/O die 组合成一个 coherent package。CU 本身相对稳定：四个 SIMD16、共享 Scalar Unit、LDS、L1 vector cache、VGPR/SGPR，以及执行 MFMA 的 Matrix Core。wave64 会在四个 cycle 内流过 16 条 SIMD lane，scheduler 在多个 resident wavefront 之间切换以隐藏 stall。CDNA 真正快速变化的是 CU 之外的 packaging 和 memory system。</p>
<p><img alt="AMD Instinct MI355X (CDNA 4) 封装布局图 — 八个 XCD（加速器复合芯片，每个大约 32 个活动 CU）通过 TSMC SoIC 混合键合到两个 IOD 基础芯片上。 IOD 搭载 256 MB Infinity Cache（每个 IOD 128 MB）、HBM PHY、Infinity Fabric 和 PCIe Gen 5。HBM3E 堆栈排列在外围； 8 个 12-Hi 堆栈，总计 288 GB。" loading="lazy" src="images/amd-gpu-chip.png"/></p>
<p><img alt="Zoom 中的一个计算单元 — 单个调度程序在四个周期内跨四个 SIMD16 Vector Engine 调度 wave64 波前；每个 SIMD 都有自己的 Matrix Core (MFMA)，在其旁边运行 matmul。一个共享 Scalar Unit、三个 register file (VGPR / SGPR / AGPR)、一个 160 KB LDS scratchpad 和一个 32 KB L1 vector 缓存完成了整个配方——自 2012 年 GCN 以来 AMD 一直保持着同样的形状。" loading="lazy" src="images/amd-cu.png"/></p>
<h5 id="compute">计算</h5>
<p>CU 内部、SIMD 和 Matrix Core 并排运行。四个 <a data-popup="simd-amd">SIMD</a> 按元素处理所有内容：激活、标准化、残差、地址算术。 <a data-popup="matrix-core">Matrix Core</a> 处理 matmul。拆分与 NVIDIA 的 <a data-popup="cuda-cores">CUDA Core</a> / <a data-popup="tensor-cores">Tensor Core</a> 拆分相同，但矩阵抽象具有沿着一条截然不同的曲线发展。</p>
<p>NVIDIA Tensor Core 的 issuer scope 一直在变化：Volta 由 32-thread warp 发射，Hopper 扩展到 128-thread warp group，Blackwell 又推进到 single-thread async issue，并支持 two-SM cluster。AMD Matrix Core 的抽象则更稳定：从 MI100 到 MI355X，MFMA 始终是 wavefront-scoped，一整个 wave64 共同发射 <code class="notranslate" translate="no">V_MFMA_*</code>，operand 来自 VGPR，accumulator 通常位于 AGPR。instruction latency 和支持的 datatype 持续变化，但 issuer scope 没有改变。CDNA 4 增加了面向 MFMA 的 LDS transpose load，使 operand 以目标 layout 直接送入 Matrix Core；它在功能上有些类似 TMA，但 MFMA 本身仍由整个 wavefront 发射。</p>
<p>吞吐量数字直接说明了格式的情况。 CDNA 1 于 2020 年推出，采用 FP32 / FP16 / <a data-popup="bf16">BF16</a> / INT8，每个 CU 每个周期 256 / 1024 / 512 / 1024 FLOP，具有原生 <a data-popup="bf16">BF16</a> 与 <a data-popup="a100">A100</a>一起支持。 CDNA 2 是 <a data-popup="fp64">FP64 的两倍</a> 通往 256 FLOPs/CU/cycle 的全速率矩阵的路径：AMD 独有的，将 MI250X 放入 <a data-popup="frontier">Frontier</a>的押注。 CDNA 3 在 <a data-popup="h100">FP8 上与</a> H100 <a data-popup="fp8">达到同等水平</a> 在 4,096 次浮点运算 (E4M3 + E5M2) 时，添加了 2:4 <a data-popup="structured-sparsity">structured sparsity</a>，并添加了 <a data-popup="xf32">TF32</a>- 通过截断尾数以 FP64 矩阵速率运行 FP32 matmul 的等效路径。 cDNA 4 再次翻倍 <a data-popup="fp4">FP4</a> 16,384 FLOPs 和 FP6，具有 <a data-popup="block-scale-multiplication">OCP MX 块缩放</a>，并在一个 MFMA 中添加了可混合的 A/B 精度：FP8 × FP4，例如。同一代将每个 CU FP64 吞吐量减半，这是第一款用 HPC 密度换取 AI 密度而不是两者兼而有之的 AMD 芯片。</p>
<p>波前范围决策体现在两种成本上。</p>
<p><em><strong>分歧。</strong></em> 半空 <a data-popup="wave64">wave64</a> 浪费 32 个 lane，而半空 warp32 浪费 16 个 lane。对于控制流基本一致的工作负载来说，这是一个很小的代价；</p>
<p><em><strong>重叠。</strong></em> NVIDIA 的异步、描述符驱动的 matmul 将问题与执行分离：发出 thread 触发指令并继续； Tensor Core 在后台运行； warp 可以运行 softmax、应用掩模或在前一个 matmul 仍在运行时预加载下一个 tile。 AMD 的波前集体 MFMA 没有给 Wave 提供等价物：发出 matmul 的同一波在待处理时无法同时进行有意义的 vector 工作。重叠可以跨越 <em>单独的</em> 波前，但必须在具有明确波前屏障的软件中进行，这更脆弱并且消耗更多的波槽和寄存器。</p>
<p>这种差异的重要程度取决于 workload。pure dense GEMM 中，matrix engine 已经饱和，matmul 周围几乎没有其他工作可 overlap，因此 AMD 的 wavefront-scoped MFMA 并不会吃太大亏，这也是 AMD 在 HPC workload 上长期强势的原因。FlashAttention-3/4 则不同：matmul、Softmax、mask 和 KV Cache load 必须紧密交错，async overlap 就是 kernel 的核心结构。MoE routing、paged attention 和 speculative decoding 也属于这一类 irregular workload。AMD 可以用 software pipeline 重建 overlap，但需要更多 hand-tuning，且通常落后于 NVIDIA 新 hardware primitive 的支持节奏。</p>
<p>NVIDIA 的矩阵指令抽象在各代中得到了进一步发展（warp → warp-group → 单 thread 异步 + 集群），而 AMD 却没有跟进。</p>
<h5 id="memory">内存</h5>
<p>AMD 的内存层次结构 <em>更少</em> 比 NVIDIA 的通用层，具有 NVIDIA 根本没有的巨型缓存。从 CU 向外：一个 64 KB <a data-popup="lds">LDS</a> scratchpad（软件管理、32 个库、AMD 的 NVIDIA 的 <a data-popup="l1-shared-memory">SMEM</a>)，一个 vector L1（早期 CDNA 上为 16 KB，从 <a data-popup="mi300x">MI300X</a> 开始为 32 KB），每<a data-popup="xcd">XCD</a> 几 MB 的 L2。不过，L2 在 XCD 之间并不一致；一致性发生在 L2 之上的一层。</p>
<p>该层是 <a data-popup="infinity-cache">Infinity Cache</a>：MI300X 上为 256 MB，分布在四个 <a data-popup="iod">IOD</a>，16 路组关联，测得约 12 TB/s，是 MI300X 5.3 TB/s 的两倍多 <a data-popup="hbm3">HBM3</a>。它起源于 RDNA 游戏 GPU，以补偿狭窄的 GDDR 总线； AMD 在 CDNA 3 上重用了 AI 的 IP，其中注意力 KV 重用和权重重用非常适合大型 LLC。 NVIDIA 押注于更大的 HBM 带宽（ <a data-popup="b200">B200 上为 8 TB/s</a>，在 <a data-popup="hbm4">Rubin 上使用</a> HBM4 <a data-popup="rubin">进行缩放</a>），AMD 则押注于缓存。</p>
<p>片外 HBM 容量大幅增长： <a data-popup="mi100">MI100 上的 32 → 64 → 128 → 192 → 256 → 288 GB</a> / <a data-popup="mi210">MI210</a> / <a data-popup="mi250x">MI250X</a> / <a data-popup="mi300x">MI300X</a> / <a data-popup="mi325x">MI325X</a> / <a data-popup="mi350x">MI350X</a>，从 2021 年起的每一代产品均达到或超过当代 NVIDIA 旗舰产品。押注是推理工作负载越来越受到容量限制，并且具有更多内存的芯片会获胜。</p>
<h5 id="numerics">数值格式</h5>
<p>格式轨迹跟踪 AI 芯片中每个人共享的精度减半模式：FP32 → FP16 → FP8 → FP4，每一步恢复精度更细粒度的缩放。 AMD 特定的轴是 <em><strong>开放性</strong></em>。 CDNA 4 的 <a data-popup="fp4">FP4</a> 和 FP6 使用 <em><strong><a data-popup="block-scale-multiplication">OCP MX</a> 块级乘法</strong></em>：与 <a data-popup="b200">Blackwell</a>的 MXFP4 和 TPU v8 的 MXU 相同的数字格式，但由开放联盟（AMD、NVIDIA、Intel、Meta、Microsoft、高通、ARM）是 AMD 帮助创建的，而不是由任何单一供应商创建的。 MI355X 中提供的格式与 B200 和 TPU v8 中提供的格式相同。</p>
<p>CDNA 4 变形值得拥有自己的行：每 CU FP64 吞吐量减半。 <a data-popup="mi300x">MI300X</a> 集训练、HPC、推理为一体； <a data-popup="mi355x">MI355X</a> 首先是一颗 AI 芯片。为 <a data-popup="frontier">Frontier</a> 提供动力的全速率 FP64 矩阵押注尚未被淘汰，但它不再承载权重。</p>
<h5 id="chiplets">Chiplet</h5>
<p>在包装上，CDNA 不再像 NVIDIA，而是开始变得不同。</p>
<p>CDNA 1 的 <a data-popup="mi100">MI100</a> 是整体式的 7 纳米。 CDNA 2 的 <a data-popup="mi250x">MI250X</a> 是 AMD 首款多芯片 GPU：两个 <a data-popup="aldebaran">Aldebaran</a> GCD 并排位于 2.5D EFB 有机基板上，由 4 个封装内 <a data-popup="infinity-fabric">Infinity Fabric</a> 链路连接，聚合速度为 400 GB/s，但作为两个单独的 GPU 提供给软件。</p>
<p>CDNA 3 是改变一切的举措。八个 <em><strong><a data-popup="xcd">XCD</a></strong></em> （TSMC N5，每个约为 115 mm²）通过 <em><strong><a data-popup="soic">TSMC SoIC 以 3D 方式堆叠</a></strong></em> 混合键合（亚微米间距 <em><strong><a data-popup="tsv">TSV</a></strong></em>，无微凸块）到四个 <em><strong><a data-popup="iod">I/O 芯片</a></strong></em> （台积电 N6）如下。 IOD 承载 <a data-popup="infinity-cache">Infinity Cache</a>、 <a data-popup="hbm3">HBM3</a> PHY、Infinity Fabric 链路和 <a data-popup="pcie-gen5">PCIe Gen 5</a>；每个 IOD 上面托管两个 XCD，旁边托管两个 HBM 堆栈。四个 IOD 由 <em><strong><a data-popup="infinity-fabric-ap">Infinity Fabric AP</a></strong></em> 以 4.8 TB/s 的平分速度拼接而成，因此 1530 亿个晶体管封装对于内核来说就像一个 GPU：缓存和地址空间在 IOD 层统一。 NVIDIA 在 <a data-popup="h100">H100</a> 中保持整体性，并且在 <a data-popup="b200">B200</a> 上仅采用两个十字线限制芯片 2.5D <a data-popup="cowos">CoWoS-L</a>。 AMD 更早地在一代上实现了 3D 堆叠，每芯片面积更小：在同一封装前沿上有不同的选择。</p>
<p> <em><strong><a data-popup="mi300a">MI300A</a> <a data-popup="apu">APU</a></strong></em> 进一步推动了押注。将 8 个 XCD 中的 2 个替换为 3 个 Zen 4 <em><strong><a data-popup="ccd">CCD</a></strong></em>，保持 HBM 和 Infinity Cache 以及 IOD 完好无损，并让 CPU 和 GPU 共享一个由 HBM3 支持的物理地址空间，并具有硬件一致性。没有主机设备副本。没有固定的内存。路径中没有 PCIe。 Zen 4 核心和 CDNA 3 XCD 从同一页面读取。 NVIDIA 的 <a data-popup="gh200">Grace-Hopper</a> 桥 <em></em> NVLink-C2C <a data-popup="nvlink-c2c">上有两个</a>包； MI300A 是 <em>一个</em>。 <em><strong><a data-popup="el-capitan">El Capitan</a></strong></em> （11,039 个节点，4× MI300A) 的部署证明了它的合理性。</p>
<p>在 CDNA 4 的 <a data-popup="mi355x">MI355X</a>上，八个 <a data-popup="xcd">XCD</a> 仍然通过 <a data-popup="soic">SoIC</a> 3D 堆叠到下面的基片上，但 XCD 转移到 TSMC N3P 每个有 32 个活动 CU（总共 256 个，而上层有 304 个） <a data-popup="mi300x">MI300X</a>;对于更大的 <a data-popup="matrix-core">Matrix Core</a> 和 160 KB <a data-popup="lds">LDS</a>），每个 XCD 数量下降到可用区域。四个 MI300X <a data-popup="iod">IOD</a> 在 TSMC N6 上折叠为两个，每个宽度是 TSMC N6 的两倍，上面托管四个 XCD，旁边托管四个 <a data-popup="hbm3e">HBM3E</a> 堆栈。现在，每个 IOD 都拥有自己的 256 MB <a data-popup="infinity-cache">Infinity Cache 中的 128 MB 切片</a>、一半的 HBM PHY、 <a data-popup="infinity-fabric">Infinity Fabric</a> 链接的份额以及 <a data-popup="pcie-gen5">PCIe Gen 5</a>. <a data-popup="infinity-fabric-ap">两个 IOD 之间的 Infinity Fabric AP</a> 以 5.5 TB/s 等分运行（比 CDNA 3 高约 15%），八个堆栈转移到 12-Hi HBM3E，以 8 TB/s 的速度提供 288 GB，比容量高 50% MI300X 具有相同的引脚数。该软件包总共有 1,850 亿个晶体管，并且仍然以一个 GPU 的形式呈现给内核。</p>
<h5 id="bets">设计取舍</h5>
<ul>
<li><em><strong>HPC 然后 AI。</strong></em> HPC 和 AI 是同一个押注 <em>直到它们不是</em>：从 CDNA 2 到 CDNA 3 传送全速率 FP64 矩阵，然后一旦推理经济学决定性地支持低精度，就在 CDNA 4 处分叉。</li>
<li><em><strong>内存</strong></em> 自 2021 年以来的每一代 HBM 容量都匹配或击败当代 NVIDIA 旗舰产品，并添加 256 MB 末级 <a data-popup="infinity-cache">Infinity Cache</a> 吸收 H100 必须命中 HBM 的重用 for.</li>
<li><em><strong>早期 3D 堆栈。</strong></em> 先于 NVIDIA 进行缓存和 I/O 3D 堆栈计算：台积电 <a data-popup="soic">SoIC</a> 2023 年在 IOD 上采用混合键合 XCD，而 NVIDIA 一直保持单一架构直至 2025 年。</li>
<li><em><strong>一致的 CPU+GPU。</strong></em>  <a data-popup="mi300a">MI300A</a> APU 是有史以来对小芯片攻击性最强的产品， <a data-popup="el-capitan">El Capitan</a> 部署是证明。</li>
<li><em><strong>开放扩展结构。</strong></em> <a data-popup="ualink">UALink</a> 和 OCP MX over <a data-popup="nvlink">NVLink</a> 和专有 FP4。</li>
</ul>
<h4 id="scaling">扩展</h4>
<p>内存押注会产生缩放结果：当 8 <a data-popup="mi300x">MI300X 时</a> 一个芯片可容纳 1.5 TB 的 HBM，8 个 <a data-popup="mi350x">MI350X</a> 芯片可容纳 2.3 TB，您可以在其中安装 405B 参数型号 <a data-popup="fp8">FP8</a> 位于单个 8-GPU 盒子内（权重、KV 缓存以及用于更长上下文和更大批量的余量），其中相同模型在 8× <a data-popup="h100">H100</a> (640 GB) 上需要仔细分片。对于 2024 年至 2025 年的推理工作负载，AMD 的扩展不需要在机架上与 NVL72 相匹配即可在盒子上具有竞争力。为了 <em>在前沿进行训练</em> 确实如此，AMD 直到 2026 年才找到答案。</p>
<div class="definitions">
<div class="definition"><div class="definition-term">纵向扩展</div><div class="definition-body">将 GPU 绑定到一个一致的内存域中 <a data-popup="infinity-fabric">Infinity Fabric</a>。通过 <a data-popup="mi355x">MI355X</a> 这会停止在 8-GPU <a data-popup="oam-ubb">OAM</a> 框（每个 GPU 896 GB/s 网格）。 <a data-popup="helios">Helios</a> 扩展通过 <a data-popup="ualink">UALink</a>连接到 72-GPU 机架，在启动时通过以太网隧道传输 (<a data-popup="ualoe">UALoE</a>)，并且来自 2027 年。</div></div>
<div class="definition"><div class="definition-term">横向扩展</div><div class="definition-body">通过以太网将这些域联网。无 <a data-popup="infiniband">InfiniBand</a>。 <a data-popup="pensando">Pensando</a> NIC（<a data-popup="pollara">Pollara 400</a>、 <a data-popup="vulcano">Vulcano 800</a>）实施 <a data-popup="uec">超以太网联盟</a>的 <a data-popup="uet">UET</a> <a data-popup="rdma">RDMA</a> 传输； <a data-popup="tomahawk-6">Broadcom Tomahawk 6</a> 提供 <a data-popup="switch-asic">交换机 ASIC</a> 和 <a data-popup="cpo">CPO</a>。</div></div>
</div>
<h5 id="scale-up">纵向扩展</h5>
<p>通过 MI355X，AMD 的规模化意味着 <strong>8-GPU <a data-popup="oam-ubb">OAM</a> 平台</strong> 超过 <a data-popup="infinity-fabric">Infinity 面料</a>。每个 <a data-popup="mi300x">MI300X</a> 都有 7 个 IF 链路（一个到盒子中的每个对等点），双向 128 GB/s，在完全连接的全对所有拓扑中提供 896 GB/s 的每 GPU 网格带宽。 <a data-popup="mi350x">MI350X</a> 将每个链接提升至 153.6 GB/s（每个 GPU 约 1,075 GB/s），但保持 8-GPU 形状。该平台符合 OCP 的 UBB 2.0：与 NVIDIA HGX 基板相同的机械插槽，因此服务器供应商可以在同一机箱上搭载 AMD 或 NVIDIA，而无需重新设计系统。</p>
<p>AMD 没有通过 MI355X 推出的是相当于 NVL72 的机架级产品。在 MI300X 集群上运行较大模型的客户可以通过以太网在多个 8-GPU 盒子之间进行扩展，为 NVIDIA 用户可以保留在内部扩展中的内容支付扩展延迟的费用。这是对训练至关重要的差距， <em><strong><a data-popup="helios">Helios</a></strong></em> 就是为了弥补这一差距而设计的。</p>
<p><img alt="AMD Helios — 72 个 MI455X GPU 位于开放式机架宽机箱中的一排 UALink 交换机下方，连接到一个连贯的 UALink 内存域。发布时，该结构在 UALoE（以太网隧道式无限结构）上运行，作为 2027 年原生 UALink 交换芯片上市之前的权宜之计。每个 GPU 都将配备 Pensando Vulcano 800 NIC。" loading="lazy" src="images/amd-scale-up.png"/></p>
<p>Helios 是 AMD 首款机架级扩展域，将于 2026 年 2 月与 <a data-popup="mi455x">MI455X</a>一起发货。每机架 72 个 GPU，约 31 TB <a data-popup="hbm4">HBM4</a>，1.4 PB/s 聚合 HBM 带宽，2.9 ExaFLOPS FP4 / 1.4 ExaFLOPS FP8，260 TB/s 纵向扩展带宽，43 TB/s 横向扩展。外形尺寸为 <strong><a data-popup="orw">开放式机架宽 (ORW)</a></strong> （Meta 的 2025 年 OCP 提交，双宽和液冷），而不是 AMD 专有机箱。基于 Meta 的参考设计而不是从头开始设计机架是 AMD 的一个深思熟虑的押注：任何在 ORW 上标准化的超大规模企业都可以部署 Helios，而无需定制数据中心设施。</p>
<p>结构是 <em><strong><a data-popup="ualink">UALink</a></strong></em>：Ultra Accelerator Link 是 AMD 与 Apple、AWS、Cisco、Google、HPE、Intel、Meta、Microsoft 和 Synopsys 共同创建的开放联盟标准。 UALink 200G 1.0（2025 年 4 月）定义了 200 GT/s lane 和每个方向 800 Gbps，交换拓扑可扩展到每个 Pod 1,024 个加速器。其承诺是可与 NVLink 相媲美的缓存一致性互连，但无人拥有：任何供应商都可以构建 UALink 交换机，任何加速器都可以与 UALink 对话，该标准属于联盟而不是最强大的卖家。</p>
<p>问题： <strong>原生 UALink 交换芯片在批量发货之前不会批量发货 2027 年</strong>。 Astera Labs 的 Scorpio 以及 Auradine、Enfabrica 和 Xconn 的竞争部件都计划在 2026 年末/2027 年部署。 Helios 在发布时使用 <em><strong><a data-popup="ualoe">UALoE</a></strong></em> （通过标准以太网隧道传输的 Infinity Fabric）作为权宜之计，在等待原生 UALink 结构时保留编程模型。原生 UALink 交换将于 2027 年随 MI500 一起推出。在发布时，Helios 更接近于快速以太网隧道一致集群，而不是 NVL72 真正的缓存一致 NVLink 域：时间线上的真正让步，以换取 2026 年 2 月推出具有竞争力的产品。</p>
<h5 id="scale-out">横向扩展</h5>
<p>AMD 不提供 <a data-popup="infiniband">InfiniBand</a>。整个横向扩展堆栈是以太网，基于不同的开放标准： <em><strong><a data-popup="uec">超以太网联盟 (UEC)</a></strong></em>。</p>
<p>UEC 1.0（2025 年 6 月发布）定义 <em><strong><a data-popup="uet">超以太网传输 (UET)</a></strong></em>：标准以太网上的新 RDMA 传输，具有数据包喷射、基于 SACK 的选择性重传和现代拥塞控制。 UET 不是 RoCEv2（它将 InfiniBand 传输封装在以太网帧中）；它是针对横向扩展 AI 结构的 RDMA 语义的彻底重新设计。 AMD 与博通、思科、Meta 和微软一样是创始成员。与 UALink 相同：拥有标准，而不是实现。</p>
<p><img alt="AMD 横向扩展 — Helios 机架通过基于开放式超以太网 (UEC) 标准的标准以太网相互通信。 UET 通过彻底重新设计 RDMA 语义来取代 RoCEv2。每个 GPU 均配备 Pensando Vulcano 800 NIC（PCIe Gen 6、800 GbE、UEC 1.0）；机架间交换采用具有共同封装光学器件的 Broadcom Tomahawk 6。 AMD 拥有 NIC 层，交换机和光学器件是合作伙伴芯片。" loading="lazy" src="images/amd-scale-out.png"/></p>
<p>网卡是 <em><strong><a data-popup="pensando">Pensando</a></strong></em>，AMD 于 2022 年收购的网络初创公司。 <em><strong><a data-popup="pollara">Pollara 400</a></strong></em> 是当前的 AI NIC：400 GbE、P4 可编程、UEC 就绪、PCIe Gen 5，与 MI300X / MI355X 配对。 <em><strong><a data-popup="vulcano">Vulcano 800</a></strong></em> 将于 2026 年与 MI455X 一起发货：符合 UEC 1.0 标准、PCIe Gen 6、原生 UALink 接口、每 GPU 横向扩展带宽是 Pollara 的 8 倍。 <em><strong><a data-popup="salina">Salina 400</a></strong></em> 是用于存储/SDN/防火墙的前端 DPU（16× Arm Neoverse-N1，双 400 GbE），相当于 NVIDIA 的 <a data-popup="bluefield-dpus">BlueField</a>，与 AI 后端网卡不同。</p>
<p>不过，开关芯片不是 AMD 的。 Helios 的 43 TB/s 横向扩展结构通过 <em><strong><a data-popup="tomahawk-6">Broadcom Tomahawk 6</a></strong></em>运行：具有共同封装光学器件的 102.4 Tbps 以太网交换机 ASIC（“Davisson”）。 AMD 没有内部 <a data-popup="cpo">CPO</a> ，也没有内部交换机 ASIC；光学层是伙伴硅。 NVIDIA 拥有整个堆栈：InfiniBand、Spectrum-X 以太网、ConnectX、BlueField、Quantum-X Photonics CPO，全部为内部产品。 AMD 拥有一层（NIC + DPU，通过 Pensando），并押注开放标准加上同类最佳的合作伙伴芯片将超过垂直整合。</p>
<p>行业已经改变了 AMD 的方式。 Dell'Oro 报告称，到 2025 年，以太网处理的 AI 横向扩展结构容量将是 InfiniBand 的两倍以上； AWS、微软、Meta、Oracle 和 xAI 都已针对其基于 AMD 的 AI 集群进行了以太网标准化。剩下的问题不是以太网是否可以在 RDMA 语义上与 InfiniBand 相匹配（UEC 缩小了这一差距），而是 Helios 是否可以足够快地缩小 <em>机架规模</em> 与 NVL72 的差距，从而赢得目前默认由 NVIDIA 承担的前沿训练工作负载。</p>
<h4 id="software">软件栈</h4>
<p><em><strong><a href="https://rocm.docs.amd.com/">ROCm</a></strong></em> 是与 <em><strong><a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/">CUDA</a></strong></em>相对应的开源版本。 NVIDIA 的堆栈是专有且垂直集成的（cuBLAS、cuDNN、TensorRT-LLM 作为二进制 blob 提供，由 NVIDIA 单独维护），而 ROCm 是 GitHub 原生的，并押注于开放标准（PyTorch、Triton、vLLM、OCP MX），而不是围墙花园库集。与 NVIDIA 之间的软件差距确实存在，但 AMD 的策略是通过开放社区来缩小差距，而不是从头开始构建并行 CUDA 堆栈。</p>
<p>堆栈的底部是 <em><strong><a data-popup="hip">HIP</a></strong></em>，AMD 的 CUDA 兼容 C++ 运行时。 <em><strong><a data-popup="hipify">hipify</a></strong></em> 自动将 CUDA 源转换为 HIP。批量 HPC 代码（HACC、Laghos、QMCPack）端口开箱即用率为 80–95%：CORAL-2 编号。modern AI kernel 的移植更糟糕：任何涉及 Hopper 或 Blackwell 特定原语的内容（<a data-popup="tma">TMA</a> 描述符、 <a data-popup="wgmma"><code class="notranslate" translate="no">wgmma</code></a>、 <code class="notranslate" translate="no">tcgen05.mma</code>) 没有干净的 ROCm 模拟，必须手动重写。</p>
<p>HIP 之上有一个库层，其结构与 NVIDIA 的镜像相同，按名称一对一： <em><strong><a href="https://github.com/ROCm/rocBLAS">rocBLAS</a></strong></em> for cuBLAS; <em><strong><a href="https://github.com/ROCm/hipBLASLt">hipBLASLt</a></strong></em> 用于 cuBLASLt； <em><strong><a href="https://github.com/ROCm/MIOpen">MIOpen</a></strong></em> 用于 cuDNN； <em><strong><a href="https://github.com/ROCm/rccl">RCCL</a></strong></em> 用于 NCCL； <em><strong><a data-popup="composable-kernel">可组合内核</a></strong></em> （及其现代 <a data-popup="ck-tile">ck-tile</a> DSL) 用于 CUTLASS；适用于 Nsight 系列的 rocprofv3 / rocprof-sys / rocprof-compute。不过，目前还没有 TensorRT-LLM 的第一方类似产品。 AMD 的答案是支持 <em><strong><a href="https://github.com/vllm-project/vllm">vLLM</a></strong></em> 作为开源服务引擎，并提供 AMD 特定的运算符 (<em><strong><a data-popup="aiter">AITER</a></strong></em>)进入其中； vLLM 专用的 ROCm CI 在 2026 年初将测试通过率从 37% 提高到 93%。</p>
<p>PyTorch 在 ROCm 上属于 first-class path。Eager-mode PyTorch 很早就已支持 ROCm；<code class="notranslate" translate="no">torch.compile</code> 通过 Triton 生成 kernel，Triton 的 ROCm backend 也已进入 upstream。ROCm 不采用 XLA 式 whole-graph IR，而是直接 lower 到 HIP、Triton 或 Composable Kernel。随着 Triton 成为 PyTorch 的主流 kernel 路径，许多 <code class="notranslate" translate="no">torch.compile</code> kernel 可以在 CUDA 与 ROCm 之间复用。AMD 的设计取舍是：让 Triton Python DSL 成为跨厂商的公共 kernel language，从而缩小 CUDA 生态带来的迁移成本。</p>
<p><em><strong><a data-popup="flashattention-versions">FlashAttention</a></strong></em> 为承重案例。 <em><strong>FA2</strong></em> 正在通过可组合内核在 MI300X 上进行生产； PyTorch 在 ROCm 上默认为 CK 或 AOTriton。 <em><strong>FA3</strong></em> （Hopper 调整）通过 AITER + CK 部分支持，但 Dao-AILab 的规范实现仍然仅支持 CUDA。 <em><strong>FA4</strong></em> （Blackwell，2026 年 3 月）根本没有 ROCm 端口。 <em><strong><a href="https://hazyresearch.stanford.edu/blog/2025-11-09-hk">HipKittens</a></strong></em>是 Hazy Research 的 ThunderKittens MI355X 端口（2025 年 11 月），声称在约 500 行中与手动调整的 AITER 具有前向传递同等性。其模式是：开源学术内核在 NVIDIA 的几个月而不是几年后关闭 AMD 的尾巴。</p>
<p>生产部署已经验证了该策略。 Microsoft Azure 的 <em><strong>ND MI300X v5</strong></em> 实例于 2024 年 5 月正式发布； OpenAI 对它们运行 GPT 推理。 Meta 通过 Grand Teton 平台在 MI300X 上提供 Llama 3 / Llama 4 推理。 Oracle OCI 的 <em><strong>BM.GPU.MI300X.8</strong></em> 于 2024 年 9 月进入通用版本，MI355X 将于 2026 年上市。这些是超大规模的真实服务队列，而不是试点。</p>
<p>诚实的差距仍然是真实的。独立基准测试（Phoronix，2026 年 3 月）显示，ROCm 7.2 在标准 PyTorch / vLLM / SGLang 工作负载上的 <strong>比同等 CUDA</strong> 慢 10-25%，且在同等芯片上具有同等精度。 ROCm 达到 7 <em>功能奇偶校验</em> ，但不是 <em>性能奇偶校验</em>。 FlashAttention-4 尾部（利用 Blackwell 最新原语的研究代码）是 NVIDIA 护城河最持久的地方；它没有干净的 ROCm 模拟，等待手写的 AITER 内核或 HipKittens 级社区端口。 NVIDIA 将工程师派往前沿实验室； AMD 通过 GitHub 发布内核。这些策略集中于常见工作负载（Llama 推理、注意力、密集 Transformer 训练），但新颖的研究代码的长尾仍然需要花费 MI300X / MI355X 部署工程时间，而 NVIDIA 用户无需支付费用。</p>
<hr/>
<h3 id="cerebras-wse">Cerebras WSE</h3>
<div class="philosophy">
<p><strong><a href="https://www.cerebras.ai/">Cerebras</a></strong> 制造了目前规模最大的商用 processor。它的出发点是：memory wall 很大程度来自把 wafer 切成独立 die，再用 HBM、interposer、NVLink 和大量 cable 把这些 die 重新连接。Cerebras 选择不切 wafer。Wafer-Scale Engine（WSE）把 84 个 reticle field 连接成一整块 46,225 mm² silicon，其中包含 900,000 个 dataflow core；每个 core 都紧邻 local SRAM，使 data movement 保持在 wafer 内部。</p>
</div>
<h4 id="genealogy">演进路线</h4>
<div class="genealogy">
<div class="gen-row"><div class="gen-year">2019</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://old.hotchips.org/hc31/HC31_1.13_Cerebras.SeanLie.v02.pdf">WSE-1</a></em></strong><span class="gen-chip"><a data-popup="cs-1">CS-1</a></span></div><div class="gen-desc">首款晶圆级处理器：1.2T 晶体管， 400,000 个 core，18 GB 晶圆上 SRAM。</div></div></div>
<div class="gen-row"><div class="gen-year">2021</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://8968533.fs1.hubspotusercontent-na2.net/hubfs/8968533/IEEE%20Micro%202023-03%20Hot%20Chips%2034%20Cerebras%20Architecture%20Deep%20Dive.pdf">WSE-2</a></em></strong><span class="gen-chip"><a data-popup="cs-2">CS-2</a></span></div><div class="gen-desc">7 nm：850,000 个 core，40 GB SRAM。 <strong>权重流</strong> 将权重从晶圆移至 <a data-popup="memoryx">MemoryX</a>。</div></div></div>
<div class="gen-row"><div class="gen-year">2023</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://www.hpcwire.com/aiwire//2023/08/30/cerebras-and-g42s-inception-unveil-jais-a-13b-parameter-arabic-llm-trained-on-condor-galaxy/">Condor Galaxy</a></em></strong><span class="gen-chip"><a data-popup="condor-galaxy">CG-1</a></span></div><div class="gen-desc">使用 <a data-popup="g42">G42</a>构建的 64 系统集群；培训了 <a data-popup="jais">Jais</a> Arabic LLM family。</div></div></div>
<div class="gen-row"><div class="gen-year">2024</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://www.cerebras.ai/press-release/cerebras-announces-third-generation-wafer-scale-engine">WSE-3</a></em></strong><span class="gen-chip"><a data-popup="cs-3">CS-3</a></span></div><div class="gen-desc">5 nm：4T 晶体管，900,000 个 core，44 GB SRAM；每个 core FP16 SIMD 加倍至 8 宽；指定到 2,048 个系统的集群。</div></div></div>
<div class="gen-row"><div class="gen-year">2024</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://www.cerebras.ai/blog/introducing-cerebras-inference-ai-at-instant-speed">推理</a></em></strong></div><div class="gen-desc">权重存放在 SRAM 中而不是流式传输：业界最快的独立测量 decode，也是现在定义公司的关键。</div></div></div>
</div>
<h4 id="architecture">架构</h4>
<p>GPU 是层层嵌套的体系：thread 组成 warp，warp 运行在 SM 上，SM 位于 die 和 package 内，package 再组成 node 与 rack；每一层都有自己的带宽、延迟和编程边界。WSE 则更像一张平面：900,000 个小 core 在二维 mesh 上首尾相连，没有 shared cache，也没有传统意义上的 global memory。每个 core 都很小：包含 48 kB local SRAM、16 个 GPR、一个 6-stage pipeline、一个 FP16 FMAC SIMD，以及连接 fabric 的五端口 router。执行采用 dataflow model：core 平时保持 idle，直到 <a data-popup="wavelet">wavelet</a> 到达；wavelet 中的控制位决定触发哪个 task，8 个 hardware microthread 根据 operand 是否就绪逐周期切换。这里没有 warp、warp scheduler、cache miss 或 reorder buffer——数据到达本身就是调度。</p>
<p><img alt="Cerebras WSE-3 — 左：晶圆，在 12×7 网格中具有 84 个 reticle field，平铺最大的正方形，适合 300 毫米，划线接缝完好无损，芯片边缘有 12×100 GbE 条带作为唯一的出路。右：放大的一个 reticle field，一个均匀的 2D 核心网格，其链接以每个芯片 2,880 GB/s 的速度穿过金属中的划线边界，因此软件可以看到一个没有接缝的 900,000 个核心结构。" loading="lazy" src="images/cerebras-wafer-die.png"/></p>
<p><img alt="放大到一个 Cerebras 核心 — 一个具有 24 种颜色静态路由的五端口光纤路由器，为运行 8 个 microthread 的数据流 task scheduler 提供数据；下面是 GPR 和 44 个张量描述符寄存器、FMAC SIMD 计算引擎旁边的 8 个单周期存储体中的 48 kB 本地 SRAM，以及收集非 structured sparsity 的发送方零过滤器。" loading="lazy" src="images/cerebras-core.png"/></p>
<h5 id="the-wafer">晶圆</h5>
<p>步进机曝光晶圆一个 <a data-popup="reticle">标线</a> 一次，每次拍摄约 850 mm²，这就是为什么每个传统芯片都生活在这个上限之下（以及为什么 B200 变成 <a data-popup="two-die-chiplet-gpu">两个芯片</a> NVIDIA 向其施压的那一刻）。与台积电的任何其他客户一样，Cerebras 在 12×7 网格中打印相同的约 550 mm² 芯片 84 次，然后在与台积电共同开发的工艺中，在锯正常运行的 &lt;1 毫米 <a data-popup="scribe-line">划线</a> 上铺设额外的高级金属。网格穿过源同步并行接口上的每个接缝（WSE-3 上每个芯片为 2,880 GB/s），整个芯片间层的成本约为 97 W。对于软件来说，接缝不存在：一个统一的网格，一个芯片。</p>
<p>晶圆级之前已经尝试过，但在良率上失败了：单片晶圆计算机中的单个缺陷会杀死整个晶圆，这是是什么在 20 世纪 80 年代埋葬了 <a data-popup="wafer-scale-integration">这个想法</a> 。 Cerebras 的答案是粒度。 H100 上的缺陷会导致整个 ~6 mm² SM 失效； WSE 上的相同缺陷会导致一个 0.05 mm² 核心失效。 WSE-3 制造了约 970,000 个核心，并交付了 900,000 个：约 7% 的备用池，加上冗余结构链路，使硬件能够围绕每个缺陷重新映射并恢复完整的逻辑网格。</p>
<h5 id="the-core">核心</h5>
<p>Cerebras core 最特别的地方不是 datapath，而是 instruction 如何描述数据。除 16 个 GPR 外，每个 core 还有 44 个 <strong><a data-popup="dsr">Data Structure Register（DSR）</a></strong>，每个 DSR 保存一个最多四维的 tensor descriptor，包括 base address、extent 和 stride。instruction 通过 DSR 引用 operand，因此一条 FMAC instruction 就能表达“把到达的数据流与驻留 tensor 相乘并累加”，hardware 会持续处理整个 tensor，而不需要 software loop 或 element-wise instruction fetch。NVIDIA 花了多代 Tensor Core 才把 matmul 推向 descriptor-driven command；在 WSE 中，tensor instruction 从一开始就是这种形式。</p>
<p>execution ordering 由 fabric 完成。<a data-popup="fabric-color">color</a> 是编译期静态路由的 virtual channel，并绑定到目标 core 上的一个 handler task；向某个 color 发送 wavelet，效果就像调用目标 core 上的代码，其中 control bits 表示调用类型，data bits 携带参数。task scheduler 在 8 个 hardware microthread 之间切换，根据 operand 是否 ready 选择下一项工作。它和 GPU warp scheduler 都在隐藏 stall，只是 WSE 需要隐藏的是 local SRAM bank conflict 或邻居 hop，而不是 HBM round trip。</p>
<p>48 kB 本地 SRAM 是为数据路径而不是局部性而组织的：8 个单端口 6 kB 存储体每个周期提供两次 64 位读取和一次 64 位写入，恰好有两个 4 元素 FP16 operand 输入和一个结果输出，即 WSE-2 FMAC 的宽度。 256 字节软件管理的缓存（WSE-3 上为 512 B）将最热的值保留在管道旁边。这是机器原理的缩影：每个核心、内存带宽和计算能力完全匹配，而晶圆继承了这一平衡 900,000 倍。</p>
<h5 id="compute">计算</h5>
<p>WSE 没有独立的 matrix unit。NVIDIA、Google 和 AMD 把主要 FLOPs 集中到 Tensor Core、MXU 或 Matrix Core；Cerebras 则利用 2D fabric 组合出 matmul。执行 GEMM 时，weight 沿一排保存 activation 的 core 广播，每个 core 对自己的 local slice 做 FMAC，partial sum 再沿 mesh reduction。Tensor Core 依靠 register tile 做 data reuse，MXU 依靠 systolic wiring，WSE 则依靠 spatial placement：activation 保持原地不动，只有 weight 和 partial sum 在 fabric 中流动。</p>
<p>解读 Cerebras 的 FLOPs 需要区分 sparse 与 dense。WSE-3 宣传的 125 PFLOPS 指 sparse FP16，并假设理想 workload 能通过 zero skipping 获得约 8× gain。按 900,000 core × 8-wide FMAC × 1.1 GHz 推算，dense FP16 约为 15.8 PFLOPS；Cerebras 没有单独公布这个数字。WSE 的重点本来也不是每瓦 dense FLOPs，而是 SRAM bandwidth：compute datapath 的规模主要是为了跟上 21 PB/s on-wafer SRAM。</p>
<p>零跳跃是数据流赢得其保留的地方。因为计算是由到达的数据触发的，所以零永远不会触发任何东西： <em><strong>零在发送者处被过滤</strong></em>，并且接收核心永远不会看到它们并且永远不会花费周期。这是非结构化的元素粒度稀疏性，是 NVIDIA 2:4 <a data-popup="structured-sparsity">structured sparsity</a> 仅采样的一般情况。到目前为止，这也是一个尚未行使的选择权。 Cerebras 自己的稀疏预训练结果（<a href="https://arxiv.org/abs/2303.10464">SPDF</a>：1.3B 参数稀疏度为 75%；后续为 6.7B）是供应商编写的且低于 7B，并且没有透露旗舰客户模型经过稀疏训练： <a data-popup="jais">Jais 2</a>，硬件上最大的运行，是密集的。唯一能够获得非 structured sparsity 的芯片尚未推出使用它的头条模型。</p>
<h5 id="memory">内存</h5>
<p>WSE 的 memory hierarchy 几乎只有一层：44 GB SRAM 分散在 900,000 个 core 内，每个 core 约 48 kB。没有 HBM、L2 或 eviction policy，local SRAM 可以单 cycle 访问。21 PB/s 是所有 local SRAM port 的 aggregate bandwidth，并不是 point-to-point link bandwidth，因此不能直接与 HBM 数字比较。更有意义的指标是 bytes per FLOP：按 dense FP16 估算，WSE 每 FLOP 可获得约 1.3 byte，而 B200 从 HBM 获得的约为 0.002 byte。decode 需要为每个 token 读取整套 weight，天然 bandwidth-bound，因此最能体现 WSE 的优势。</p>
<p>这层超高带宽 SRAM 的边界也很明显：整片 wafer 对外只有 12×100 GbE，也就是 1.2 Tb/s。on-wafer SRAM 与 off-wafer Ethernet 之间相差约五个数量级。GPU hierarchy 的带宽通常逐层下降，而 WSE 是 wafer 内极快、wafer 外突然变慢。换句话说，wafer 既是一座高带宽 island，也被自己的边界限制。</p>
<p>另一个限制是 SRAM density 的 scaling 已明显放缓。即使 WSE-3 换用更先进 process node、transistor count 增加约 54%，SRAM capacity 也只比 WSE-2 增长约 10%。logic 仍能随着 process 缩小，但 6T SRAM cell 的收益有限，因此下一代 process 不会自动带来同比例的 on-wafer capacity 增长。</p>
<h5 id="weight-streaming">权重流式传输</h5>
<p>晶圆上的训练颠倒了其他人认为理所当然的流程：在 GPU 或 TPU 上，权重是驻留的，激活流通过；在 WSE 上， <em><strong>激活是常驻的，权重通过</strong></em>流动。主权重位于 <em><strong><a data-popup="memoryx">MemoryX</a></strong></em>中，它是集群旁边的 DRAM 和闪存设备。权重逐层流过晶圆，针对固定在 SRAM 中的激活触发乘法累加，然后离开；梯度流在后向传递中返回，优化器步骤在 CPU 上的 MemoryX 内部运行（权重更新是 O（参数）的 element-wise 工作，没有重用，因此 CPU 级计算保持同步）。晶圆从不存储权重，“即使是暂时的”（<a href="https://www.kisacoresearch.com/sites/default/files/documents/cs_weight_streaming_white_paper_-_cerebras.pdf">Cerebras 的短语</a>）。模型大小受 MemoryX 限制，而不是 44 GB； 44 GB 限制激活和批处理。</p>
<p>这种 weight streaming 换来的是更简单的 programming model：一个 wafer 可以容纳完整一层的 activation，因此不需要 tensor parallel、pipeline parallel 或 FSDP sharding。70B model 可以按单设备程序编写；扩展到多个系统时，只需使用 SwarmX 做纯 data parallel——把同一份 weight stream 广播到 N 个 wafer，并在返回路径上聚合 gradient。GPU 训练里复杂的 parallelism strategy，在 Cerebras 上被大幅压缩。</p>
<p>代价体现在实际 scale 上。Cerebras 规格曾描述最多 2,048 台 CS-3，但公开披露的最大 deployment 是 64 台 Condor Galaxy 3。平台上公开的最大 from-scratch model 是 Jais 2（70B parameter、2.6T token），由 G42 与 Cerebras engineer 协同训练。另一个缺失指标是 MFU：GPU lab 通常会公开 35–45% 一类数字，但 Cerebras 尚未为这些大规模 run 披露可直接比较的利用率。</p>
<h5 id="numerics">数值格式</h5>
<p>WSE 的主计算格式是 FP16/BF16、FP32 accumulate；WSE-3 另外公开了一条 16-wide 8-bit fixed-point path，但没有 FP8、FP4 或 microscaling。其他 AI accelerator 通过降低 precision、再用 block scaling 恢复 accuracy 来提升 throughput 和 capacity，而 Cerebras 仍以 16-bit 为主，并把它作为 quality selling point。这里存在明显 tension：SRAM capacity 是 WSE 最稀缺的资源，如果能使用 8-bit weight，model 所需 wafer 数量理论上可以减半。因此，坚持 16-bit 究竟是 numerical choice 还是 datapath roadmap 的限制，仍是开放问题。</p>
<h5 id="bets">设计取舍</h5>
<ul>
<li><em><strong>不要切割晶圆。</strong></em> 芯片边界是行业其他部分支付的税：SerDes、中介层、HBM 堆栈、电缆、交换机。金属中的 Stitch 84 reticle field 和竞争对手系统中的最高带宽边界根本不存在。</li>
<li><em><strong>SRAM 是唯一的存储器。</strong></em> 以业界最陡的比率交换带宽容量：44 GB，晶圆上聚合 21 PB/s。平衡机器，而不是将不平衡隐藏在层次结构后面。</li>
<li><em><strong>数据流核心，无矩阵单元。</strong></em> 由到达的 wavelet 触发的 900,000 个微小核心，通过广播、FMAC 和网格缩减组装 matmul：跳过零是免费的，而不是特殊模式。</li>
<li><em><strong>权重移动，激活保持不变。</strong></em> 权重流将模型大小 (MemoryX) 与晶圆内存 (44 GB) 解耦，并将集群扩展压缩为纯 data parallel。</li>
<li><em><strong>销售延迟，不是吞吐量。</strong></em> 晶圆重新读取每个 token 的整个模型的速度比基于 HBM 构建的任何东西都要快；价格作为优质产品加速，而不是在每个 token 的成本上竞争。</li>
</ul>
<h4 id="scaling">扩展</h4>
<p>在 Cerebras 上，scale-up 与 scale-out 的含义和 GPU 不同。WSE 在 fab 阶段就把 900,000 个 core 做成一个 coherent on-wafer domain，因此不存在“把 72 个 package 连接成一台机器”的 scale-up 问题。真正的边界是 wafer 外部：一旦 workload 超出单片 wafer，就必须通过带宽低得多的 Ethernet 与其他系统通信。</p>
<div class="definitions">
<div class="definition"><div class="definition-term">纵向扩展</div><div class="definition-body">纵向扩展单元就是整片晶圆：900,000 个核心组成一张 2D 网格，采用 32 位链路、单周期跳转、跨 24 种 <a data-popup="fabric-color">颜色</a>的静态路由和原生广播，聚合互连带宽达 214 Pbit/s。其尺寸固定为 46,225 mm²，也就是一整片 300 毫米晶圆。</div></div>
<div class="definition"><div class="definition-term">横向扩展</div><div class="definition-body">以太网，立即：每个系统 12×100 GbE (1.2 Tb/s)。通过 <a data-popup="swarmx">SwarmX</a> （数据并行广播/减少 <a data-popup="rdma">RoCE</a>）进行扩展；推理分片模型在层边界跨系统、管道并行。</div></div>
</div>
<h5 id="scale-up">纵向扩展</h5>
<p>晶圆的内部结构没有 <a data-popup="serdes">SerDes</a>，没有电缆，没有收发器，每条链路没有边际成本：路由被编译，每一跳都是一个周期，广播是一种原生结构原语而不是交换机功能。 NVL72 花费了 5,184 根铜缆和一个 <a data-popup="nvswitch">NVSwitch 托盘</a> ASIC 为 72 个 GPU 提供 130 TB/s 的全对全，WSE 的等效域是单个光刻对象。问题是域的大小是一个常数。 NVIDIA 的扩展领域每一代都在增长（三年内从 NVL72 到 NVL576）；自 2019 年以来，晶圆尺寸一直为 46,225 mm²，并将保持这一水平。 300 毫米是业界最大的晶圆（450 毫米过渡已于十年前消亡），因此 Cerebras 的扩展路线图取决于下一个节点的密度产量：没有更多的区域可供使用。</p>
<h5 id="scale-out">横向扩展</h5>
<p>训练横向扩展 <a data-popup="swarmx">SwarmX</a>，它只做一件事：复制。将权重流广播到 N 个晶圆，减少它们在返回路径上的梯度；批次随着系统数量的增加而增加，但模型大小却不会。声称的 2,048 个系统的上限（“256 exaFLOPS”，稀疏）从未建成； 64 有。</p>
<p>推理完全放弃权重流；算术是致命的。对于每个 decode 的 token，通过 ~150 GB/s 的管道从 MemoryX 流式传输 70B 模型的 140 GB 大约每个 token 花费一秒。因此，推理 <em><strong>将权重存储在 SRAM</strong></em> 中，并在层边界跨晶圆对模型进行分片：Llama 70B 位于“少至四个”CS-3 上，通过以太网进行管道并行，每个附加晶圆贡献 44 GB 的权重加 -<a data-popup="kv-cache">KV</a> 容量和 23 kW 负载。</p>
<p>速度是真实的，并经过独立验证。 <em><strong><a data-popup="artificial-analysis">人工分析</a></strong></em> 在 2024 年 8 月发布时在 Llama 3.1 8B 上测量了 1,850 个 token/秒，在 70B 上测量了 446 个 token/秒，在 Llama 405B 上测量了 969 个 token/秒（第一个 token 需要 240 毫秒），在 2025 年在 Llama 4 Maverick 上测量了 2,522 个 token/秒，约为 Blackwell 最佳发布量的 2.4 倍的时间数。供应商引用的峰值更高（使用 <a data-popup="speculative-decoding">speculative decoding</a>的 70B 上为 2,100；GPT-OSS-120B 上为 3,000，其中实时独立测量值接近 2,000）。没有任何 GPU 提供商能够在每用户 decode 速度方面与您相媲美。</p>
<p>经济性是最显着的优势。每晶圆 44 GB 意味着前沿规模模型会消耗大量设备： <a href="https://newsletter.semianalysis.com/p/cerebras-faster-tokens-please">SemiAnalysis</a> 估计 1.6T 参数级模型需要约 24 个 CS-3，可安装在少数 GPU 机架中，每个系统分析师估计约 45 万美元的物料清单售价约为 2-300 万美元（从未正式披露）。在 decode 期间，晶圆的大量 FLOP 大部分处于闲置状态； Cerebras 拒绝透露批量大小，也从未公布过每个系统的吞吐量。对于相同的开放模型，每个 token API 的定价大约为基于 GPU 的提供商的 3-5 倍，而 Llama 405B 已悄然从 API 中删除，SemiAnaanalysis 将其视为服务于尚不明确的经济学。固定 SRAM 还对上下文进行定价：KV 缓存与权重位于相同的 44 GB 中，因此长上下文会窃取容量并迫使每个副本有更多系统； API 上限为 131K 个 token，而前沿提供商提供 256K–1M 个 token。 <a data-popup="moe">MoE</a> 提供服务（Qwen3-235B 约为 1,500 个 token/秒，供应商引用），但这是该格式最糟糕的情况：巨大的参数占用量触及了一些专家一次，保存在最昂贵的内存中。</p>
<p>市场已经诚实地定价了。 Mistral 的 Le Chat（约 1,100 个 token/秒）、Perplexity Sonar 和 Meta 的 Llama API 都为延迟付出了代价； 2026 年 1 月，OpenAI 签署了 <em><strong>到 2028 年将拥有 750 MW 的 CS-3 产能</strong></em>， <a href="https://www.cnbc.com/2026/01/14/cerebras-scores-openai-deal-worth-over-10-billion.html">据报道超过 $10B</a> 在签约时 <a href="https://finance.yahoo.com/technology/ai/articles/cerebras-systems-openai-tout-20b-040208708.html">自增长超过 $20B</a>以来，成为有史以来获得的最大晶圆代言规模。第一个搭载该容量的旗舰产品是 <em><strong><a href="https://openai.com/index/gpt-5-6/">GPT-5.6 Sol</a></strong></em>，于 2026 年 7 月推出，报价为 750 个 token/秒。</p>
<h4 id="software">软件栈</h4>
<p>Cerebras 的软件栈同样由 compiler 驱动，但开放面比 TPU 更窄。<code class="notranslate" translate="no">cerebras.pytorch</code> 先通过 lazy tensor 把 train step 捕获为 Torch-MLIR/graph IR，再把 subgraph 匹配到手写 kernel library；没有匹配项时才回退到较慢的自动生成 kernel。公开限制包括 static graph、无 dynamic shape、无 data-dependent control flow、执行过程中不能随时读取 intermediate tensor，以及固定的 PyTorch version。换句话说，它并不是普通 PyTorch code 的 1:1 drop-in backend。</p>
<p>这套 ML stack 也没有面向普通用户的 kernel escape hatch。CUDA 可以自己写 kernel，TPU 有 Pallas，ROCm 有 Triton；Cerebras graph matcher 无法覆盖某个关键 operator 时，通常需要 Cerebras engineer 介入。独立的 CSL language 能直接暴露 task、wavelet 和 fabric color 等底层机制，并在 HPC workload 上取得很强结果，但它与 PyTorch model workflow 是两个相对独立的世界。</p>
<p>这其中有一种奇怪的免疫力。 <a data-popup="flashattention-versions">FlashAttention</a>，GPU 的定义内核谱系时代，是一种通过内存层次结构平铺注意力的方案，而 WSE 没有可以平铺的层次结构：导致 AMD 多年移植延迟的优化类根本不适用。但免疫力和贫困是同一个事实。在 CUDA 上复合的 third-party kernel 生态系统没有可以附着的表面；该平台历史上的每一项内核改进都有一位作者。</p>
<p>这会将晶圆留在哪里？拥有真正的利基市场，诚实地赢得胜利：批量 decode 速度，独立验证，由将延迟定价高于成本的客户支付。在利基市场周围，有硬墙：每个 token 的定价为 3-5 倍，七年的培训上限为 70B，到 2025 年，收入仍约 86% 集中在两个与阿布扎比相关的客户（根据 2026 年 5 月 IPO 前后的 S-1 文件），以及最稀缺的资源，SRAM 密度，随着模型的不断增长而停止扩展。轩尼诗和帕特森承诺寒武纪大爆发； WSE 是其最极端的机身设计，它决定将内存墙作为一种封装选择，并花费 46,225 mm² 的硅片拒绝制造它。</p>
<hr/>
<h3 id="aws-trainium">AWS Trainium</h3>
<div class="philosophy">
<p><strong><a data-popup="annapurna-labs">Annapurna Labs</a></strong> 是 AWS Nitro 卡和 Graviton CPU 背后的芯片团队。它把 <strong>Trainium</strong> 定位为 TPU 路线的 fast follower：计算核心沿用 128×128 weight-stationary systolic array、software-managed scratchpad 和 whole-program compilation，并直接复用 <a href="https://openxla.org/xla">XLA</a>。Scale-out 则建立在 AWS 已有的 Nitro/EFA 网络之上。Trainium 真正独特的地方，是在这套计算核心旁加入专用 collective communication hardware，并利用 AWS 从芯片、服务器到云服务定价的垂直整合，只需要在 AWS 内部把总体成本做到优于 NVIDIA。</p>
</div>
<h4 id="genealogy">演进路线</h4>
<div class="genealogy">
<div class="gen-row"><div class="gen-year">2015</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://en.wikipedia.org/wiki/Annapurna_Labs">Annapurna Labs</a></em></strong></div><div class="gen-desc">亚马逊以约 3.5 亿美元收购了这家以色列芯片初创公司；它成为 AWS 的内部芯片团队。</div></div></div>
<div class="gen-row"><div class="gen-year">2018</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://en.wikipedia.org/wiki/AWS_Graviton">Graviton</a> + <a data-popup="nitro">Nitro</a></em></strong></div><div class="gen-desc">Arm 服务器 CPU 和 <a data-popup="dpu">DPU</a> 卸载结构。</div></div></div>
<div class="gen-row"><div class="gen-year">2019</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/inferentia.html">Inferentia</a></em></strong><span class="gen-chip"><a data-popup="neuroncore-v1">NeuronCore-v1</a></span></div><div class="gen-desc">首款 AWS ML 芯片，仅限推理：4 个 NeuronCore、8 GB DRAM、三个固定引擎。</div></div></div>
<div class="gen-row"><div class="gen-year">2022</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://aws.amazon.com/blogs/aws/amazon-ec2-trn1-instances-for-high-performance-model-training-are-now-available/">Trainium1</a></em></strong><span class="gen-chip"><a data-popup="trn1">Trn1</a>， <a data-popup="neuroncore-v2">v2</a></span></div><div class="gen-desc">第一个训练芯片：2 个 NeuronCore-v2、可编程 <a data-popup="gpsimd-engine">GPSIMD</a> 引擎、32 GB HBM、NeuronLink 2D 环面。</div></div></div>
<div class="gen-row"><div class="gen-year">2023</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://aws.amazon.com/blogs/machine-learning/aws-inferentia2-builds-on-aws-inferentia1-by-delivering-4x-higher-throughput-and-10x-lower-latency/">Inferentia2</a></em></strong><span class="gen-chip"><a data-popup="neuroncore-v2">v2</a></span></div><div class="gen-desc">与 NeuronCore-v2 共享 Trn1：推理和训练谱系汇聚在一个微架构上。</div></div></div>
<div class="gen-row"><div class="gen-year">2024</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://aws.amazon.com/blogs/aws/amazon-ec2-trn2-instances-and-trn2-ultraservers-for-aiml-training-and-inference-is-now-available/">Trainium2</a></em></strong><span class="gen-chip"><a data-popup="trn2">Trn2</a>， <a data-popup="neuroncore-v3">v3</a></span></div><div class="gen-desc">8 NeuronCore-v3，第一个真正的 <a data-popup="fp8">FP8</a> 加速，96 GB <a data-popup="hbm3">HBM3</a>； 64 芯片 <a data-popup="ultraserver">UltraServer</a>。助力 <a data-popup="project-rainier">Project Rainier</a>。</div></div></div>
<div class="gen-row"><div class="gen-year">2025</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-ec2-trn3-ultraservers/">Trainium3</a></em></strong><span class="gen-chip"><a data-popup="trn3">Trn3</a>、 <a data-popup="neuroncore-v4">v4</a></span></div><div class="gen-desc">首款 3 nm AWS 芯片 (TSMC N3P)； OCP <a data-popup="microscaling-formats">MXFP8/MXFP4</a>； <a data-popup="neuronswitch">NeuronSwitch</a> 全对全结构取代了环面。 144 芯片 UltraServer。</div></div></div>
</div>
<h4 id="architecture">架构</h4>
<p>Trainium 可以看作是 AWS 按自己的系统条件重新组装了一遍 TPU playbook：没有硬件 cache，由 compiler 提前安排 software-managed SRAM 和数据搬运，但各单元的组织方式不同。每颗 Trainium 只有少量 NeuronCore（Trn1 为 2 个，Trn2/Trn3 为 8 个）；每个 NeuronCore 也不是单一的 matmul engine，而是一组解耦的专用引擎：128×128 systolic array 组成的 <a data-popup="tensor-engine">Tensor Engine</a>、负责 reduction 的 <a data-popup="vector-engine">Vector Engine</a>、负责 point-wise ops 的 <a data-popup="scalar-engine">Scalar Engine</a>，以及 8 个可编程的 512-bit GPSIMD engine。外围还有 128 个 DMA engine、负责传输排序的 Sync Engine，以及 Trn2 开始加入的 CC-Core。系统没有 warp 或 wavefront；这些引擎按 compiler 生成的静态 dataflow schedule 并行工作。</p>
<p><img alt="AWS Trainium2 封装平面图 — 两个计算芯片并排放置在 CoWoS 中介层上，每个芯片 4 个 NeuronCore-v3（每个芯片 8 个）；每个芯片的两侧都有两个 HBM3 堆栈，通过外缘上的内存控制器实现。中央 NeuronLink 块承载封装内芯片到芯片链路和芯片到芯片环形端口；顶部的一个小型 PCIe / Nitro EFA 条是通往主机和横向扩展结构的路径。" loading="lazy" src="images/aws-trainium-chip.png"/></p>
<p><img alt="放大的一个 NeuronCore-v3 — 128×128 weight-stationaryTensor Engine 位于中心，从 SBUF 状态缓冲区（128 个分区）馈送 operand，并将部分和排入小型 PSUM accumulator。vector、scalar 和可编程 GPSIMD 引擎在同一个 SBUF 上并行运行；来自 HBM 的 128 个 DMA 引擎和一个同步引擎级块，以及一组 CC-Core 驱动 NeuronLink 端口，以便在计算的同时进行集合。" loading="lazy" src="images/aws-trainium-neuroncore.png"/></p>
<h5 id="compute">计算</h5>
<p><strong><a data-popup="tensor-engine">Tensor Engine</a></strong> 提供主要的 matmul throughput，其他 engine 负责 element-wise 与 control workload。它包含一个 128×128 weight-stationary systolic array：一块 operand 通过 <code class="notranslate" translate="no">LoadStationary</code> 固定在 array 中，另一块通过 <code class="notranslate" translate="no">MultiplyMoving</code> 流过阵列。partial sum 写入小型 accumulator SRAM——<a data-popup="psum">PSUM</a>——并通过 read-add-write 在 K dimension 上持续累加。底层仍然是 tile MMA；区别在于 NVIDIA 把它放进 warp hierarchy，Google 由 VLIW bundle issue，而 Trainium 将其暴露成访问命名 scratchpad 的显式 instruction。</p>
<p>所有三代的阵列物理尺寸均固定为 128×128；变化的是每个单元装载的产品数量。 <a data-popup="trn1">Trn1</a>的 NeuronCore-v2 运行 <a data-popup="bf16">BF16</a>/FP16 和 <a data-popup="fp32">FP32</a> 累计提供 <a data-popup="fp8">FP8</a> 仅在 BF16 速率（无加速）。 <a data-popup="trn2">Trn2</a>的 v3 双泵 FP8 呈现有效的 256×128 阵列，第一个具有真正 2× 的 Trainium 8 位。 <a data-popup="trn3">Trn3</a>的 v4 包 <a data-popup="microscaling-formats">microscaling</a> operand 以呈现有效的 512×128，4 倍 BF16 速率。物理乘加单元的计数永远不会移动；数据路径只是为它们提供更窄的数字。</p>
<p>其余 engine 负责让 Tensor Engine 持续有数据可算：Vector Engine 执行 LayerNorm、Softmax、pooling 等 reduction；Scalar Engine 处理 activation、GELU 等 point-wise op；8 个 GPSIMD engine 运行可编程 C code，承接其他 engine 无法覆盖的 long-tail operator。高质量 compile schedule 会让四类 engine overlap：Tensor Engine 计算当前 matmul，Vector Engine 处理上一块 tile，DMA engine 同时搬运下一块 tile。代价也很明确：无法映射到专用 engine 的新 operator 会落到较慢的 GPSIMD path，并可能成为瓶颈。</p>
<h5 id="memory">内存</h5>
<p>内存层次结构是应用于存储的计算原理： <em><strong>三层，全部由软件管理，任何地方都没有硬件缓存</strong></em>。 AWS 自己的文档进行了对比，指出与 CPU 或 GPU 不同，NeuronCore 没有缓存，并且“所有内存移动在程序本身中都是明确的”。片外为 <em><strong><a data-popup="hbm">HBM</a></strong></em> （Trn1 上为 32 GB，Trn2 上为 96 GB <a data-popup="hbm3">HBM3</a> ， Trn3 上 144 GB <a data-popup="hbm3e">HBM3e</a> ）。片上最靠近引擎的是 <em><strong><a data-popup="sbuf">状态缓冲区 (SBUF)</a></strong></em>：主要 scratchpad，大约 20 倍 HBM 带宽，组织为 128 个分区，每个 NeuronCore 大小为 24 MiB (v2)、28 MiB (v3)、32 MiB (v4)。在数组和 SBUF 之间有 <em><strong><a data-popup="psum">PSUM</a></strong></em>，这是一个专用于 matmul 输出的 2 MiB accumulator。数据移动 HBM → SBUF → Tensor Engine → PSUM → SBUF，每一跳均由编译器发出；硬件不会预取或逐出任何内容。</p>
<p>这与 Google VMEM 的思路一致：由 compiler 管理显式 scratchpad，不依赖 cache 掩盖 schedule 错误。schedule 正确时，data movement 与 compute 可以持续 overlap；schedule 错误时，则没有硬件 cache 或 prefetcher 兜底。Trainium 为相对适中的 peak FLOPs 配置了较充足的 HBM，因此每单位计算可用的 memory capacity 较高；但绝对容量仍落后于同期高端 GPU。AWS 真正利用的杠杆不是单芯片 memory leadership，而是自研芯片在 compute、HBM 和 cloud pricing 上的总体成本。</p>
<h5 id="numerics">数值格式</h5>
<p>Trainium 同样沿着 FP32 → BF16 → FP8 → FP4 演进，但有两个特点。第一是 <strong><a data-popup="cfp8">configurable FP8</a></strong>：Tensor Engine 不只固定支持 E4M3/E5M2，还允许调整 exponent bias，并支持 E3M4，让 compiler 能按 tensor 在 range 与 precision 之间取舍。第二是 Trn3 的 FP4 主要节省 memory capacity 和 bandwidth，并不提高 compute throughput：MXFP4 operand 在进入 array 前会转换为 MXFP8，因此仍以 FP8 rate 执行。Trainium 也使用 microscaling 和 hardware stochastic rounding 来恢复低精度 accuracy。需要注意的是，AWS 对 sparse peak 的宣传口径与 dense datapath 并不完全一致，比较数字时必须区分 sparse 与 dense。</p>
<h5 id="collectives-in-silicon">芯片内 collective communication</h5>
<p>Trainium 中最没有直接 GPU 对应物的是 <strong><a data-popup="cc-core">CC-Core</a></strong>。distributed training/inference 大量依赖 all-reduce、all-gather、reduce-scatter 和 all-to-all；GPU 通常通过 NCCL kernel 在 SM 上执行这些 collective，因此 communication 与 compute 会竞争同一批 execution resource。Trn2 每颗 chip 配置 20 个 CC-Core，并直接连接 NeuronLink port，使 collective 可以和 Tensor/Vector Engine 并行运行。思路与 TPU SparseCore 类似：把主 compute engine 不擅长的 workload 交给旁路专用 hardware。</p>
<h5 id="bets">设计取舍</h5>
<ul>
<li><em><strong>押注 1：云是</strong></em> Annapurna 设计芯片、服务器、机架、 <a data-popup="nitro">Nitro</a> 网络和云 API 作为一个堆栈，因此 Trainium 只需要在 AWS 内部的性价比上取胜，而不必在商业芯片规格表上取胜。</li>
<li><em><strong>借用计算论文，不要重新发明它。</strong></em> A 128×128 <a data-popup="weight-stationary">weight-stationary</a> 阵列，软件管理 <a data-popup="sbuf">SBUF</a>/<a data-popup="psum">PSUM</a> scratchpad 和整个程序编译是 TPU 的押注，可重复使用以共享 Google 的 <a data-popup="openxla">OpenXLA</a>。节省的精力将投入到网络和机架中。</li>
<li><em><strong>集体属于芯片。</strong></em> 专用 <a data-popup="cc-core">CC 内核</a> 重叠 <a data-popup="all-reduce">all-reduce</a> 和 <a data-popup="all-to-all">all-to-all</a> 在硬件中进行计算，而不是将它们作为从 matmul 单元窃取 FLOP 的内核运行。</li>
<li><em><strong>重用云自己的网络。</strong></em> 横向扩展是 <a data-popup="efa">EFA</a> 与 <a data-popup="srd">SRD</a> 传输：相同 <a data-popup="nitro">Nitro</a>-卸载、数据包喷射 <a data-popup="rdma">RDMA</a> 已运行 AWS 的其余部分。否 <a data-popup="infiniband">InfiniBand</a>。</li>
<li><em><strong>将拓扑移至工作负载。</strong></em> Trn1 和 Trn2 复制了 TPU 的 <a data-popup="torus">torus</a>； Trn3 的 <a data-popup="neuronswitch">NeuronSwitch</a> 将其替换为交换 <a data-popup="all-to-all">all-to-all</a> 结构 <a data-popup="moe">MoE</a> 流量超过了最近邻流量。老实说，这是遵循剧本的：首先是 Google，现在是 NVIDIA。</li>
</ul>
<h4 id="scaling">扩展</h4>
<p>Trainium 把互连分成两层：紧耦合的 <strong><a data-popup="neuronlink">NeuronLink</a></strong> domain 连接需要协同执行的芯片，domain 之外则使用 AWS 通用的 <strong><a data-popup="efa">EFA</a></strong> fabric。NeuronLink domain 不是 NVLink 那样的 coherent shared memory；虽然 AWS 把 UltraServer 描述为 pooled multi-TB memory system，底层语义仍是 point-to-point message passing，因此更接近 TPU ICI，而不是 NVSwitch crossbar。</p>
<div class="definitions">
<div class="definition"><div class="definition-term">扩展</div><div class="definition-body"><a data-popup="neuronlink">NeuronLink</a> 将芯片绑定到一个 <a data-popup="ultraserver">UltraServer</a>。通过 Trn2，拓扑是一个 <a data-popup="torus">环面</a> （4×4 2D 环面中每个实例 16 个芯片，4×4×4 3D 环面中每个 UltraServer 64 个芯片）； Trn3 将其替换为 <a data-popup="neuronswitch">NeuronSwitch</a> 全方位面料。消息传递，非连贯加载/存储。</div></div>
<div class="definition"><div class="definition-term">通过以太网横向扩展</div><div class="definition-body"><a data-popup="efa">弹性结构适配器</a> ，卸载到 <a data-popup="nitro">Nitro</a>。 <a data-popup="srd">SRD</a> 传输将每个流喷射到多个路径上，并可靠但无序地传送； <a data-popup="ultracluster">UltraClusters</a> 覆盖范围 <a data-popup="10p10u">10p10u</a> 结构上有数十万个芯片。</div></div>
</div>
<h5 id="scale-up">纵向扩展</h5>
<p>NeuronLink 是 Trainium 的在芯片到芯片结构中， <a data-popup="nvlink">NVLink</a> 为 NVIDIA 扮演角色， <a data-popup="ici">ICI</a> 为 TPU 扮演角色。通过 Trn2，它将芯片连接到 <em><strong><a data-popup="torus">torus</a></strong></em>，这正是 TPU 的选择：单个 <em><strong><a data-popup="trn2">trn2</a></strong></em> 实例是 4×4 2D 环面中有 16 个芯片，每个芯片约为 1.28 TB/s， <em><strong><a data-popup="ultraserver">Trn2 UltraServer</a></strong></em> 将四个实例连接到 4×4×4 3D 环面上的 64 个芯片中，呈现 83 个密集 <a data-popup="fp8">FP8</a> PetaFLOPS 和约 6 TB 的 <a data-popup="hbm">HBM</a> 作为一个扩展域。第三个 torus 轴故意设计得很薄（实例间环的运行速度为每个芯片约 256 GB/s，而实例内部的运行速度为 1.28 TB/s），这是 torus 的特点：廉价的布线和巨大的最近邻带宽，但代价是直径上的许多跳。 AWS 将 64 芯片 UltraServer 定位为 NVIDIA 的 72-GPU <a data-popup="gb200">NVL72</a>；聚合计算处于同一水平，但环面不是 <a data-popup="crossbar">crossbar</a>，并且两者在非最近邻流量上的表现非常不同。</p>
<p>这种交易是 Trn3 放弃 <em><strong><a data-popup="neuronswitch">NeuronSwitch-v1</a></strong></em> 是一种交换 <em><strong><a data-popup="all-to-all">全对</a></strong></em> 结构，可将芯片间带宽大致增加一倍，更重要的是，使直径变平，以便任何芯片都能在一次切换的跳跃中到达任何其他芯片。 Trn3 UltraServer 可扩展到 144 个芯片，实现 362 个密集 FP8 PetaFLOPS 和 20.7 TB 的 <a data-popup="hbm3e">HBM3e</a>。这一动机也推动 Google 走向高基数拓扑，以实现 <a data-popup="moe">MoE</a> 推理： <a data-popup="moe-expert-routing">专家路由</a> 是 all-to-all，这是环面的最坏情况，而交换机将最长的跳跃对变成单个交叉。 Trainium 的互连路线图是行业路线图的压缩版：在工作负载为最近邻时采用环面，在不是最近邻时切换到 crossbar。</p>
<p><img alt="Trn3 UltraServer 纵向扩展 — Trn3 放弃了 Trn2 环面，采用 NeuronSwitch-v1，这是 NeuronLink-v4 上的一种交换式全对全结构（每芯片约 2 TB/s）。在服务器内，芯片通过第一级 (L1) NeuronSwitch 连接，因此任何芯片都可以通过一跳到达任何其他芯片；在服务器之间，两个二级 (L2) NeuronSwitch 将 144 芯片 UltraServer 连接到一个全对所有域（20.7 TB HBM3e、362 个密集 FP8 PetaFLOPS）。 MoE 和 all-to-all 集体的扁平直径，其中环面支付跳数。" loading="lazy" src="images/aws-trainium-scale-up.png"/></p>
<h5 id="scale-out">横向扩展</h5>
<p>Scale-out 直接复用 AWS 的数据中心网络。每个 Trainium instance 通过 Elastic Fabric Adapter（EFA）接入网络，SRD transport 则由 Nitro card offload。与 RoCE 或 InfiniBand 的单条有序 flow 不同，SRD 会把 packet 分散到最多 64 条并行 path 上，可靠但乱序地送达，再由 collective library 重组，从而避免某一条拥塞 path 引发 head-of-line blocking。这套 transport 原本服务 AWS cloud network，后来直接复用于 accelerator fabric。</p>
<p><img alt="AWS Trainium 横向扩展 — UltraServer 通过卸载到 Nitro 卡的 Elastic Fabric Adapter NIC 通过标准以太网而不是 InfiniBand 进行连接。 SRD 传输将每个流喷洒在多达 64 条路径上，并可靠但无序地传送，避开队头阻塞。 10p10u UltraCluster 结构（不到 10 微秒，约 10 petabits/s）将数十万个芯片连接在一起； Rainier 项目是 Anthropic 跨多个美国数据中心的约 500,000 个 Trainium2 芯片。" loading="lazy" src="images/aws-trainium-scale-out.png"/></p>
<p>最上层是 <strong><a data-popup="ultracluster">UltraCluster</a></strong>，通过 10p10u network 把大量 UltraServer 连接起来；10p10u 表示整个数据中心约 10 petabits/s 的带宽和低于 10 μs 的延迟。代表性部署是 Project Rainier：约 50 万颗 Trainium2 分布在多个美国数据中心，并在 2025 年底为 Anthropic 上线；到 2026 年初，Claude 使用的 Trainium chip 已超过 100 万颗。AWS 声称 Trainium2 相比同代 Hopper instance 可提供更高 price-performance，但这个优势来自 end-to-end economics：Amazon 同时拥有 chip、Nitro network、server 和 cloud pricing。</p>
<h4 id="software">软件栈</h4>
<p>Trainium 的软件栈同样是 compiler-first。<strong><a href="https://awsdocs-neuron.readthedocs-hosted.com/">Neuron SDK</a></strong> 建立在 OpenXLA 之上：<code class="notranslate" translate="no">neuronx-cc</code> 接收 XLA HLO，并 lower 成由 Neuron Runtime 加载到 NeuronCore 的 <a data-popup="neff">NEFF</a> binary。<a data-popup="torch-neuronx">torch-neuronx</a> 通过 PyTorch/XLA 的 LazyTensor 机制记录计算图，在 step boundary 触发编译；<strong>jax-neuronx</strong> 则经 StableHLO 接入。若把 CUDA 看作 kernel-driven 的一端，把 TPU/XLA 看作 whole-program compilation 的另一端，Trainium 明显更接近后者：compiler 基本就是整个系统。</p>
<p>与 TPU 不同的是，Trainium 还提供更直接的 kernel escape hatch。XLA 不一定能为新的 attention variant 或 fused MoE schedule 自动生成最佳实现，因此 Neuron 提供 <strong><a data-popup="nki">NKI（Neuron Kernel Interface）</a></strong>：一种 Python tile-level kernel language，直接暴露各执行引擎以及 SBUF/PSUM scratchpad。它在 Trainium 上扮演的角色，接近 TPU 的 Pallas 或 GPU 的 Triton——当性能关键在 schedule、tiling 和 DMA，而不是代数化简时，开发者可以手动控制底层。其下的 collective communication library 把 all-reduce 和 all-to-all 映射到 CC-Core 与 NeuronLink，<a data-popup="nxd">NeuronX Distributed</a> 则提供上层的分布式训练与 sharding。</p>
<p>Trainium 与 CUDA、甚至 TPU software stack 的主要差距仍是 maturity。NKI、JAX path 和 distributed library 在较长时间内处于快速演进状态；model port 只能运行在 AWS 上，也缺少成熟的跨厂商 fallback。Anthropic 的实践最能说明这一点：它不是简单地把现有 PyTorch model 指向 Trainium，而是与 Annapurna 深度协作，自行开发低层 NKI kernel 并向 Neuron stack upstream fix。Trainium 已能支撑 frontier workload，但往往需要 hardware/software co-design，而不是开箱即用。</p>
<hr/>
<h3 id="groq-lpu">Groq LPU</h3>
<div class="philosophy">
<p>Groq LPU 的核心是 <strong>deterministic execution</strong>。GPU/TPU 通常保留 cache、scheduler、arbitration 或 dynamic network 来容忍 latency variation；Groq 则尽量删除这些 reactive component，把 instruction、data movement 和 chip-to-chip transfer 全部交给 compiler 做 cycle-accurate scheduling。片上 memory 全部是 SRAM，network 也按静态 schedule 运行，因此一组 LPU 可以像一台大机器一样，以可预测的 latency 执行同一个 program。代价是 flexibility 较低，但优势是 batch-one autoregressive decode 的 latency 极小且稳定。</p>
</div>
<h4 id="genealogy">演进路线</h4>
<div class="genealogy">
<div class="gen-row"><div class="gen-year">2016</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://en.wikipedia.org/wiki/Groq">创立</a></em></strong></div><div class="gen-desc"><a data-popup="jonathan-ross">乔纳森·罗斯</a>，他创立了 Google <a data-popup="tpu-v1">TPU</a> 作为 20% 的项目，剩下来构建确定性推理芯片。</div></div></div>
<div class="gen-row"><div class="gen-year">2020</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://groq.humain.ai/wp-content/uploads/2024/02/2020-Isca.pdf">TSP</a></em></strong><span class="gen-chip"><a data-popup="groqchip">GroqChip 1</a></span></div><div class="gen-desc">第一颗芯片（<a data-popup="isca">ISCA</a> 2020， <em>快速思考</em>)：单个 <a data-popup="functional-slice">functional slice</a> 核心，14 nm，无 <a data-popup="hbm">HBM</a>，无缓存。</div></div></div>
<div class="gen-row"><div class="gen-year">2022</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://dl.acm.org/doi/10.1145/3470496.3527405">Multi-chip</a></em></strong></div><div class="gen-desc"><a data-popup="isca">ISCA</a> 2022： <a data-popup="software-scheduled-networking">Software-Scheduled Network</a> 通过编译的软件将确定性调度扩展到数千个芯片 <a data-popup="dragonfly">Dragonfly</a>。</div></div></div>
<div class="gen-row"><div class="gen-year">2023</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://www.prnewswire.com/news-releases/groq-selects-samsung-foundry-to-bring-next-gen-lpu-to-the-ai-acceleration-market-301900464.html">三星 4 纳米</a></em></strong></div><div class="gen-desc">第二代 LPU 在三星 <a data-popup="sf4x">SF4X</a>上发布；它从未发货（报告的流片失败）。</div></div></div>
<div class="gen-row"><div class="gen-year">2024</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://techcrunch.com/2024/03/01/ai-chip-startup-groq-forms-new-business-unit-acquires-definitive-intelligence/">LPU / GroqCloud</a></em></strong></div><div class="gen-desc">TSP 已更名为 <a data-popup="lpu">Language Processing Unit</a>;该公司以创纪录的 decode 速度从销售卡转向销售 token。</div></div></div>
<div class="gen-row"><div class="gen-year">2025 年</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://groq.com/newsroom/groq-and-nvidia-enter-non-exclusive-inference-technology-licensing-agreement-to-accelerate-ai-inference-at-global-scale">NVIDIA 许可</a></em></strong></div><div class="gen-desc">NVIDIA 获得 LPU 技术 <a data-popup="nvidia-groq-deal">非独占许可</a> ，并聘请 Ross 和大部分技术人员</div></div></div>
<div class="gen-row"><div class="gen-year">2026</div><div class="gen-body"><div class="gen-head"><strong><em><a href="https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/">NVIDIA Groq 3 LPU</a></em></strong><span class="gen-chip"><a data-popup="lp30">LP30 / LPX</a></span></div><div class="gen-desc">该技术再次出现在 <a data-popup="gtc">GTC</a> 2026 作为 <a data-popup="rubin">Rubin</a> NVL72 旁边的延迟协处理器，通过 <a data-popup="attention-ffn-disaggregation">注意力 FFN 分解</a>。</div></div></div>
</div>
<h4 id="architecture">架构</h4>
<p>其他加速器通常先设计一个 core，再把 SM、TensorCore、CU 或 dataflow core 复制到整个 die 和整个集群。Groq LPU 采用了相反方向：它把一个传统 core 拆成贯穿芯片高度的 <strong><a data-popup="functional-slice">functional slice</a></strong>，分别承担 instruction control、vector ALU、matrix unit、SRAM 和 network。每个 slice 内部高度同质，但整个芯片由不同功能的 slice 横向拼接而成。数据不会停在 register file 中等待某个执行单元，而是像流水线上的部件一样，在 slice 之间向东或向西流动，每周期前进一个 register hop；VLIW instruction 则从南侧的 control plane 向北发射。compiler 在编译期知道每个 cycle、每个 operand 的确切位置，hardware 只需按确定的时序执行。这也是它最初被称为 <strong>Tensor Streaming Processor（TSP）</strong>、后来改名 Language Processing Unit 的原因。</p>
<p><img alt="Groq LPU 布局图 — 芯片围绕中央 VXM vector 切片分为镜像的东半球和西半球。向外读取：边缘的 MXM 矩阵平面，然后是 SXM 开关切片，然后是 VXM 侧面的 MEM SRAM 切片组。指令控制 (ICU) 沿着南边运行，并向北向每个片发出 VLIW 包；operand 流在切片之间向东和向西流动，每个周期一个寄存器跳。 320  个 lane 垂直堆叠为 20  个 superlane。" loading="lazy" src="images/groq-chip.png"/></p>
<p>纵轴表示 SIMD width：芯片共有 320 个 lane，组织成 20 个 <strong><a data-popup="superlane">superlane</a></strong>，每个 superlane 包含 16 个 lane；额外的第 21 个 superlane 用作 yield spare，对 software 不可见。每个 functional slice 同时作用于全部 320 个 lane。横轴表示时间：每个 lane 有 64 个逻辑 <a data-popup="stream-register">stream register</a>，其中 32 个向东、32 个向西；每个 cycle，stream register 都会沿对应方向前进一个 slice，直到被消费或离开 die。slice 从经过的数据流读取 operand，执行计算，再把结果写入继续流向下一 slice 的 stream register。芯片以中央 VXM 为轴左右镜像，因此同一个结果可以被两侧的 slice 使用。</p>
<h5 id="compute">计算</h5>
<p>LPU 仍然把 matrix workload 放到专用单元，把其他计算放到 Vector Engine，只是两者都以 stream slice 的形式排列。matrix path 是 <strong><a data-popup="mxm">MXM</a></strong>：四个独立的 320×320 multiply-accumulate plane，每个半区两个，共包含 409,600 个 multiplier。weight 可以在 40 cycle 内装入整个 plane，随后 activation 流过并完成 accumulation。在 900 MHz 下，peak throughput 约为 750 INT8 TOPS 或 188 FP16 TFLOPS。这里没有依赖 structured sparsity 的倍率：TSP 刻意不做 data-dependent zero skipping，因为执行时间必须保持 deterministic。</p>
<p>芯片中央的 <strong><a data-popup="vxm">VXM</a></strong> 构成 vector path：每个 lane 有 16 个 ALU，排列成 4×4 mesh，总计 5,120 个 32-bit ALU，用于 activation、normalization、quantization 和 residual add。由于 compute unit 在空间上串联，operand 可以连续穿过多级 VXM ALU，再直接进入 MXM，而不必回写 memory；GPU kernel 需要手工实现的 operator fusion，在这里就是 slice 的物理顺序。第三类 SXM slice 负责 lane shift、320-lane permute、transpose 和 chip-to-chip link，因此 cross-lane data rearrangement 是原生 operation，不需要绕行 SRAM。</p>
<h5 id="memory">内存</h5>
<p>没有 HBM，没有 DRAM，也没有缓存。片上是 <em><strong><a data-popup="mem-slice">MEM</a></strong></em> 片：88 个片（每个半球 44 个）中包含 230 MB SRAM，每个字节是计算片的单个周期，合计约为 80 TB/s。这就是整个层次结构：一层、扁平、软件寻址，没有任何会引入可变延迟访问的逐出、预取或一致性机制。</p>
<p>这套设计最直接的约束是容量：230 MB on-chip SRAM 根本放不下大模型。Llama-2 70B 的 FP16 权重约 140 GB，因此一个 model replica 必须切到数百颗 LPU 上，权重分布在整座机架的 SRAM 中，典型配置约需要 576 颗芯片。GPU 通常把模型放在少量 GPU 的 HBM 中，让 token 流经这些 GPU；LPU 则把模型铺在整个 SRAM cluster 上，让 token 流经整个 cluster。因此芯片数量首先由 weight capacity 决定，而不是由 compute throughput 决定。它与 Cerebras 都押注 SRAM，但方向相反：Cerebras 做一颗巨大的 wafer-scale chip，Groq 则用大量普通尺寸的 die 拼出足够大的 SRAM pool。</p>
<h5 id="numerics">数值格式</h5>
<p>数字是未走的路。这里的其他供应商每一代都将精度减半， <a data-popup="fp16">FP16</a> 到 <a data-popup="fp8">FP8</a> 到 <a data-popup="fp4">FP4</a> 具有块缩放功能以重新获得准确性。 TSP 停留在 <em><strong>FP16 和 INT8</strong></em> FP32 累积，从未在芯片中交付 FP8 或 FP4。它的一个数字思想是 <em><strong><a data-popup="truepoint">TruePoint</a></strong></em>：将 320 个元素的点积融合到具有 FP32 累加的单个舍入步骤中，因此 FP16 乘法器阵列在缩减方面接近 FP32 精度（Groq 报告相对于 FP32 基线）。</p>
<p>无论 16 位是信念还是从未获得低精度刷新的数据路径，都很难与第二代芯片从未发货的事实分开。 SRAM 容量是该架构最稀缺的资源，8 位权重将使模型所需的芯片数量减半；这种容量限制的机器有充分的理由想要 FP8，但没有将其移植到芯片上。同样的悬而未决的问题也笼罩在 Cerebras 的 16 位数据路径上，也面临同样的压力：供应商最渴望以最宽的精度进行容量计算。</p>
<h5 id="determinism">确定性</h5>
<p>所有其他加速器都隐藏延迟； LPU <em><strong>公开</strong></em> 它。 ISA 承载每条指令的执行延迟，数据路径在构造上是固定延迟的，因此编译器会提前计算每个结果出现的确切周期。硬件中的任何东西都不会干扰该调度：没有缓存会丢失，没有仲裁器会停止，没有分支会错误预测，没有推测会展开。 Groq 自己的测量就是证明：BERT-Large 的运行在约 75 µs 的范围内返回了 24,240 次，编译器的预测延迟在测量值的 2% 以内。</p>
<p>这是 TPU 的本能（将调度移至编译器中，删除事后猜测的硬件）更进一步。 TPU 编译器调度芯片； LPU 编译器调度 <em><strong>系统</strong></em>，因为确定性也适用于整个网络。它与 Cerebras 完全相反，其核心是 <em><strong><a data-popup="dataflow-processor">数据流</a></strong></em>，每当 operand 恰好到达时就会触发：WSE 对数据做出反应，LPU 对其进行定时。两台机器都删除调度程序；一个将其替换为到达，另一个将其替换为时钟。</p>
<h5 id="bets">设计取舍</h5>
<ul>
<li><em><strong>确定性优于容忍。</strong></em> 删除每个反应性组件（缓存、仲裁器、预测器、重新排序缓冲区）并让编译器拥有每个周期。</li>
<li><em><strong>空间功能切片。</strong></em> 将核心分解为切片并通过它们流式传输 operand，因此融合是布局规划和数据重用，而不是 register file 舞蹈。</li>
<li><em><strong>SRAM 是唯一的内存。</strong></em> 不惜任何容量成本，无需 HBM。将保留片上模型的能力换成单周期、固定延迟访问，接受模型必须跨越数百个芯片。</li>
<li><em><strong>也安排网络。</strong></em> 让芯片成为自己的路由器并逐周期编译通信，因此一千个芯片集群是一个确定性程序，没有交换机，也没有拥塞。</li>
<li><em><strong>销售延迟，而不是吞吐量。</strong></em> 针对第 1 批中每个用户每秒的 token 数进行优化，该机制 GPU 最差，并且定价与产品速度相同，而不是在每个 token 的成本上进行竞争。</li>
</ul>
<h4 id="scaling">扩展</h4>
<p>扩展 LPU 与这里的其他任何东西不同，因为没有单独的扩展结构需要构建：芯片已经是一个交换机。每个 LPU 最多承载 16 个芯片到芯片 <em><strong><a data-popup="realscale">RealScale</a></strong></em> 链路（11 个暴露在卡上），并同时充当计算端点和路由器。将芯片直接相互连接，集群就是一个 <em><strong><a data-popup="glueless-multiprocessor">无缝 Multi-chip</a></strong></em>：无 <a data-popup="nic">NIC</a>，无 <a data-popup="nvswitch">交换机 ASIC</a>，无架顶交换机。由于确定性贯穿这些链接，因此整个集群按照一个编译时计划运行。</p>
<div class="definitions">
<div class="definition"><div class="definition-term">纵向扩展</div><div class="definition-body">节点：8 块接口板通过 <a data-popup="realscale">RealScale</a> C2C 全连接，形成一个 <a data-popup="dragonfly">Dragonfly</a> 呈现为单个高基虚拟路由器的组。软件调度、无开关、无一致性。</div></div>
<div class="definition"><div class="definition-term">横向扩展</div><div class="definition-body">相同的结构，经过扩展。 <a data-popup="dragonfly">Dragonfly</a> 节点：每机架 9 个（72 个芯片，一个节点为热备用），可扩展至规格 10,440 个芯片，每一跳仍遵循已编译的确定性计划。</div></div>
</div>
<h5 id="scale-up">纵向扩展</h5>
<p>该节点有 8 个 LPU，完全连接：每个芯片的 7 个链路将其连接到其他七个芯片，因此节点中的每个芯片彼此之间都是一跳。每个芯片上剩余的 4 个链路（跨节点的 32 个链路）捆绑到 ISCA 论文中所谓的 32 端口虚拟路由器中，将节点的上行链路连接到更大的结构中。没有底板交换机，也没有一致的地址空间；远程 operand 未加载，它按 <em>计划</em> 到达，由源芯片在编译器选择的周期上注入，并由目标在其到达的周期上消耗。</p>
<p><img alt="Groq 横向扩展 — 8 个 LPU 完全连接形成一个节点（一个 Dragonfly 组呈现为一个高基数虚拟路由器）； 9 个节点组成一个 72 芯片机架，一个节点一个热备。这些芯片就是路由器：没有网卡，没有交换机。编译器逐周期调度每个芯片到芯片的传输（调度，非路由），准同步链路通过每 256 个周期交换的硬件对齐计数器保持同步，并用 FEC 代替重传，因此重试永远不会扰乱调度。 70B 型号跨越整个 SRAM 机架。" loading="lazy" src="images/groq-scale.png"/></p>
<h5 id="scale-out">横向扩展</h5>
<p>在节点之外，节点连接到 <em><strong><a data-popup="dragonfly">Dragonfly</a></strong></em>：9 个节点构成一个 72 芯片机架（第 9 个是热备用，因此 64 个活动），并且拓扑可扩展到指定的 10,440 个芯片，其中任何两个节点相距六跳以下。结构是 <em><strong><a data-popup="software-scheduled-networking">软件调度</a></strong></em>：路由和流量控制移至编译时，论文的框架很生硬， <em>已安排，未路由</em>。没有背压，也没有动态仲裁，因为编译器已经证明接收器已经准备好；链接携带 <a data-popup="fec">前向纠错</a> 而不是重传，因为重试会扰乱时间表。保持独立时钟芯片机架同步本身就有问题：链接是 <em><strong><a data-popup="plesiochronous">准同步的</a></strong></em>，并且该结构通过 <em><strong><a data-popup="hardware-aligned-counter">硬件对齐计数器</a></strong></em> 在生成树上每 256 个周期进行交换来保持全局共识时间，并通过定期去歪斜指令使每个芯片恢复对齐。 Groq 报告的回报是 8 路 <a data-popup="all-reduce">all-reduce</a> 匹配 <a data-popup="a100">A100</a>/<a data-popup="nvswitch">NVSwitch</a> 节点在大张量上，并在小张量上击败它，其中预定的结构不会支付动态张量所需要的握手延迟。</p>
<p>这种 SRAM-first 设计的成本最终体现在系统规模上。根据 SemiAnalysis 的估算，一个 Llama-2 70B replica 需要约 576 颗 LPU，并配套大量 host CPU 和 host memory；而 GPU 方案可能只需要一个 8-GPU server。单颗 14 nm LPU die 的成本不高，但完整系统需要数百颗芯片，而且 decode 时大部分 matrix compute 会空闲，真正繁忙的是 SRAM bandwidth。结论很直接：当目标是最低 latency 时，LPU 很有优势；一旦通过 batching 追求每美元 throughput，GPU 通常更划算。Groq 竞争的核心不是最低 token cost，而是响应速度。</p>
<h4 id="software">软件栈</h4>
<p>Groq 的 programming model 最能体现“compiler is the machine”。平台没有用户手写 kernel API：开发者输入 PyTorch、TensorFlow 或 ONNX model，compiler 将其 lower 到较小的 tensor op set，再静态安排每条 instruction、每个 stream 以及每次 chip-to-chip transfer。由于 hardware 中不存在需要动态调度的部分，开发者也不需要直接编写 <code class="notranslate" translate="no">wgmma</code> 或手调 tile。优势是 compiler 可以一次性看到全局 schedule；代价是 compiler、profiler 和 runtime 都由 Groq 控制，生态规模远小于 CUDA。</p>
<p>该枢纽是讲述架构的用途。 LPU 的构造是 <em><strong>仅推理</strong></em> （Ross 的框架是训练是本地游戏，推理是全局游戏），而且它在单用户 decode 延迟这一点上是不败的。独立测量支持了这一说法， <em><strong><a href="https://artificialanalysis.ai/providers/groq">人工分析</a></strong></em> 将 Groq 列为开放模型上每秒 token 速度最快的提供商之一。它与其他部分的匹配度很差：不适合 SRAM 机架的模型、需要大批量实现每美元吞吐量的工作负载，或者静态调度无法表达的动态控制流。 <a data-popup="moe">MoE</a> 提供了服务，但其依赖于数据的专家路由对于想要了解所有内容的编译器来说很尴尬。进展，Groq 几乎没有发表关于如何协调两者的文章。</p>
<p>故事的结尾是 NVIDIA 接手了这套技术。2025 年 12 月，NVIDIA 获得 LPU 技术的 <strong><a data-popup="nvidia-groq-deal">非独占许可</a></strong>，并聘用了 Jonathan Ross 和团队的大部分成员。按 NVIDIA 10-K 的表述，这不是公司收购：产品、客户合同和股权都没有转移，只是交易金额让很多媒体把它简称为 acquisition。到 GTC 2026，这项技术以 <strong>NVIDIA Groq 3 LPU</strong> 的形式重新出现：一组纯 SRAM inference chip 与 Rubin NVL72 配合，GPU 执行 attention，LPU 执行 FFN 和 MoE，由 Dynamo 协调两侧切换。高度 deterministic 的 LPU 最终成为通用 GPU 系统中的 latency coprocessor；GroqCloud 则继续在原来的 14 nm 芯片上提供 token service。</p>
<hr/>
<h3 id="comparison">对比</h3>
<p>表中的算力均为相应精度下的 peak dense throughput；若厂商没有公开底层数据，则单独标记。Memory bandwidth 取各架构最主要的存储层：GPU、TPU、Trainium 使用 HBM，Cerebras 和 Groq 使用聚合 on-chip SRAM bandwidth，因此这些数字不能直接横向比较。Scale-up bandwidth 也沿用各厂商自己的口径，可能表示 per-chip aggregate、rack aggregate，或真正的 bisection bandwidth。</p>
<h5 id="per-chip">单芯片</h5>
<div class="comparison-wrap"><table class="comparison-table">
<thead>
<tr><th>公司</th><th>年份</th><th>芯片</th><th>加速器内存</th><th>内存带宽</th><th>旗舰密集 FLOPs</th><th>TDP</th><th>纵向扩展带宽</th></tr>
</thead>
<tbody>
<tr><td class="company" rowspan="6"><img alt="NVIDIA" class="company-logo" loading="lazy" src="images/NVIDIA.png"/></td><td>2023</td><td>H100 SXM5</td><td>80 GB HBM3</td><td>3.4 TB/s</td><td>1.98 PetaFLOPS FP8</td><td>700 W</td><td>900 GB/s</td></tr>
<tr><td>2024</td><td>H200 SXM</td><td>141 GB HBM3e</td><td>4.8 TB/s</td><td>1.98 PetaFLOPS FP8</td><td>700 W</td><td>900 GB/s</td></tr>
<tr><td>2024</td><td>B200</td><td>192 GB HBM3e</td><td>8 TB/s</td><td>4.5 PetaFLOPS FP8 / 9 PetaFLOPS FP4</td><td>1,000 W</td><td>1.8 TB/s</td></tr>
<tr><td>2025</td><td>B300</td><td>288 GB HBM3e</td><td>8 TB/s</td><td>7.5 PetaFLOPS FP8 / 15 PetaFLOPS FP4</td><td>1,400 W</td><td>1.8 TB/s</td></tr>
<tr><td>2026</td><td>Rubin</td><td>288 GB HBM4*</td><td>~13 TB/s*</td><td>~17 PetaFLOPS FP8* / ~50 PetaFLOPS FP4*</td><td>~1,500 W*</td><td>3.6 TB/s</td></tr>
<tr><td>2027</td><td>Rubin Ultra</td><td>1 TB HBM4e*</td><td>~32 TB/s*</td><td>~33 PetaFLOPS FP8* / ~100 PetaFLOPS FP4*</td><td>~1,800 W*</td><td>3.6 TB/s</td></tr>
</tbody>
<tbody>
<tr><td class="company" rowspan="3"><img alt="Google" class="company-logo" loading="lazy" src="images/google.png"/></td><td>2023</td><td>TPU v5p</td><td>95 GB HBM2e</td><td>2.8 TB/s</td><td>0.46 PetaFLOPS BF16</td><td>n/d</td><td>1.2 TB/s</td></tr>
<tr><td>2025</td><td>TPU Ironwood (v7)</td><td>192 GB HBM3e</td><td>7.4 TB/s</td><td>4.6 PetaFLOPS FP8</td><td>n/d</td><td>1.2 TB/s</td></tr>
<tr><td>2026</td><td>TPU v8t Sunfish</td><td>216 GB HBM3e</td><td>6.5 TB/s</td><td>12.6 PetaFLOPS FP4</td><td>n/d</td><td>n/d</td></tr>
</tbody>
<tbody>
<tr><td class="company" rowspan="4"><img alt="AMD" class="company-logo is-tight" loading="lazy" src="images/favicon.ico"/></td><td>2023</td><td>MI300X</td><td>192 GB HBM3</td><td>5.3 TB/s</td><td>2.6 PetaFLOPS FP8</td><td>750 W</td><td>896 GB/s</td></tr>
<tr><td>2024</td><td>MI325X</td><td>256 GB HBM3e</td><td>6.0 TB/s</td><td>2.6 PetaFLOPS FP8</td><td>1,000 W</td><td>896 GB/s</td></tr>
<tr><td>2025</td><td>MI355X</td><td>288 GB HBM3e</td><td>8 TB/s</td><td>10 PetaFLOPS FP8 / 20 PetaFLOPS FP4</td><td>1,400 W</td><td>1,075 GB/s</td></tr>
<tr><td>2026</td><td>MI455X</td><td>待定</td><td>待定</td><td>~40 PetaFLOPS FP4*</td><td>待定</td><td>n/d</td></tr>
</tbody>
<tbody>
<tr><td class="company" rowspan="2"><img alt="Cerebras" class="company-logo" loading="lazy" src="images/cerebras-color.svg"/></td><td>2021</td><td>WSE-2</td><td>40 GB SRAM（晶圆上）</td><td>20 PB/s（聚合）</td><td>7.5 PetaFLOPS FP16</td><td>23 kW（系统）</td><td>（域 = 晶圆）</td></tr>
<tr><td>2024</td><td>WSE-3</td><td>44 GB SRAM（晶圆上）</td><td>21 PB/s（聚合）</td><td>~15.8 PetaFLOPS FP16*</td><td>23 kW（系统）</td><td>（域 = 晶圆）</td></tr>
</tbody>
<tbody>
<tr><td class="company" rowspan="3"><img alt="AWS" class="company-logo" loading="lazy" src="images/aws.png"/></td><td>2022</td><td>Trainium1</td><td>32 GB HBM2e*</td><td>820 GB/s</td><td>0.19 PetaFLOPS BF16/FP8</td><td>不适用</td><td>不适用</td></tr>
<tr><td>2024</td><td>Trainium2</td><td>96 GB HBM3</td><td>2.9 TB/s</td><td>1.3 PetaFLOPS FP8</td><td>~500 W*</td><td>1.28 TB/s</td></tr>
<tr><td>2025</td><td>Trainium3</td><td>144 GB HBM3e</td><td>4.9 TB/s</td><td>2.5 PetaFLOPS FP8</td><td>n/d</td><td>n/d</td></tr>
</tbody>
<tbody>
<tr><td class="company" rowspan="2"><img alt="Groq" class="company-logo" loading="lazy" src="images/groq.png"/></td><td>2020</td><td>GroqChip （第一代 TSP/LPU）</td><td>230 MB SRAM</td><td>80 TB/s（片上聚合）</td><td>0.188 PetaFLOPS FP16</td><td>215 W</td><td>330 GB/s（11 链路卡）</td></tr>
<tr><td>2026</td><td>NVIDIA Groq 3 LP30</td><td>500 MB SRAM</td><td>150 TB/s（片上聚合）</td><td>~1.2 PetaFLOPS FP8*</td><td>n/d</td><td>2.5 TB/s</td></tr>
</tbody>
</table></div>
<h5 id="per-rack-pod">单机架 / Pod</h5>
<div class="comparison-wrap"><table class="comparison-table">
<thead>
<tr><th>公司</th><th>年</th><th>系统</th><th>芯片</th><th>聚合密集 FLOP</th><th>加速器内存总计</th><th>扩展结构 BW</th><th><a data-popup="per-chip-nic">每芯片 NIC</a></th><th>电源</th><th>冷却</th></tr>
</thead>
<tbody>
<tr><td class="company" rowspan="6"><img alt="NVIDIA" class="company-logo" loading="lazy" src="images/NVIDIA.png"/></td><td>2023</td><td>HGX H100</td><td>8</td><td>16 PetaFLOPS FP8</td><td>640 GB</td><td>7.2 TB/s</td><td>400 Gbps (CX-7)</td><td>~10 kW</td><td>风冷</td></tr>
<tr><td>2024</td><td>HGX H200</td><td>8</td><td>16 PetaFLOPS FP8</td><td>1.1 TB</td><td>7.2 TB/s</td><td>400 Gbps</td><td>~10 kW</td><td>风冷</td></tr>
<tr><td>2024</td><td>GB200 NVL72</td><td>72</td><td>360 PetaFLOPS FP8 / 720 PetaFLOPS FP4</td><td>13.4 TB</td><td>130 TB/s</td><td>800 Gbps (CX-8)</td><td>~120 kW</td><td>液体</td></tr>
<tr><td>2025</td><td>GB300 NVL72</td><td>72</td><td>540 PetaFLOPS FP8 / 1,100 PetaFLOPS FP4</td><td>20.7 TB</td><td>130 TB/s</td><td>800 Gbps</td><td>~120 kW</td><td>液体</td></tr>
<tr><td>2026</td><td>NVL144</td><td>144</td><td>~1.2 ExaFLOPS FP8 / ~3.6 ExaFLOPS FP4</td><td>~21 TB</td><td>~260 TB/s*</td><td>1.6 Tbps (CX-9)</td><td>~200 kW*</td><td>液体</td></tr>
<tr><td>2027</td><td>NVL576 (Kyber)</td><td>576</td><td>~5 ExaFLOPS FP8 / ~15 ExaFLOPS FP4</td><td>~144 TB</td><td>n/d</td><td>1.6 Tbps</td><td>~600 kW*</td><td>液冷</td></tr>
</tbody>
<tbody>
<tr><td class="company" rowspan="3"><img alt="Google" class="company-logo" loading="lazy" src="images/google.png"/></td><td>2023</td><td>TPU v5p Pod</td><td>8,960</td><td>4.1 ExaFLOPS BF16</td><td>852 TB</td><td>（3D 环面）</td><td>（ICI = 纵向扩展 + 横向扩展）</td><td>n/d</td><td>液体</td></tr>
<tr><td>2025</td><td>TPU Ironwood Pod</td><td>9,216</td><td>42.5 ExaFLOPS FP8</td><td>1.77 PB</td><td>（3D 环面）</td><td>光学 OCS</td><td>~10 MW*</td><td>液体</td></tr>
<tr><td>2026</td><td>TPU v8t Sunfish Pod</td><td>9,600</td><td>121 ExaFLOPS FP4</td><td>~2 PB</td><td>(Boardfly)</td><td>光学 OCS</td><td>n/d</td><td>液体</td></tr>
</tbody>
<tbody>
<tr><td class="company" rowspan="4"><img alt="AMD" class="company-logo is-tight" loading="lazy" src="images/favicon.ico"/></td><td>2023</td><td>MI300X 8-GPU OAM</td><td>8</td><td>21 PetaFLOPS FP8</td><td>1.5 TB</td><td>7.2 TB/s</td><td>400 Gbps</td><td>~10 kW</td><td>风冷</td></tr>
<tr><td>2024</td><td>MI325X 8-GPU OAM</td><td>8</td><td>21 PetaFLOPS FP8</td><td>2.0 TB</td><td>7.2 TB/s</td><td>400 Gbps</td><td>~12 kW*</td><td>风冷</td></tr>
<tr><td>2025</td><td>MI355X 8-GPU OAM</td><td>8</td><td>80 PetaFLOPS FP8 / 160 PetaFLOPS FP4</td><td>2.3 TB</td><td>8.6 TB/s</td><td>400 Gbps</td><td>~16 kW*</td><td>液体</td></tr>
<tr><td>2026</td><td>Helios (MI455X)</td><td>72</td><td>1.4 ExaFLOPS FP8 / 2.9 ExaFLOPS FP4</td><td>31 TB</td><td>260 TB/s</td><td>n/d</td><td>n/d</td><td>液体</td></tr>
</tbody>
<tbody>
<tr><td class="company" rowspan="1"><img alt="Cerebras" class="company-logo" loading="lazy" src="images/cerebras-color.svg"/></td><td>2024</td><td>Condor Galaxy 3</td><td>64 晶圆</td><td>~1 ExaFLOPS FP16*</td><td>2.8 TB SRAM + MemoryX</td><td>（以太网树）</td><td>1.2 Tb/s 以太网</td><td>~1.5 MW*</td><td>液体</td></tr>
</tbody>
<tbody>
<tr><td class="company" rowspan="3"><img alt="AWS" class="company-logo" loading="lazy" src="images/aws.png"/></td><td>2022</td><td>Trn1 实例</td><td>16</td><td>3 PetaFLOPS BF16</td><td>512 GB</td><td>（2D 环面）</td><td>~50 Gbps (EFA)</td><td>n/d</td><td>风冷</td></tr>
<tr><td>2024</td><td>Trn2 UltraServer</td><td>64</td><td>83 PetaFLOPS FP8</td><td>6.1 TB</td><td>（3D 环形）</td><td>200 Gbps (EFAv3)</td><td>n/d</td><td>风冷</td></tr>
<tr><td>2025</td><td>Trn3 UltraServer</td><td>144</td><td>362 PetaFLOPS FP8</td><td>20.7 TB</td><td>(NeuronSwitch)</td><td>n/d</td><td>n/d</td><td>液体</td></tr>
</tbody>
<tbody>
<tr><td class="company" rowspan="2"><img alt="Groq" class="company-logo" loading="lazy" src="images/groq.png"/></td><td>2022</td><td>GroqRack</td><td>64 个活跃（已安装 72 个）</td><td>12 PetaFLOPS FP16</td><td>14 GB SRAM</td><td>3.2 TB/s 二分</td><td>（RealScale；无每芯片 NIC）</td><td>n/d</td><td>风冷</td></tr>
<tr><td>2026</td><td>NVIDIA Groq 3 LPX</td><td>256</td><td>315 PetaFLOPS FP8</td><td>128 GB SRAM + 12 TB DDR5</td><td>n/d（640 TB/s 聚合 C2C）</td><td>n/d</td><td>n/d</td><td>液冷</td></tr>
</tbody>
</table></div>
<p class="comparison-caption"><code class="notranslate" translate="no">*</code> 标记由分析师得出、由时代推断或由供应商聚合得出数字； <code class="notranslate" translate="no">n/d</code> 标记供应商尚未披露的规格。</p>
<h5 id="what-this-shows">这些数据说明了什么</h5>
<ul>
<li><strong>每芯片 FP8 已收敛。</strong> B200 (4.5 PF)、Ironwood (4.6 PF) 和 MI355X (10 PF) 彼此相差约 2 倍以内。单芯片军备竞赛已经接近尾声；机架和 Pod 是架构的分歧点。</li>
<li><strong>HBM 容量是 AMD 的持久胜利。</strong> 从 2023 年到 2025 年，每代产品的 192 → 256 → 288 GB 都可以与 NVIDIA 相媲美或击败。 NVIDIA 仅通过 B300（2025 年末）就达到了 288 GB； Rubin Ultra 将于 2026 年以 1 TB/封装重新占据领先地位。</li>
<li><strong>机架规模扩展是 NVIDIA 在 2026 年之前的胜利。</strong> GB200 / GB300 NVL72 是唯一一款在 2026 年出货的连贯机架规模域。 2024–2025； AMD 在盒子上进行了扩展，直到 Helios 才达到机架规模。 TPU 回避了这个问题：它的 torus 既是机架又是集群。</li>
<li><strong>TPU pod 的芯片数量让任何 NVIDIA 机架相形见绌。</strong> Ironwood pod = 9,216 个芯片，实现 42.5 ExaFLOPS FP8； NVL576 = 576 个 GPU，可实现约 5 ExaFLOPS FP8。 TPU 的单芯片统一速率 × 大规模 Pod 方案可以为每个系统带来更多的聚合计算，但代价是每芯片带宽。</li>
<li><strong>每个芯片的功耗正在快速上升。</strong> 700 W（Hopper）→ 1,000 W（Blackwell、MI325X）→ 1,400 W（B300、MI355X）→ ~1,800 W（Rubin Ultra，分析师）。超过 ~1,000 W 时必须采用液体冷却；空气冷却有效地以 Hopper 结束。</li>
<li><strong>横向扩展 NIC 带宽使每一代 NVIDIA 翻倍。</strong> 400 Gbps（CX-7、Hopper）→ 800 Gbps（CX-8、Blackwell）→ 1.6 Tbps（CX-9、Rubin）。 AMD 落后一代（Pollara 400 → Vulcano 800），反映出 Pensando 较小的安装基础和较晚的集成。</li>
<li><strong>Cerebras 打破了桌子的轴。</strong> 完全没有 HBM：44 GB 晶圆上 SRAM，总计 21 PB/s，每个密集 FLOP 约 1.3 字节，其中 GPU 行数接近 0.002。成本在同一行中显而易见：比单个 H200 更少的总内存、每个现代 GPU 后面的每瓦密集 FLOP 以及一个空的扩展列，因为相干域是晶圆本身。</li>
<li><strong>Trainium 在经济上竞争，而不是规格表。</strong> 它落后的每个芯片（Trn2 的 1.3 PF FP8 大约是 MI355X 的四分之一），但 Trn2 UltraServer 在 2024 年与 NVL72 一起达到了 64 芯片机架规模扩展，作为消息传递环面而不是相干 crossbar，并且 Trn3 转向交换 NeuronSwitch 结构。 AWS 拥有从 Nitro 卡到 API 的每一层，并且由一个主力租户（Anthropic，超过一百万个 Trainium2 芯片）对其进行前沿规模验证。</li>
<li><strong>Groq 用容量换取 SRAM 带宽，然后根据芯片数量扩展内存池。</strong> 第一个 GroqRack 仅公开 14 个 GB 跨 64 个活动芯片； Groq 3 LPX 在 256 个芯片上以 40 PB/s 的总 SRAM 带宽将其容量增加到 128 GB。其 12 TB DDR5 层以及与 Rubin 的搭配表明，LPU 是对大内存 GPU 机架的补充，而不是取代。</li>
</ul>
<hr/>
<!--
### Tenstorrent

<div class="philosophy">

***[Tenstorrent](https://tenstorrent.com/)***, led by Jim Keller, is building the open-source alternative. Everything is ***[RISC-V](https://en.wikipedia.org/wiki/RISC-V)***. Everything is inspectable. The philosophy prioritises programmability and accessibility over raw peak performance.

</div>

#### Architecture

Each ***Tensix core*** contains 5 small RISC-V "baby cores": 2 ***Data Mover*** cores coordinate transfers between local SRAM, off-chip DRAM, and the Network-on-Chip, and 3 ***Compute*** cores drive the matrix FPU, vector SFPU, and pack/unpack units. ~1.5 MB local SRAM per Tensix core, two independent NoC routers for concurrent bidirectional communication. The separation of data movement from computation mirrors Google's TPU — overlap transfer with math — but implemented with programmable RISC-V cores rather than fixed hardware.

***Blackhole*** has 140 Tensix cores plus 16 ***Big RISC-V*** CPU cores — dual-issue, in-order, 64-bit cores with 64 MB L2 and coherent access to 32 GB GDDR6. Powerful enough to ***run Linux***, making Blackhole a standalone AI computer that needs no host CPU. Architecturally novel — every other accelerator requires a host.

Tenstorrent uses ***GDDR6*** (32 GB, ~512 GB/s) instead of <a data-popup="hbm">HBM</a> (~8 TB/s). A massive bandwidth disadvantage, a significant cost advantage. <a data-popup="hbm">HBM</a> needs expensive 2.5D packaging with through-silicon vias; GDDR6 uses standard packaging. A Blackhole card costs &#36;1K–&#36;5K; an <a data-popup="h100">H100</a> costs &#36;25K–&#36;40K. The bet: for edge inference, development, and smaller models, the price difference matters more than peak bandwidth.

#### Scaling

Scale-out uses standard 400 Gbps ***Ethernet*** links, not proprietary interconnects. A ***Blackhole Galaxy*** meshes 32 chips in a 4×8 configuration for ~24 PetaFLOPS <a data-popup="fp8">FP8</a>. Because the interconnect is standard Ethernet, clusters can be built from off-the-shelf networking equipment. A Blackhole chip can function as compute, memory, or a high-bandwidth network switch — LEGO-like <a data-popup="cluster">cluster</a> construction.

#### Software

The stack is entirely open source. ***TT-Metalium*** is a CUDA-equivalent low-level programming model, designed from scratch for dataflow on the Tensix mesh — kernels are plain C++, with explicit data tiling, core placement, and inter-core synchronisation via circular buffers. ***TT-NN*** provides higher-level operators. ***TT-Forge / TT-MLIR*** compile PyTorch, ONNX, and JAX down to TT-Metalium. The trade-off versus CUDA: more hardware exposed (explicit NoC routing, circular buffers, per-core SRAM), giving experts more control at a higher barrier to entry.

The Register's November 2025 Blackhole QuietBox review found models running on ***Wormhole-era kernels*** — forward-compatible but unoptimised — generating tokens at speeds consistent with being artificially capped at Wormhole's 288 GB/s rather than exploiting Blackhole's full capability. The classic chicken-and-egg: you cannot optimise kernels without hardware, but unoptimised hardware does not demonstrate its value.

---
-->
<!--
### Tesla Dojo → AI5/AI6

<div class="philosophy">

Tesla's AI silicon is the purest vertical integration in the industry. One company collects the training data (cameras on millions of cars), designs the training hardware, trains the models, and deploys them on its own inference hardware (in vehicles and robots).

</div>

The ***[D1](https://en.wikipedia.org/wiki/Tesla_Dojo)*** chip — designed by ex-AMD veterans — was unlike anything else. Each core was a ***general-purpose 64-bit superscalar CPU*** with simultaneous multithreading, optimised for tensor math and custom floating-point formats (CFloat8/CFloat16). Tesla's insight: video-training workloads (millions of dashcam clips) need more flexible compute than pure matmul engines. 25 D1 chips formed a ***<a data-popup="training">Training</a> Tile*** in a 5×5 grid, 36 TB/s aggregate bandwidth via 40 I/O chips, 11 GB SRAM per tile, 9 PetaFLOPS at <a data-popup="bf16">BF16</a>. A custom 2D mesh with physical memory addressing (no virtual translation). Six tiles made a System Tray; 10 cabinets an ExaPOD.

In August 2025, Musk disbanded the Dojo team, concluding it "doesn't make sense to divide resources" between training and inference architectures. The new strategy: a unified ***AI5/AI6*** platform — inference-first, training-capable — manufactured by Samsung.

The pivot reflects a broader industry insight: ***inference now dominates AI compute budgets***. Every Tesla needs inference chips; there are millions of them. <a data-popup="training">Training</a> happens at a few datacentres. The volume economics overwhelmingly favour designing for inference and making it "pretty good" at training, not the reverse. AI6 is projected at 22,500 TOPS, 800W, serving as both the in-vehicle computer and the building block for datacentre training clusters. In January 2026, Musk announced a partial Dojo revival as AI7 — the training architecture is not fully dead, just being subsumed into the unified platform.

---
-->
<!--
### Meta MTIA

<div class="philosophy">

Meta's strategy is startlingly different: instead of designing one amazing chip, ship a ***good-enough*** chip and iterate every six months. They have shipped six generations in roughly two years (MTIA 100, 200, 300, 400, 450, 500), using modular chiplets so compute or memory can be upgraded independently.

</div>

#### Architecture

The architecture is a grid of 64 ***Processing Elements*** in an 8×8 array. Each PE contains 2 RISC-V cores (one with vector extensions), a ***Dot Product Engine*** (fixed-function matmul), a ***SIMD Engine*** (quantisation, nonlinear functions via lookup tables), a ***Reduction Engine*** (accumulation and inter-PE communication), and a ***DMA Engine***. PEs execute an ***asynchronous dataflow*** model — operations fire when inputs are ready, without centralised scheduling. Closer to a TPU's compiler-driven execution than a GPU's dynamic scheduling, but with programmable RISC-V cores doing the coordination.

MTIA v1 and v2 used ***LPDDR5*** (64–128 GB, 176–204 GB/s) instead of <a data-popup="hbm">HBM</a> — 20× less bandwidth than an <a data-popup="h100">H100</a>. This seems bizarre, but Meta's recommendation models are ***latency-sensitive at small batch sizes***, not bandwidth-hungry. The bottleneck for batch-of-one requests is compute startup latency, not sustained throughput. LPDDR5's lower cost and smaller form factor let Meta pack 24 MTIA chips into one server, giving more aggregate compute per dollar than a few <a data-popup="hbm">HBM</a> GPUs. Later generations (MTIA 400+) added <a data-popup="hbm">HBM</a> for generative workloads that really are bandwidth-bound.

Meta co-designs the hardware with the models. The ISCA 2025 paper on MTIA 2i revealed joint optimisation of model architecture, precision format, operator fusion, and chip design. When a new model architecture emerges — HSTU, a transformer-based recommendation model with 10–100× more compute per request — Meta adjusts the next chip generation to handle it. A chip designed in 2023 for anticipated 2025 workloads encounters models that look nothing like what was planned. The six-month cadence is Meta's response to this fundamental mismatch between chip design timelines and AI research velocity. MTIA 2i-based servers (24 chips) reach total performance comparable to 8-GPU servers at ***44% lower TCO*** — not because MTIA is faster, but because the chip is exactly sized and optimised for Meta's specific workload.

#### Software

MTIA supports ***PyTorch eager mode*** natively (`device='mtia'`), with hardware-accelerated job launches in <1 μs. Most non-GPU accelerators require static graph compilation, limiting iteration speed. ***Triton-MTIA*** generates kernels for the RISC-V ISA extensions. OCP (Open Compute Project) compatibility means MTIA chips mount on standard server designs alongside GPU and CPU servers.

---
-->
<!--
### Intel Gaudi

<div class="philosophy">

Intel's ***Gaudi*** (acquired from Habana Labs) makes its stand on ***open software*** and ***standard networking***. Where everyone else uses proprietary interconnects (<a data-popup="nvlink">NVLink</a>, <a data-popup="ici">ICI</a>, NeuronLink), Gaudi uses ***Ethernet***.

</div>

Gaudi 3 is a multi-chiplet design on TSMC 5nm: 64 ***Tensor Processor Cores*** (programmable — contrast NVIDIA's fixed-function <a data-popup="tensor-cores">Tensor Cores</a>), 8 ***Matrix Multiplication Engines*** (256×256 systolic arrays), and 24 integrated 200 Gb/s RoCEv2 Ethernet ports (4.8 Tb/s total networking built into the chip). The key architectural point is that MMEs, TPCs, and NICs all operate in parallel — matrix math, activations, and data transfer happen simultaneously without contending.

Integrated Ethernet means scaling uses ***standard Ethernet switches*** from any vendor. For enterprises with existing Ethernet infrastructure, this eliminates a massive capital expense. Intel positions this as "no infrastructure retrofitting needed." Raw specs lag Blackwell — 128 GB HBM2e at 3.7 TB/s vs. 192 GB <a data-popup="hbm3e">HBM3e</a> at 8 TB/s — but Intel claims 70% better price-performance than <a data-popup="h100">H100</a> on Llama 3 inference. Intel's AI roadmap has been turbulent: multiple cancellations, organisational restructuring, and the CEO's public admission that Intel may be "too late" to catch up. Dell, VMware, and other enterprise partners continue shipping Gaudi 3 systems, but the product line's long-term future is uncertain.

---
-->
<!--
### MatX

<div class="philosophy">

***[MatX](https://matx.com/)*** was founded by ***Reiner Pope*** (led TPU software at Google) and ***Mike Gunter*** (lead TPU hardware architect). 100-person team, &#36;630M total raised (including a &#36;500M Series B in February 2026), designing a chip that trades away everything except LLM performance.

</div>

The headline innovation is a ***splittable systolic array***. A traditional systolic array is a fixed grid — small matrices waste silicon. MatX's array dynamically partitions into multiple smaller arrays. This is critical for LLMs: attention heads are small (64–128 dimensions), MoE routing produces variable-shaped matrices, <a data-popup="kv-cache">KV cache</a> operations have dimensions that depend on sequence length. A fixed 256×256 array would waste most of its capacity on these. A splittable array can reconfigure into, say, four 64×256 arrays, maintaining high utilisation across the diverse matrix shapes of a transformer.

Memory is hybrid ***SRAM + <a data-popup="hbm">HBM</a>***. Most weights live in SRAM (latency advantage, like Groq); <a data-popup="hbm">HBM</a> stores the <a data-popup="kv-cache">KV cache</a> (capacity for context length). Architecturally clever — weights have predictable streaming access patterns (perfect for SRAM), while KV caches have dynamic random-access patterns (better suited to <a data-popup="hbm">HBM</a>). Numerics are custom, tuned to the statistical distributions of transformer weights and activations — claiming <a data-popup="fp4">FP4</a>'s speed with <a data-popup="fp8">FP8</a>'s accuracy by exploiting structure that generic formats waste bits representing.

No silicon has shipped. Tapeout planned within a year of February 2026, initial TSMC shipments targeted for 2027. The &#36;500M raise — led by Jane Street and Leopold Aschenbrenner's Situational Awareness fund — signals serious conviction, but this is still a bet on execution.

---
-->
<!--
### Taalas

<div class="philosophy">

Every other architecture in this post stores model weights in some form of memory (<a data-popup="hbm">HBM</a>, SRAM, GDDR, LPDDR) and moves them to compute units at runtime. ***[Taalas](https://taalas.com/)*** eliminates this entirely. Model weights are ***physically encoded into the transistor-level layout of the chip*** using mask ROM. The model does not run on the chip — ***the model is the chip***.

</div>

Taalas uses a proprietary flow that translates a model's computational graph — weights, architecture, activations, all of it — directly into the physical layout of a custom ASIC. Their ***mask ROM recall fabric*** stores 4 bits per module and performs the associated multiply with a ***single transistor***. In a conventional digital MAC, multiply-accumulate requires dozens of transistors for the multiplier, adder, and register file. Taalas collapses this to ~1 transistor per weight-multiply by hardwiring the weight value into the circuit topology itself.

The ***HC1*** runs Llama 3.1 8B at ~17,000 tokens/sec per user, sub-200ms end-to-end latency, ~200W power draw (vs. 120–600 kW for a GPU <a data-popup="rack">rack</a> doing the same work). No <a data-popup="hbm">HBM</a>, no external DRAM for weights — weights are in the silicon. An SRAM recall fabric handles the dynamic state (<a data-popup="kv-cache">KV cache</a>, activations) that changes per query. LoRA adapters and configurable context windows provide limited runtime flexibility. Base weights are immutable, fixed at fabrication.

The manufacturing trick: Taalas only modifies ***two metal layers*** on a pre-fabricated base die. TSMC can turn this around on its N6 process in ~2 months, versus 4–6 months for a full chip fabrication. A new chip for a new model takes ~2 months from the moment weights are frozen — fast enough to track the open-source release cadence (Llama, DeepSeek, Mistral). The base die — SRAM fabric, I/O, control logic, routing — is fabricated once and reused across models. Only the weight-encoding layers change per model, amortising NRE across many model-specific variants.

The radical trade-off: ***zero flexibility***. Each chip runs exactly one model. An 8B chip cannot run a 70B, cannot run DeepSeek-R1, cannot run a fine-tuned variant. Any significant update requires a new fabrication run. The chip has zero residual value once the model is superseded. The radical upside: by eliminating the <a data-popup="memory-wall">memory wall</a> entirely — weights never move because they are part of the circuit — Taalas claims 73× throughput improvement over an <a data-popup="h200">H200</a> and ~1000× improvement in performance-per-watt. If inference cost drops by 2–3 orders of magnitude, workloads that are currently uneconomical (always-on personal AI assistants, real-time AI in every IoT device) become viable.

Taalas represents the logical endpoint of the specialisation axis. NVIDIA GPUs run any model, any workload. TPUs are specialised for tensor operations. Groq is specialised for autoregressive inference. Etched's Sohu is specialised for transformer architectures. Taalas is specialised for a single specific model — the maximum possible specialisation. HC1 (Llama 3.1 8B) is live. HC2 (20B) expected summer 2026. A frontier-class chip targeted for end of 2026. Founded by ***Ljubisa Bajic*** (Tenstorrent co-founder, ex-AMD/NVIDIA architect), &#36;219M total from Quiet Capital, Fidelity, and Pierre Lamond.

---
-->
<!--
### The Axes

Every chip makes trade-offs across a few fundamental dimensions.

***Generality vs. specialisation.*** GPU (most general) → AMD GPU → Intel Gaudi → TPU → Trainium → MTIA → Groq / MatX → Etched Sohu (transformer-only) → Taalas (single-model, most specialised). More generality means a larger addressable market and ecosystem leverage, but wasted silicon on capabilities most workloads do not need. More specialisation means higher efficiency on the target workload, but brittleness when models or workloads evolve.

***Dynamic vs. static scheduling.*** GPU (fully dynamic, hardware schedulers) → TPU / Trainium (mostly static, compiler-driven) → Groq (fully static, deterministic). Dynamic adapts to unpredictable workloads at the cost of scheduling hardware and latency variance. Static achieves near-perfect utilisation on predictable workloads at the cost of adaptability.

***Memory hierarchy.*** GPU (<a data-popup="hbm">HBM</a> → L2 → L1/SMEM → registers — deep and complex). TPU (<a data-popup="hbm">HBM</a> → VMEM → MXU — shallow and explicit). Groq (SRAM only — flat and deterministic). Tenstorrent (GDDR6 + SRAM — cheap). MTIA (LPDDR5 + SRAM — cost-optimised).

***Interconnect.*** NVIDIA (proprietary <a data-popup="nvlink">NVLink</a>, highest bandwidth, vendor lock-in). Google (custom <a data-popup="ici">ICI</a> torus + optical switches, flexible, Google-controlled). AWS (custom NeuronLink, system-optimised). Intel Gaudi and Tenstorrent (standard Ethernet, open, lower bandwidth).

***Scale unit.*** NVIDIA: 8 GPUs per node, nodes over <a data-popup="infiniband">InfiniBand</a>. Google: 9,216-chip superpods with torus <a data-popup="ici">ICI</a>. AWS: 144-chip UltraServers, scaling to 1M chips. Groq: rack-level (hundreds of LPUs per model instance). Meta: 24 small chips per server.

---

### What Wins What

Large-scale training (pre-training frontier models): NVIDIA (ecosystem), TPU (Google's scale), Trainium (AWS economics). All three can do it; the choice is economics and ecosystem preference.

High-throughput batch inference (millions of users): TPU, MTIA. Systolic and purpose-built approaches shine when you can batch requests.

Low-latency real-time inference (chatbots, voice assistants): Groq (unmatched latency), NVIDIA (flexibility).

Memory-capacity-bound workloads (very large models, long contexts): AMD MI350 (288 GB <a data-popup="hbm3e">HBM3e</a>), NVIDIA <a data-popup="b200">B200</a> (192 GB + NVL72 pooling).

Cost-sensitive and edge deployment: Tenstorrent (cheap, open), Intel Gaudi (Ethernet, no retrofit), AWS Trainium (cloud economics).

Recommendation inference at hyperscale: Meta MTIA (co-designed for the exact workload).

Near-zero-cost fixed-model inference (edge devices, always-on assistants, IoT): Taalas, if model stability is acceptable.

---

### The Deeper Point

The AI chip landscape is not converging — it is ***diverging***. As workloads fragment (training vs. inference vs. reasoning, dense vs. MoE vs. diffusion, cloud vs. edge vs. on-device), optimal hardware diverges with them. The era of "one chip to rule them all" may be ending. The future likely involves heterogeneous infrastructure where different workloads run on different specialised hardware, orchestrated by software that abstracts the complexity.

The winners will be those that combine good-enough hardware with great software, tight system integration, and economic scale. NVIDIA's CUDA moat is real but not impregnable. Google's and AWS's vertical integration is powerful but captive. AMD's openness is appealing but unproven at the frontier. The specialists — Groq (now inside NVIDIA), MatX, Tenstorrent — each have a thesis that could reshape the market if the world breaks their way. Taalas is the genuinely novel endpoint of the design space: the question of whether inference should run on programmable hardware at all, or whether mature models should be "printed" into fixed silicon like an H.264 decoder or a Bitcoin ASIC.

The only certainty is that AI compute demand will continue growing faster than any single architecture can supply. This is one of the most dynamic and consequential engineering races in progress.
-->
</div>
