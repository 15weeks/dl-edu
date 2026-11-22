/* DL-Edu 공통 script: 왼쪽 메뉴, 수식, 그림 도구
   Shared script: left menu, formulas, figure helpers */
"use strict";

/* 언어: <html lang="en">이면 영어, 아니면 한글. 한글 file은 이름 끝에 -KR
   Language: English when <html lang="en">, Korean otherwise. Korean files end in -KR */
const LANG = (document.documentElement.lang || "ko").toLowerCase().startsWith("en") ? "en" : "ko";
const tr = (ko, en) => (LANG === "en" ? en : ko);
const pageFile = (base, lang = LANG) => base + (lang === "en" ? ".html" : "-KR.html");

/* 다른 사이트 (DLEDU_PLAN의 장 목록은 assets/plan.js에 있다)
   Other sites (the chapter list DLEDU_PLAN lives in assets/plan.js) */
const DLEDU_LINKS = [
  { href: "https://claude.ai/artifact/AE77hA5WiMADLaMDH1JR9X", ko: "CNN 따라가기 (artifact)", en: "CNN walkthrough (artifact, Korean)" },
  { href: "https://claude.ai/artifact/BFo42Kgskf8A8iVjw8UxvF", ko: "Attention·Transformer 따라가기 (artifact)", en: "Attention and Transformer walkthrough (artifact, Korean)" },
  { href: "https://claude.ai/artifact/1jP4Z11VE2184rdbt7D9cQ", ko: "통계와 노이즈 따라가기 (artifact)", en: "Statistics and noise walkthrough (artifact, Korean)" }
];

function dleduStore(key, val) {
  try { if (val === undefined) return localStorage.getItem(key); localStorage.setItem(key, val); } catch (e) { return null; }
  return null;
}

/* 왼쪽 메뉴 만들기
   Build the left menu */
function buildSide() {
  const cur = +(document.body.dataset.chapter || 0);
  const base = document.body.dataset.base || "index";
  const side = document.createElement("nav"); side.id = "side"; side.setAttribute("aria-label", tr("장과 절 목록", "Chapters and sections"));
  let h = `<a class="site" href="${pageFile("index")}">DL-Edu<small>${tr("딥러닝을 그림으로 따라가기", "Deep learning, step by step in pictures")}</small></a>`;
  h += `<a class="lang" href="${pageFile(base, LANG === "en" ? "ko" : "en")}" hreflang="${LANG === "en" ? "ko" : "en"}">${tr("English", "한국어")}</a>`;
  h += `<h4>${tr("장 (CHAPTERS)", "CHAPTERS")}</h4><ol>`;
  DLEDU_PLAN.forEach(c => {
    const inner = `<span class="n">${c.n}</span><span>${c.name[LANG]}</span>`;
    h += `<li${c.n === cur ? ' class="cur"' : ""}>` + (c.ready ? `<a class="ch" href="${pageFile(c.file)}" title="${c.topic[LANG]}">${inner}</a>` : `<span class="ch" title="${c.topic[LANG]} · ${tr("준비 중", "coming soon")}">${inner}</span>`);
    if (c.n === cur) {
      h += `<ul class="secs">`;
      document.querySelectorAll("main section[id] > h2, main h3[id]").forEach(el => {
        const id = el.tagName === "H2" ? el.parentElement.id : el.id;
        const copy = el.cloneNode(true), no = copy.querySelector(".no");
        if (no) no.remove();
        const short = el.dataset.short || copy.textContent.trim();
        const label = (no ? no.textContent.trim() + " " : "") + short;
        h += `<li${el.tagName === "H3" ? ' class="sub"' : ""}><a href="#${id}" data-sec="${id}">${label}</a></li>`;
      });
      h += `</ul>`;
    }
    h += `</li>`;
  });
  h += `</ol><h4>${tr("다른 페이지 (LINKS)", "OTHER PAGES")}</h4><ul class="ext">` + DLEDU_LINKS.map(l => `<li><a href="${l.href}" target="_blank" rel="noopener">${l[LANG]}</a></li>`).join("") + `</ul>`;
  side.innerHTML = h;
  document.body.prepend(side);
  const btn = document.createElement("button"); btn.id = "sideToggle"; btn.type = "button"; btn.textContent = "☰";
  btn.setAttribute("aria-label", tr("메뉴 열고 닫기", "Open or close the menu")); btn.setAttribute("aria-controls", "side");
  document.body.prepend(btn);
  const narrow = () => window.matchMedia("(max-width: 960px)").matches;
  const root = document.documentElement;
  if (dleduStore("dledu.side") === "closed") root.classList.add("side-closed");
  const sync = () => btn.setAttribute("aria-expanded", narrow() ? String(root.classList.contains("side-open")) : String(!root.classList.contains("side-closed")));
  btn.addEventListener("click", () => {
    if (narrow()) root.classList.toggle("side-open");
    else { root.classList.toggle("side-closed"); dleduStore("dledu.side", root.classList.contains("side-closed") ? "closed" : "open"); }
    sync();
  });
  side.addEventListener("click", e => { if (narrow() && e.target.closest("a")) { root.classList.remove("side-open"); sync(); } });
  sync();
  /* 지금 읽는 절 표시
     Highlight the section being read */
  const links = [...side.querySelectorAll(".secs a")];
  if (links.length && "IntersectionObserver" in window) {
    const io = new IntersectionObserver(entries => {
      entries.forEach(en => { if (en.isIntersecting) { links.forEach(a => a.classList.toggle("active", a.dataset.sec === en.target.id)); } });
    }, { rootMargin: "-10% 0px -75% 0px" });
    links.forEach(a => { const t = document.getElementById(a.dataset.sec); if (t) io.observe(t); });
  }
}

/* 수식과 code 복사. 수식 안의 색: \Cb{...} 파랑, \Cg 초록, \Co 주황, \Cr 빨강, \Cv 보라 (그림의 색과 맞춘다)
   Formulas and code copy. Colours inside formulas: \Cb{...} blue, \Cg green, \Co orange, \Cr red, \Cv violet (matching the figures) */
const KATEX_OPT = {
  throwOnError: false,
  trust: ctx => ctx.command === "\\htmlClass",
  strict: code => (code === "htmlExtension" ? "ignore" : "warn"),
  macros: { "\\Cb": "\\htmlClass{c-sig}{#1}", "\\Cg": "\\htmlClass{c-ok}{#1}", "\\Co": "\\htmlClass{c-miss}{#1}", "\\Cr": "\\htmlClass{c-fa}{#1}", "\\Cv": "\\htmlClass{c-vio}{#1}" }
};
function renderMath() {
  document.querySelectorAll("[data-tex]").forEach(el => {
    const tex = el.getAttribute("data-tex");
    try { if (!window.katex) throw new Error("no katex"); katex.render(tex, el, { ...KATEX_OPT, macros: { ...KATEX_OPT.macros }, displayMode: el.classList.contains("eq") }); }
    catch (e) { el.textContent = tex; el.style.fontFamily = "var(--mono)"; }
  });
}
function addCopyButtons() {
  document.querySelectorAll(".codebox").forEach(box => {
    const pre = box.querySelector("pre"); if (!pre) return;
    const b = document.createElement("button"); b.className = "copy"; b.type = "button"; b.textContent = tr("복사", "Copy");
    b.addEventListener("click", () => {
      const done = () => { b.textContent = tr("복사됨", "Copied"); setTimeout(() => { b.textContent = tr("복사", "Copy"); }, 1200); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(pre.innerText).then(done, () => selectText(pre));
      else selectText(pre);
    });
    box.appendChild(b);
  });
}
function selectText(el) { const r = document.createRange(); r.selectNodeContents(el); const s = getSelection(); s.removeAllRanges(); s.addRange(r); }

/* 난수
   Random numbers */
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function makeRng(seed) {
  const u = mulberry32(seed); let spare = null;
  return { u, n() { if (spare !== null) { const s = spare; spare = null; return s; } let a, b, s; do { a = 2 * u() - 1; b = 2 * u() - 1; s = a * a + b * b; } while (s >= 1 || s === 0); const m = Math.sqrt(-2 * Math.log(s) / s); spare = b * m; return a * m; } };
}
let seedCounter = 1000;
const newSeed = () => (seedCounter += 7919);

/* canvas 그림 도구
   Canvas drawing helpers */
const FS = '"IBM Plex Sans KR","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif';
const FM = '"IBM Plex Mono",ui-monospace,Menlo,monospace';
function col() {
  const s = getComputedStyle(document.documentElement), g = n => s.getPropertyValue(n).trim();
  return { bg: g("--surface"), ink: g("--ink"), muted: g("--muted"), line: g("--line"), wash: g("--wash"), sig: g("--sig"), fa: g("--fa"), miss: g("--miss"), ok: g("--ok"), vio: g("--vio"), sigwash: g("--sigwash"), okwash: g("--okwash") };
}
const figs = [];
function fig(id, hf, draw) {
  const cv = document.getElementById(id);
  const f = { cv, ctx: cv.getContext("2d"), hf, draw, w: 0, h: 0, dpr: 0 };
  figs.push(f);
  new ResizeObserver(() => { if (resize(f)) paint(f); }).observe(cv);
  resize(f);
  return f;
}
function resize(f) {
  const w = Math.round(f.cv.clientWidth); if (!w) return false;
  const h = Math.round(typeof f.hf === "function" ? f.hf(w) : f.hf), dpr = window.devicePixelRatio || 1;
  if (w === f.w && h === f.h && dpr === f.dpr) return false;
  f.w = w; f.h = h; f.dpr = dpr; f.cv.width = Math.round(w * dpr); f.cv.height = Math.round(h * dpr); f.cv.style.height = h + "px";
  f.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return true;
}
function paint(f) { if (!f.w) resize(f); if (!f.w) return; const c = col(); f.ctx.clearRect(0, 0, f.w, f.h); f.draw(f.ctx, f.w, f.h, c); }
const repaintAll = () => figs.forEach(paint);
function txt(ctx, s, x, y, o = {}) {
  ctx.save(); ctx.font = (o.w || "400") + " " + (o.size || 11.5) + "px " + (o.mono ? FM : FS);
  ctx.fillStyle = o.color || "#000"; ctx.textAlign = o.align || "left"; ctx.textBaseline = o.base || "alphabetic";
  ctx.fillText(s, x, y); ctx.restore();
}
function fmtNum(v) {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e4 || a < 1e-3) { const e = Math.floor(Math.log10(a)); const m = v / Math.pow(10, e); return (Math.abs(m - 1) < 1e-9 ? "" : m.toFixed(1) + "×") + "10^" + e; }
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1).replace(/\.0$/, "");
  if (a >= 1) return v.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  return v.toPrecision(2).replace(/0+$/, "").replace(/\.$/, "");
}
/* 축과 눈금
   Axes and ticks. box = {l, t, w, h} */
function axes(ctx, c, box, xr, yr, o = {}) {
  const { l, t, w, h } = box;
  const X = v => l + (v - xr[0]) / (xr[1] - xr[0]) * w;
  const Y = v => t + h - (v - yr[0]) / (yr[1] - yr[0]) * h;
  ctx.save(); ctx.lineWidth = 1;
  (o.xt || []).forEach(it => {
    const v = Array.isArray(it) ? it[0] : it, lab = Array.isArray(it) ? it[1] : fmtNum(v), x = X(v);
    if (o.grid !== false) { ctx.strokeStyle = c.line; ctx.globalAlpha = 0.6; ctx.beginPath(); ctx.moveTo(x, t); ctx.lineTo(x, t + h); ctx.stroke(); ctx.globalAlpha = 1; }
    txt(ctx, lab, x, t + h + 14, { color: c.muted, align: "center", mono: true, size: 10.5 });
  });
  (o.yt || []).forEach(it => {
    const v = Array.isArray(it) ? it[0] : it, lab = Array.isArray(it) ? it[1] : fmtNum(v), y = Y(v);
    if (o.grid !== false) { ctx.strokeStyle = c.line; ctx.globalAlpha = 0.6; ctx.beginPath(); ctx.moveTo(l, y); ctx.lineTo(l + w, y); ctx.stroke(); ctx.globalAlpha = 1; }
    txt(ctx, lab, l - 5, y + 3.5, { color: c.muted, align: "right", mono: true, size: 10.5 });
  });
  ctx.strokeStyle = c.muted; ctx.beginPath(); ctx.moveTo(l, t); ctx.lineTo(l, t + h); ctx.lineTo(l + w, t + h); ctx.stroke();
  if (o.xlabel) txt(ctx, o.xlabel, l + w / 2, t + h + 30, { color: c.muted, align: "center", size: 11.5 });
  if (o.ylabel) txt(ctx, o.ylabel, l, t - 8, { color: c.muted, size: 11.5 });
  if (o.title) txt(ctx, o.title, l + w, t - 8, { color: c.ink, align: "right", w: "600", size: 12.5 });
  ctx.restore();
  return { X, Y };
}
function line(ctx, pts, color, width = 2, dash = null) {
  ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width; if (dash) ctx.setLineDash(dash);
  ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.stroke(); ctx.restore();
}
function dot(ctx, x, y, r, fill, stroke) { ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); } }
function arrow(ctx, x0, y0, x1, y1, color, width = 2) {
  line(ctx, [[x0, y0], [x1, y1]], color, width);
  const a = Math.atan2(y1 - y0, x1 - x0), L = 9;
  ctx.save(); ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - L * Math.cos(a - 0.4), y1 - L * Math.sin(a - 0.4)); ctx.lineTo(x1 - L * Math.cos(a + 0.4), y1 - L * Math.sin(a + 0.4)); ctx.closePath(); ctx.fill(); ctx.restore();
}
function clipBox(ctx, b) { ctx.save(); ctx.beginPath(); ctx.rect(b.l, b.t, b.w, b.h); ctx.clip(); }
function panels(w, h, n, gap, minW) {
  const side = w >= minW * n + gap * (n - 1);
  return Array.from({ length: n }, (_, i) => side ? { x: i * (w + gap) / n, y: 0, w: (w - gap * (n - 1)) / n, h } : { x: 0, y: i * h / n, w, h: h / n });
}
function stacked(w, minW, n, gap) { return w < minW * n + gap * (n - 1); }
function bindRange(id, outId, fmt, on) {
  const el = document.getElementById(id), out = outId ? document.getElementById(outId) : null;
  const upd = () => { if (out) out.textContent = fmt(+el.value); on(+el.value); };
  el.addEventListener("input", upd); if (out) out.textContent = fmt(+el.value);
  return el;
}
function bindSeg(id, on) {
  const g = document.getElementById(id);
  g.addEventListener("click", e => {
    const b = e.target.closest("button[data-v]"); if (!b) return;
    g.querySelectorAll("button[data-v]").forEach(x => x.setAttribute("aria-pressed", x === b ? "true" : "false"));
    on(b.dataset.v);
  });
}
/* 끌 수 있는 점
   Draggable handles on a canvas */
function dragOn(f, pick, move) {
  const cv = f.cv; let on = null;
  const pos = ev => { const r = cv.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; };
  cv.addEventListener("pointerdown", ev => { on = pick(...pos(ev)); if (on !== null && on !== undefined) { cv.setPointerCapture(ev.pointerId); move(on, ...pos(ev)); ev.preventDefault(); } });
  cv.addEventListener("pointermove", ev => { if (on !== null && on !== undefined) move(on, ...pos(ev)); });
  cv.addEventListener("pointerup", () => { on = null; });
  cv.addEventListener("pointercancel", () => { on = null; });
  cv.style.touchAction = "none";
}

document.addEventListener("DOMContentLoaded", () => {
  buildSide(); renderMath(); addCopyButtons();
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", repaintAll);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(repaintAll);
});
