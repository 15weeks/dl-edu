/* 차례 page: 15장 지도와 접히는 장 계획을 plan.js에서 그린다 (한·영 같은 code)
   Home page: draw the 15-chapter map and the collapsible chapter plan from plan.js (same code for both languages) */
"use strict";
(function () {
  const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const byN = Object.fromEntries(DLEDU_PLAN.map(c => [c.n, c]));

  // 지도: 단계마다 한 열, 장마다 상자 하나
  // Map: one column per stage, one box per chapter
  const colW = 119, gapX = 8, x0 = 12, boxH = 32, stepY = 40, top = 40;
  let svg = `<svg viewBox="0 0 780 206" role="img" aria-label="${esc(tr("15장의 지도", "Map of the 15 chapters"))}" style="font-family: var(--sans);">`;
  svg += `<defs><marker id="mh-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--muted)"/></marker></defs>`;
  DLEDU_STAGES.forEach((st, i) => {
    const x = x0 + i * (colW + gapX), lines = st[LANG].split("|");
    svg += `<text x="${x}" y="${lines.length > 1 ? 14 : 24}" font-size="11.5" font-weight="700" fill="var(--ink)">` + lines.map((t, k) => `<tspan x="${x}" dy="${k ? 13 : 0}">${esc(t)}</tspan>`).join("") + `</text>`;
    st.ch.forEach((n, k) => {
      const c = byN[n], y = top + k * stepY, href = c.ready ? pageFile(c.file) : `#ch-${n}`;
      svg += `<a href="${href}"><rect x="${x}" y="${y}" width="${colW}" height="${boxH}" rx="6" fill="var(${c.ready ? "--sigwash" : "--surface"})" stroke="var(${c.ready ? "--sig" : "--line"})" stroke-width="${c.ready ? 1.5 : 1}"/>`;
      svg += `<text x="${x + 7}" y="${y + 20}" font-size="10.8" font-weight="${c.ready ? 600 : 400}" fill="var(${c.ready ? "--ink" : "--muted"})">${n} ${esc(c.name[LANG])}</text><title>${esc(c.topic[LANG])}</title></a>`;
    });
    if (i < DLEDU_STAGES.length - 1) svg += `<path d="M${x + colW} ${top + boxH / 2}H${x + colW + gapX - 1}" fill="none" stroke="var(--muted)" stroke-width="1.3" marker-end="url(#mh-arrow)"/>`;
  });
  svg += `<rect x="12" y="166" width="756" height="30" rx="6" fill="var(--wash)"/>`;
  svg += `<text x="390" y="186" font-size="11.3" text-anchor="middle" fill="var(--muted)">${esc(tr("파란 상자 = 만든 장 (누르면 열린다) · 회색 = 준비 중 (누르면 아래 계획으로)", "Blue = written (click to open) · grey = planned (click for the plan below)"))}</text></svg>`;
  document.getElementById("homeMap").innerHTML = svg;

  // 장 계획: 장마다 접히는 상자 하나
  // Chapter plan: one collapsible box per chapter
  const list = document.getElementById("planList");
  DLEDU_PLAN.forEach(c => {
    const d = document.createElement("details"); d.id = `ch-${c.n}`; if (c.ready) d.open = true;
    let h = `<summary><span class="n">${c.n}</span><span class="t">${esc(c.name[LANG])} <span class="badge ${c.ready ? "ready" : "todo"}">${c.ready ? tr("완성", "ready") : tr("계획", "planned")}</span></span></summary><div class="body">`;
    h += `<p class="lab">${tr("다루는 것", "COVERS")}</p><p>${esc(c.topic[LANG])}</p><p class="lab">${tr("목표", "GOAL")}</p><p>${esc(c.goal[LANG])}</p>`;
    if (c.ready) h += `<p><a class="open-link" href="${pageFile(c.file)}">${tr("이 장 열기 →", "Open this chapter →")}</a></p>`;
    h += `<p class="lab">${tr("절", "SECTIONS")}</p><ol>` + c.sections.map(s => `<li>${esc(s[LANG === "en" ? 1 : 0])}</li>`).join("") + `</ol>`;
    h += `<p class="lab">${tr("들어갈 interactive 그림", "INTERACTIVE FIGURES")}</p><ul>` + c.figures.map(s => `<li>${esc(s[LANG === "en" ? 1 : 0])}</li>`).join("") + `</ul>`;
    if (c.udemy) h += `<p class="lab">${tr("대응 Udemy 강의", "MATCHING UDEMY LECTURES")}</p><p>${esc(c.udemy)}</p>`;
    if (c.note) h += `<p class="lab">${tr("참고", "NOTE")}</p><p>${esc(c.note[LANG])}</p>`;
    d.innerHTML = h + `</div>`;
    list.appendChild(d);
  });
  const btn = document.getElementById("toggleAll");
  btn.addEventListener("click", () => {
    const all = [...list.querySelectorAll("details")], open = !all.every(x => x.open);
    all.forEach(x => { x.open = open; });
    btn.textContent = open ? tr("모두 접기", "Collapse all") : tr("모두 펼치기", "Expand all");
  });
})();
