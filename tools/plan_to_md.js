// assets/plan.js에서 PLAN_KR.md와 PLAN_EN.md를 만든다
// Generate PLAN_KR.md and PLAN_EN.md from assets/plan.js
// 실행 / run: node tools/plan_to_md.js
const fs = require("fs"), path = require("path");
const root = path.join(__dirname, "..");
const { DLEDU_PLAN, DLEDU_STAGES } = require(path.join(root, "assets", "plan.js"));

function build(lang) {
  const k = lang === "en" ? 1 : 0, T = (ko, en) => (lang === "en" ? en : ko);
  const file = c => c.file + (lang === "en" ? ".html" : "-KR.html");
  const out = [];
  out.push(T("# DL-Edu 15장 계획 (KR)", "# DL-Edu 15-chapter plan (EN)"), "");
  out.push(T("`assets/plan.js`에서 자동으로 만든 문서다. 계획을 고칠 때는 plan.js를 고치고 `node tools/plan_to_md.js`를 실행한다.",
             "Generated from `assets/plan.js`. To change the plan, edit plan.js and run `node tools/plan_to_md.js`."), "");
  out.push(T("## 한눈에 보기", "## At a glance"), "");
  out.push(T("| 단계 | 장 | 이름 | 다루는 것 | 상태 |", "| Stage | Ch | Name | Covers | Status |"), "|---|---|---|---|---|");
  DLEDU_STAGES.forEach(st => st.ch.forEach(n => {
    const c = DLEDU_PLAN.find(x => x.n === n);
    out.push(`| ${st[lang].replace("|", " ")} | ${n} | ${c.name[lang]} | ${c.topic[lang]} | ${c.ready ? T("완성", "ready") : T("계획", "planned")} |`);
  }));
  out.push("");
  DLEDU_PLAN.forEach(c => {
    out.push(`## ${c.n}. ${c.name[lang]}`, "");
    out.push(`${T("**다루는 것**", "**Covers**")}: ${c.topic[lang]}`, "");
    out.push(`${T("**목표**", "**Goal**")}: ${c.goal[lang]}`, "");
    if (c.ready) out.push(`${T("**파일**", "**File**")}: \`${file(c)}\``, "");
    out.push(T("**절**", "**Sections**"), "");
    c.sections.forEach((s, i) => out.push(`${i + 1}. ${s[k]}`));
    out.push("", T("**들어갈 interactive 그림**", "**Interactive figures**"), "");
    c.figures.forEach(s => out.push(`- ${s[k]}`));
    out.push("");
    if (c.udemy) out.push(`${T("**대응 Udemy 강의**", "**Matching Udemy lectures**")}: ${c.udemy}`, "");
    if (c.note) out.push(`${T("**참고**", "**Note**")}: ${c.note[lang]}`, "");
  });
  return out.join("\n");
}
fs.writeFileSync(path.join(root, "PLAN_KR.md"), build("ko"));
fs.writeFileSync(path.join(root, "PLAN_EN.md"), build("en"));
console.log("wrote PLAN_KR.md, PLAN_EN.md");
