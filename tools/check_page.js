// page 하나를 정적으로 확인: 수식이 모두 KaTeX로 풀리나, id 중복, JS가 쓰는 id가 HTML에 있나, 영어판에 한글이 남았나
// Static check of one page: every formula parses in KaTeX, no duplicate ids, ids used by the JS exist, no Korean left in an English page
// 실행 / run: node tools/check_page.js chNN-name.html chNN.js
const fs = require("fs"), path = require("path");
const katex = require(path.join(__dirname, "..", "assets", "katex", "katex.min.js"));
const [htmlFile, jsFile] = process.argv.slice(2);
const html = fs.readFileSync(htmlFile, "utf8");
const dec = s => s.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&amp;/g, "&");
const opt = { throwOnError: true, trust: c => c.command === "\\htmlClass", strict: code => (code === "htmlExtension" ? "ignore" : "error"),
  macros: { "\\Cb": "\\htmlClass{c-sig}{#1}", "\\Cg": "\\htmlClass{c-ok}{#1}", "\\Co": "\\htmlClass{c-miss}{#1}", "\\Cr": "\\htmlClass{c-fa}{#1}", "\\Cv": "\\htmlClass{c-vio}{#1}" } };
let n = 0, bad = 0;
for (const m of html.matchAll(/data-tex="([^"]*)"/g)) {
  n++;
  try { katex.renderToString(dec(m[1]), { ...opt, macros: { ...opt.macros } }); }
  catch (e) { bad++; console.log("BAD TEX:", m[1].slice(0, 70), "→", e.message.slice(0, 110)); }
}
const ids = [...html.matchAll(/\sid="([^"]+)"/g)].map(m => m[1]), dup = ids.filter((v, i) => ids.indexOf(v) !== i);
let missing = [];
if (jsFile) {
  const js = fs.readFileSync(jsFile, "utf8");
  const used = [...js.matchAll(/\$\("([^"]+)"\)|fig\("([^"]+)"|bind(?:Range|Seg)\("([^"]+)"(?:, "([^"]+)")?|setRange\("([^"]+)"/g)].flatMap(m => m.slice(1).filter(Boolean));
  missing = [...new Set(used)].filter(id => !ids.includes(id));
}
const lang = (html.match(/<html lang="([^"]+)"/) || [])[1];
const hangul = lang === "en" ? html.split("\n").filter(l => /[가-힣]/.test(l)).length : 0;
console.log(`${path.basename(htmlFile)}: tex ${n} (bad ${bad}), ids ${ids.length}, duplicate ${dup.length ? dup.join(",") : "none"}, missing in HTML ${missing.length ? missing.join(",") : "none"}${lang === "en" ? `, Korean lines ${hangul}` : ""}`);
process.exitCode = bad || dup.length || missing.length || hangul ? 1 : 0;
