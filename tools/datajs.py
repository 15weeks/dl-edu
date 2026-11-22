# 장 결과 파일(assets/data/chNN_data.js)을 여러 script가 나눠 쓰기 위한 도구 (14장부터)
# Helper so several scripts can share a chapter's results file (assets/data/chNN_data.js), from chapter 14 on
import json, pathlib

DATA = pathlib.Path(__file__).resolve().parent.parent / "assets" / "data"

def save_keys(ch, upd, makers):
    # chNN_data.js를 읽어 몇 항목만 바꿔 쓴다. makers = 이 파일을 만드는 script 이름들 (머리 주석용)
    # Read chNN_data.js and replace only some keys. makers = names of the scripts that write this file (for the header comment)
    p = DATA / f"ch{ch:02d}_data.js"; d = {}
    if p.exists():
        t = p.read_text(encoding="utf-8"); d = json.loads(t[t.index("{"):t.rindex("}") + 1])
    d.update(upd)
    p.write_text(f"/* {ch}장 결과. {', '.join(makers)}가 만든다 (손으로 고치지 말 것)\n"
                 f"   Chapter {ch} results. Made by {', '.join(makers)} (do not edit by hand) */\n"
                 f"const CH{ch}_DATA = " + json.dumps(d, separators=(",", ":"), ensure_ascii=False) + ";\n", encoding="utf-8")
    return p
