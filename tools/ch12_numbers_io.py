# 12장 결과 파일(assets/data/ch12_data.js)을 여러 script가 나눠 쓰기 위한 도구
# Helper so several scripts can share the chapter 12 results file (assets/data/ch12_data.js)
import json, pathlib

OUT = pathlib.Path(__file__).resolve().parent.parent / "assets" / "data" / "ch12_data.js"

def save_keys(upd):
    # ch12_data.js를 읽어 몇 항목만 바꿔 쓴다
    # Read ch12_data.js and replace only some keys
    d = {}
    if OUT.exists():
        t = OUT.read_text(encoding="utf-8"); d = json.loads(t[t.index("{"):t.rindex("}") + 1])
    d.update(upd)
    OUT.write_text("/* 12장 결과. tools/ch12_numbers.py, ch12_style.py, ch12_cifar.py가 만든다 (손으로 고치지 말 것)\n"
                   "   Chapter 12 results. Made by tools/ch12_numbers.py, ch12_style.py and ch12_cifar.py (do not edit by hand) */\n"
                   "const CH12_DATA = " + json.dumps(d, separators=(",", ":")) + ";\n", encoding="utf-8")
