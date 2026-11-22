#!/bin/bash
# 절 하나씩만 보이게 한 화면을 찍는다 (headless Chrome)
# Screenshot one section at a time (headless Chrome)
# 실행 / run: tools/shots.sh chNN-name.html 출력폴더/outdir 폭/width 높이/height head s1 s2 ...
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
src="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; out="$2"; W="$3"; H="$4"; shift 4
mkdir -p "$out"
for sec in "$@"; do
  if [ "$sec" = head ]; then css='main > section{display:none !important}'; else css="main > header.ch-head, main > section:not(#$sec){display:none !important}"; fi
  python3 - "$src" "$out/_t_$sec.html" "$css" <<'PY'
import sys, urllib.parse, pathlib
src = pathlib.Path(sys.argv[1]); base = "file://" + urllib.parse.quote(str(src.parent)) + "/"
s = src.read_text(encoding="utf-8").replace("<head>", '<head><base href="' + base + '">', 1).replace("</head>", "<style>" + sys.argv[3] + "</style></head>", 1)
open(sys.argv[2], "w", encoding="utf-8").write(s)
PY
  "$CHROME" --headless=new --allow-file-access-from-files --virtual-time-budget=9000 --window-size=$W,$H --screenshot="$out/$sec.png" "file://$out/_t_$sec.html" 2>/dev/null
  rm -f "$out/_t_$sec.html"
done
