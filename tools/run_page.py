# page를 headless Chrome에서 돌려 본다: 모든 버튼·slider·canvas를 건드린 뒤 JS error, 폭 넘침, 메뉴 절 수를 보고
# Run a page in headless Chrome: press every button, move every slider, poke every canvas, then report JS errors, width overflow and menu section count
# 실행 / run: python3 tools/run_page.py chNN-name.html [폭/width=1280]
import sys, os, re, subprocess, tempfile, urllib.parse, pathlib
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
src = pathlib.Path(sys.argv[1]).resolve(); width = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
base = "file://" + urllib.parse.quote(str(src.parent)) + "/"
h = src.read_text(encoding="utf-8")
head = ('<base href="' + base + '"><script>window.__errs=[];addEventListener("error",e=>__errs.push(String(e.message)+" @"+(e.filename||"").split("/").pop()+":"+e.lineno));'
        'const _w=console.warn;console.warn=(...a)=>{__errs.push("warn: "+a.join(" ").slice(0,120));_w(...a);};</script>')
ex = """<script>
setTimeout(()=>{try{
 document.querySelectorAll('main button').forEach(b=>b.click());
 document.querySelectorAll('main input[type=range]').forEach(r=>{const v=r.value;[r.min,r.max,v].forEach(x=>{r.value=x;r.dispatchEvent(new Event('input'));});});
 document.querySelectorAll('main canvas').forEach(cv=>{const b=cv.getBoundingClientRect();const o={bubbles:true,clientX:b.left+b.width*0.3,clientY:b.top+b.height*0.4,pointerId:1};
   try{cv.dispatchEvent(new PointerEvent('pointerdown',o));cv.dispatchEvent(new PointerEvent('pointermove',{...o,clientX:b.left+b.width*0.6}));cv.dispatchEvent(new PointerEvent('pointerup',o));}catch(e){__errs.push('pointer '+cv.id+': '+e.message)}});
 document.querySelectorAll('.seg button[aria-pressed=true]').forEach(b=>b.click());
}catch(e){__errs.push('exercise: '+e.message)}
setTimeout(()=>{const w=document.documentElement.scrollWidth;
 const wide=[...document.querySelectorAll('main *')].filter(e=>e.getBoundingClientRect().right>innerWidth+1&&!e.closest('.mapstage,.tbl,.eq,.codebox,pre')).slice(0,4).map(e=>e.tagName+'.'+e.className);
 document.body.setAttribute('data-result',JSON.stringify({errors:__errs,width:w+'/'+innerWidth,wide,sections:document.querySelectorAll('#side .secs a').length}));},7000);},700);
</script>"""
h = h.replace("<head>", "<head>" + head, 1).replace("</body>", ex + "</body>")
d = tempfile.mkdtemp(); t = os.path.join(d, "t.html"); open(t, "w", encoding="utf-8").write(h)
out = subprocess.run([CHROME, "--headless=new", "--allow-file-access-from-files", "--virtual-time-budget=14000", f"--window-size={width},1600", "--dump-dom", "file://" + t], capture_output=True, text=True, timeout=180).stdout
m = re.search(r'data-result="([^"]*)"', out)
print(src.name, f"@{width}px:", (m.group(1).replace("&quot;", '"') if m else "NO RESULT (page did not finish)"))
