# 12장 (2): VGG-16으로 style transfer (Gatys 방식). 그림은 모두 이 script가 그린다 (사진·그림 파일을 쓰지 않는다)
# Chapter 12 (2): style transfer with VGG-16 (Gatys method). Every picture is drawn by this script (no photo or artwork files)
# 결과(PNG, base64)는 assets/data/ch12_data.js의 "style" 항목에 쓴다
# Results (PNG, base64) go to the "style" key of assets/data/ch12_data.js
# 실행 / run: python tools/ch12_style.py
import io, json, base64, pathlib, sys
import numpy as np
import torch, torchvision
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ch12_numbers_io import save_keys

dev = "mps" if torch.backends.mps.is_available() else "cpu"
S = 160

# 내용 그림: 하늘, 해, 집, 나무, 땅
# Content picture: sky, sun, house, tree, ground
def content_img():
    im = Image.new("RGB", (S, S)); d = ImageDraw.Draw(im)
    for y in range(S): d.line([(0, y), (S, y)], fill=(int(120 + 100 * y / S), int(170 + 70 * y / S), 235))
    d.rectangle([0, 112, S, S], fill=(70, 140, 60))
    d.ellipse([112, 14, 146, 48], fill=(250, 205, 60))
    d.rectangle([30, 72, 92, 124], fill=(205, 120, 70)); d.polygon([(22, 74), (61, 40), (100, 74)], fill=(150, 40, 40))
    d.rectangle([54, 96, 70, 124], fill=(90, 55, 30)); d.rectangle([36, 82, 50, 94], fill=(240, 240, 200)); d.rectangle([74, 82, 88, 94], fill=(240, 240, 200))
    d.rectangle([120, 88, 128, 122], fill=(100, 65, 35)); d.ellipse([106, 60, 142, 96], fill=(40, 110, 50))
    return im
# 무늬 그림 1: 색유리 (Voronoi 칸 + 검은 테두리), 무늬 그림 2: 굽이치는 붓결
# Style picture 1: stained glass (Voronoi cells + black borders); style picture 2: swirling brush strokes
def glass_img(seed=3):
    r = np.random.RandomState(seed); pts = r.rand(70, 2) * S; cols = r.randint(40, 255, (70, 3))
    yy, xx = np.mgrid[0:S, 0:S]; dd = np.stack([(xx - p[0]) ** 2 + (yy - p[1]) ** 2 for p in pts]); o = np.sort(dd, 0)
    img = cols[dd.argmin(0)].astype(np.float32); edge = (np.sqrt(o[1]) - np.sqrt(o[0])) < 2.2
    img[edge] = 20; return Image.fromarray(img.astype(np.uint8))
def swirl_img():
    yy, xx = np.mgrid[0:S, 0:S] / S; t = np.sin(14 * xx + 6 * np.sin(5 * yy)) + np.sin(11 * yy + 5 * np.cos(4 * xx))
    a = (t - t.min()) / (t.max() - t.min()); stripes = (np.sin(40 * a) > 0).astype(np.float32)
    c1, c2, c3 = np.array([25, 45, 120]), np.array([240, 200, 60]), np.array([70, 140, 200])
    img = (c1[None, None] * (1 - a[..., None]) + c3[None, None] * a[..., None]) * (1 - 0.55 * stripes[..., None]) + c2[None, None] * 0.55 * stripes[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

vgg = torchvision.models.vgg16(weights="IMAGENET1K_V1").features[:26].eval().to(dev)
for p in vgg.parameters(): p.requires_grad = False
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
STYLE_L, CONTENT_L = [1, 6, 11, 18, 25], 20       # relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 / relu4_2
def feats(x):
    out, h = {}, (x - MEAN) / STD
    for i, m in enumerate(vgg):
        h = m(h)
        if i in STYLE_L or i == CONTENT_L: out[i] = h
    return out
def gram(f):
    _, C, H, W = f.shape; f = f.view(C, H * W); return f @ f.T / (C * H * W)
to_t = lambda im: torch.tensor(np.asarray(im, dtype=np.float32) / 255, device=dev).permute(2, 0, 1)[None]
def png(x):
    a = (x[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8); b = io.BytesIO(); Image.fromarray(a).save(b, "PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode()

def transfer(content, style, beta, iters=300, snaps=()):
    cf, sf = feats(content), feats(style); Gs = {i: gram(sf[i]) for i in STYLE_L}
    x = content.clone().requires_grad_(True); opt = torch.optim.Adam([x], lr=0.02); shots, log = {}, []
    for it in range(iters + 1):
        if it in snaps: shots[it] = png(x)
        if it == iters: break
        opt.zero_grad(); f = feats(x)
        lc = F.mse_loss(f[CONTENT_L], cf[CONTENT_L]); ls = sum(F.mse_loss(gram(f[i]), Gs[i]) for i in STYLE_L) / len(STYLE_L)
        (lc + beta * ls).backward(); opt.step()
        with torch.no_grad(): x.clamp_(0, 1)
        if it % 50 == 0: log.append([it, round(lc.item(), 4), round(ls.item() * 1e6, 3)])
    return x, shots, log

C, G1, G2 = content_img(), glass_img(), swirl_img()
ct, g1, g2 = to_t(C), to_t(G1), to_t(G2)
res = {"content": png(ct), "styles": [png(g1), png(g2)], "beta": [], "out": [], "snaps": {}, "log": []}
BETAS = [1e3, 1e4, 1e5]
for k, st in enumerate((g1, g2)):
    outs = []
    for b in BETAS:
        x, shots, log = transfer(ct, st, b, snaps=(0, 25, 50, 100, 200, 300) if (k == 0 and b == 1e5) else ())
        outs.append(png(x)); print(f"style {k} beta {b:g}: content/style loss log", log[-1])
        if shots: res["snaps"] = {str(i): v for i, v in shots.items()}; res["log"] = log
    res["out"].append(outs)
res["beta"] = BETAS
save_keys({"style": res})
print("style images written; total base64 kB", round(sum(len(v) for v in [res["content"], *res["styles"], *sum(res["out"], []), *res["snaps"].values()]) / 1024))
