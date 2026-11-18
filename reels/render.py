"""Harputlu AVM - Öğretmenler Günü çanta kampanyası reels animasyonu.

Kullanım:  python3 hazirla.py katmanlar && python3 render.py katmanlar cikti.mp4
"""
import cv2, numpy as np, json, subprocess, sys, math

KAT = sys.argv[1] if len(sys.argv) > 1 else 'katmanlar'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'harputlu_reels.mp4'
PREVIEW = sys.argv[3] if len(sys.argv) > 3 else None  # "t1,t2,..." -> sadece kare kaydet
W, H, FPS, DUR = 1080, 1920, 30, 12.0

def ld(p, flag=cv2.IMREAD_UNCHANGED):
    return cv2.imread(f'{KAT}/{p}', flag)

bg = ld('arkaplan.png').astype(np.float32) / 255
meta = json.load(open(f'{KAT}/meta.json'))
L = {}
for k, box in meta.items():
    im = ld(f'{k}.png').astype(np.float32) / 255
    g = ld(f'{k}_parlak.png') if k != 'eski_cizgi' else None
    L[k] = dict(rgb=im[..., :3], a=im[..., 3], box=box,
                glyph=None if g is None else g.astype(np.float32) / 255)

# ---------- yardımcılar ----------
def clamp(x, a=0., b=1.): return max(a, min(b, x))
def prog(t, t0, t1): return clamp((t - t0) / (t1 - t0))
def ease_out(x): return 1 - (1 - x) ** 3
def ease_io(x): return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2
def ease_back(x, s=1.9):
    x -= 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2

def shine(rgb, glyph, p, width=34, strength=.75):
    """p: 0..1 parlama bandının konumu (soldan sağa çapraz)."""
    if p <= 0 or p >= 1: return rgb
    h, w = glyph.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xx + .45 * yy
    pos = -120 + p * (w + .45 * h + 240)
    band = np.exp(-((u - pos) / width) ** 2) * glyph * strength
    col = np.array([.80, .95, 1.0], np.float32)  # BGR: sıcak beyaz
    return np.clip(rgb + band[..., None] * col, 0, 1)

def put(canvas, k, t_opacity=1., dx=0., dy=0., scale=1., blur=0., wipe=None,
        vgrow=None, shine_p=None, glow=0., alpha_mul=None):
    """Katmanı dönüşümle tuvale yerleştirir."""
    if t_opacity <= 0: return
    ly = L[k]
    x0, y0, x1, y1 = ly['box']
    rgb, a = ly['rgb'], ly['a'].copy()
    h, w = a.shape
    if shine_p is not None and ly['glyph'] is not None:
        rgb = shine(rgb, ly['glyph'], shine_p)
    if wipe is not None:  # soldan sağa yumuşak açılma
        edge = 140.
        xs = np.arange(w, dtype=np.float32)
        ramp = np.clip((wipe * (w + edge) - xs) / edge, 0, 1)
        a = a * ramp[None, :]
    if vgrow is not None:  # ortadan dikey uzama
        ys = np.abs(np.arange(h, dtype=np.float32) - h / 2)
        ramp = np.clip((vgrow * (h / 2 + 10) - ys) / 10, 0, 1)
        a = a * ramp[:, None]
    if alpha_mul is not None: a = a * alpha_mul
    a = a * t_opacity
    pre = np.dstack([rgb * a[..., None], a])
    if glow > 0 and ly['glyph'] is not None:
        gl = cv2.GaussianBlur(ly['glyph'], (0, 0), 22) * glow
        pre[..., :3] += gl[..., None] * np.array([.35, .78, 1.0], np.float32)
    if blur > .3:
        pre = cv2.GaussianBlur(pre, (0, 0), blur)
    # dönüşüm: merkez etrafında ölçek + kaydırma, tuval koordinatına
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    # hedef kutu
    hw, hh = w * scale / 2 + 4, h * scale / 2 + 4
    X0 = int(max(math.floor(cx + dx - hw), 0)); X1 = int(min(math.ceil(cx + dx + hw), W))
    Y0 = int(max(math.floor(cy + dy - hh), 0)); Y1 = int(min(math.ceil(cy + dy + hh), H))
    if X1 <= X0 or Y1 <= Y0: return
    # katman pikseli (u,v) -> tuval (x,y): x = (u - w/2)*s + cx+dx
    M = np.array([[scale, 0, (cx + dx) - X0 - scale * w / 2],
                  [0, scale, (cy + dy) - Y0 - scale * h / 2]], np.float32)
    if abs(scale - 1) < 1e-4 and float(dx).is_integer() and float(dy).is_integer():
        ox, oy = int(cx + dx - w / 2), int(cy + dy - h / 2)
        warped = np.zeros((Y1 - Y0, X1 - X0, 4), np.float32)
        sx0, sy0 = X0 - ox, Y0 - oy
        sub = pre[max(sy0, 0):sy0 + (Y1 - Y0), max(sx0, 0):sx0 + (X1 - X0)]
        warped[max(-sy0, 0):max(-sy0, 0) + sub.shape[0], max(-sx0, 0):max(-sx0, 0) + sub.shape[1]] = sub
    else:
        warped = cv2.warpAffine(pre, M, (X1 - X0, Y1 - Y0), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    reg = canvas[Y0:Y1, X0:X1]
    reg *= (1 - warped[..., 3:])
    reg += warped[..., :3]

# ---------- ışık huzmeleri ----------
def make_rays():
    rr = np.zeros((H, W), np.float32)
    rng = np.random.default_rng(7)
    for i in range(7):
        off = rng.uniform(-700, 900); wid = rng.uniform(20, 70)
        p1 = (int(off), 0); p2 = (int(off + 1500), int(1500 * 1.25))
        cv2.line(rr, p1, p2, float(rng.uniform(.4, 1)), int(wid))
    rr = cv2.GaussianBlur(rr, (0, 0), 28)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    fall = np.exp(-((xx / W) ** 2 + (yy / (H * .55)) ** 2) * 2.2)  # sol üst köşeden
    return rr * fall
RAYS = make_rays()

# ---------- altın parçacıklar ----------
rng = np.random.default_rng(11)
NP = 46
PX = rng.uniform(0, W, NP); PY0 = rng.uniform(0, H, NP)
PV = rng.uniform(25, 75, NP); PR = rng.uniform(1.6, 5.5, NP)
PPH = rng.uniform(0, 2 * np.pi, NP); PSW = rng.uniform(8, 26, NP); PF = rng.uniform(.4, 1.1, NP)
def sprite(r):
    s = int(r * 6) | 1
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32) - s // 2
    d = np.sqrt(xx ** 2 + yy ** 2)
    core = np.exp(-(d / r) ** 2) + .35 * np.exp(-(d / (r * 2.6)) ** 2)
    return core
SPR = [sprite(r) for r in PR]
GOLD = np.array([.45, .82, 1.0], np.float32)

def particles(canvas, t, amt):
    if amt <= 0: return
    for i in range(NP):
        y = (PY0[i] - PV[i] * t) % (H + 80) - 40
        x = PX[i] + PSW[i] * math.sin(t * PF[i] + PPH[i])
        tw = .5 + .5 * math.sin(t * 2.3 * PF[i] + PPH[i] * 3)
        a = amt * (.25 + .75 * tw) * .55
        sp = SPR[i]; s = sp.shape[0]; hs = s // 2
        xi, yi = int(x) - hs, int(y) - hs
        if xi < 0 or yi < 0 or xi + s > W or yi + s > H: continue
        canvas[yi:yi + s, xi:xi + s] += (sp * a)[..., None] * GOLD

# ---------- kare üretimi ----------
def frame(t):
    c = bg.copy()
    # ışık huzmeleri: yavaş nefes alan parlaklık
    ray_amt = .07 + .045 * math.sin(t * .9)
    shift = int(30 * math.sin(t * .35))
    c += np.roll(RAYS, shift, axis=1)[..., None] * ray_amt * np.array([.55, .85, .95], np.float32)

    # --- yazı öğeleri ---
    p = ease_out(prog(t, .45, 1.45))
    if p > 0:
        sh = [prog(t, 1.7, 2.5), prog(t, 7.0, 7.8), prog(t, 10.6, 11.4)]
        sp = next((s for s in sh if 0 < s < 1), None)
        put(c, 'logo', t_opacity=p, scale=.86 + .14 * p, blur=9 * (1 - p), shine_p=sp)

    p = ease_out(prog(t, 1.3, 2.05))
    if p > 0: put(c, 'tarih', t_opacity=p, dx=round(-80 * (1 - p)))
    p = ease_io(prog(t, 1.55, 2.15))
    if p > 0: put(c, 'cizgi', vgrow=p)
    p = ease_out(prog(t, 1.6, 2.4))
    if p > 0:
        sp = prog(t, 2.6, 3.4)
        put(c, 'baslik', t_opacity=p, dx=round(80 * (1 - p)), shine_p=sp if 0 < sp < 1 else None)

    p = prog(t, 2.5, 3.35)
    if p > 0:
        e = ease_out(p)
        sp = prog(t, 8.6, 9.4)
        put(c, 'cantalar', wipe=ease_io(p), dy=round(24 * (1 - e)), shine_p=sp if 0 < sp < 1 else None)

    p = ease_out(prog(t, 3.35, 3.85))
    if p > 0: put(c, 'eski', t_opacity=p, dy=round(18 * (1 - p)))
    p = prog(t, 3.95, 4.35)
    if p > 0:
        L['eski_cizgi']['glyph'] = None
        put(c, 'eski_cizgi', wipe=ease_io(p))

    p = prog(t, 4.35, 5.0)
    if p > 0:
        s = .35 + .65 * ease_back(p)
        breathe = 1 + .022 * math.sin(2 * math.pi * (t - 6.2) / 1.7) * prog(t, 6.2, 6.8)
        glow = .9 * math.exp(-max(t - 4.75, 0) * 2.2) * prog(t, 4.5, 4.75) \
               + .25 * (.5 + .5 * math.sin(2 * math.pi * (t - 6.2) / 1.7)) * prog(t, 6.2, 6.8)
        sh = [prog(t, 5.2, 5.9), prog(t, 7.9, 8.6), prog(t, 10.0, 10.7)]
        sp = next((x for x in sh if 0 < x < 1), None)
        put(c, 'yeni', t_opacity=clamp(p * 4), scale=s * breathe, glow=glow, shine_p=sp)

    for k, t0, sd in (('armine', 5.15, 6.5), ('tiffany', 5.35, 6.7)):
        p = ease_out(prog(t, t0, t0 + .6))
        if p > 0:
            sp = prog(t, sd, sd + .7)
            put(c, k, t_opacity=p, dy=round(46 * (1 - p)), shine_p=sp if 0 < sp < 1 else None)

    particles(c, t, prog(t, .3, 1.8))

    # --- kamera: yavaş içeri doğru yaklaşma (Ken Burns) ---
    z = 1.075 - .075 * ease_io(prog(t, 0, 7.5))
    if z > 1.0005:
        M = cv2.getRotationMatrix2D((W / 2, H * .52), 0, z)
        c = cv2.warpAffine(c, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # --- giriş: karanlıktan aydınlanma + spot ışık ---
    fade = ease_out(prog(t, 0, 1.0))
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    spot = 1 - (1 - fade) * .0 - (1 - ease_out(prog(t, 0, 1.8))) * .55 * (1 - yy)
    c *= fade * spot
    # hafif vinyet
    return np.clip(c * VIG, 0, 1)

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = (1 - .12 * (((xx - W / 2) / (W * .75)) ** 2 + ((yy - H / 2) / (H * .75)) ** 2))[..., None]
VIG = np.clip(VIG, 0, 1)

if PREVIEW:
    for ts in PREVIEW.split(','):
        f = frame(float(ts))
        cv2.imwrite(f'{OUT}_{ts}.png', (f * 255 + .5).astype(np.uint8))
    sys.exit()

ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                       '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                       '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p',
                       '-profile:v', 'high', '-movflags', '+faststart', OUT], stdin=subprocess.PIPE)
N = int(DUR * FPS)
for i in range(N):
    f = frame(i / FPS)
    ff.stdin.write((f * 255 + .5).astype(np.uint8).tobytes())
    if i % 60 == 0: print(f'{i}/{N}', flush=True)
ff.stdin.close(); ff.wait()
print('bitti', OUT)
