"""Harputlu AVM - Öğretmenler Günü kampanya reels animasyonu.

Kullanım (profil: canta | fincan):
  python3 hazirla.py <profil> katmanlar
  python3 render.py <profil> katmanlar video.mp4            (sessiz video)
  python3 ses.py <profil> video.mp4 muzik_kaynak.mp4 son.mp4   (müzik + efekt sesleri)
Önizleme kareleri: python3 render.py <profil> katmanlar onek 1.0,4.5,...
"""
import cv2, numpy as np, json, subprocess, sys, math
from profiller import PROFILLER, ALTIN_SUPURME, yildiz_zamanlari

PROF = PROFILLER[sys.argv[1]]
KAT = sys.argv[2] if len(sys.argv) > 2 else 'katmanlar'
OUT = sys.argv[3] if len(sys.argv) > 3 else 'harputlu_reels.mp4'
PREVIEW = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != '-' else None  # "t1,t2,..." -> sadece kare kaydet
MUZIK = sys.argv[5] if len(sys.argv) > 5 else None  # sesi alınacak video/ses dosyası
W, H, FPS, DUR = 1080, 1920, 30, 12.0

def ld(p, flag=cv2.IMREAD_UNCHANGED):
    return cv2.imread(f'{KAT}/{p}', flag)

bg = ld('arkaplan.png').astype(np.float32) / 255
meta = json.load(open(f'{KAT}/meta.json'))
L = {}
YILDIZ = meta.pop('_yildizlar', [])
_src = cv2.imread(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), PROF['kaynak'])).shape
PROF['kutular_px'] = {k: (int(a * W / _src[1]), int(b * H / _src[0]), int(c_ * W / _src[1]), int(d * H / _src[0])) for k, (a, b, c_, d) in PROF['kutular'].items()}
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
        vgrow=None, shine_p=None, glow=0., alpha_mul=None, shine_k=.75):
    """Katmanı dönüşümle tuvale yerleştirir."""
    if t_opacity <= 0: return
    ly = L[k]
    x0, y0, x1, y1 = ly['box']
    rgb, a = ly['rgb'], ly['a'].copy()
    h, w = a.shape
    if shine_p is not None and ly['glyph'] is not None:
        rgb = shine(rgb, ly['glyph'], shine_p, strength=shine_k)
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

# ---------- altın parlama + yıldız ışıltıları (ürün bölgesi) ----------
ALTIN = None
if PROF['altin_isilti']:
    ALTIN = ld('altin.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    AY0 = int(np.nonzero(ALTIN.max(axis=1) > 0)[0].min())
    ALTIN = ALTIN[AY0:]
    _yy, _xx = np.mgrid[AY0:H, 0:W].astype(np.float32)
    AU = _xx + .6 * _yy
    _ev = yildiz_zamanlari(len(YILDIZ))
    YZ = [a for a, _ in _ev]; YI = [b for _, b in _ev]

def star_sprite(r=64):
    s = 2 * r + 1
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    d = np.sqrt(xx ** 2 + yy ** 2) + 1e-3
    core = np.exp(-(d / 3.2) ** 2) + .45 * np.exp(-(d / 11) ** 2)
    rays = np.exp(-np.abs(yy) / 1.1) * np.exp(-np.abs(xx) / (r * .42)) \
         + np.exp(-np.abs(xx) / 1.1) * np.exp(-np.abs(yy) / (r * .42))
    diag = (np.exp(-np.abs(xx - yy) / 1.0) + np.exp(-np.abs(xx + yy) / 1.0)) * np.exp(-d / (r * .16)) * .5
    return np.clip(core + rays * .9 + diag, 0, 2)
STAR = star_sprite()
STAR_COL = np.array([.82, .95, 1.0], np.float32)

def altin_isilti(c, t):
    for t0, t1 in ALTIN_SUPURME:
        p = prog(t, t0, t1)
        if 0 < p < 1:
            pos = -250 + ease_io(p) * (W + .6 * H + 500)
            band = np.exp(-((AU - pos) / 110) ** 2) * ALTIN * .95
            c[AY0:] += band[..., None] * np.array([.55, .88, 1.0], np.float32)
    for ts, i in zip(YZ, YI):
        p = prog(t, ts, ts + .7)
        if 0 < p < 1:
            k = math.sin(math.pi * p) ** 1.5
            sc = .45 + .75 * k
            M = cv2.getRotationMatrix2D((STAR.shape[1] / 2, STAR.shape[0] / 2), 25 * p, sc)
            sp = cv2.warpAffine(STAR, M, STAR.shape[::-1]) * k * 2.0
            x, y = YILDIZ[i]; r = STAR.shape[0] // 2
            x0, y0 = x - r, y - r
            ax0, ay0 = max(x0, 0), max(y0, 0); ax1, ay1 = min(x0 + STAR.shape[1], W), min(y0 + STAR.shape[0], H)
            c[ay0:ay1, ax0:ax1] += sp[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0, None] * STAR_COL

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
        put(c, 'urun', wipe=ease_io(p), dy=round(24 * (1 - e)), shine_p=sp if 0 < sp < 1 else None)

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

    for i, k in enumerate(PROF['markalar']):
        t0, sd = 5.15 + .2 * i, 6.5 + .2 * i
        p = ease_out(prog(t, t0, t0 + .6))
        if p > 0:
            sp = prog(t, sd, sd + .7)
            put(c, k, t_opacity=p, dy=round(46 * (1 - p)), shine_p=sp if 0 < sp < 1 else None)

    if ALTIN is not None:
        altin_isilti(c, t)
    particles(c, t, prog(t, .3, 1.8))

    # --- kamera: yavaş içeri doğru yaklaşma (Ken Burns) ---
    z = 1.05 - .05 * ease_io(prog(t, 0, 7.5))
    if z > 1.0005:
        M = cv2.getRotationMatrix2D((W / 2, H * .52), 0, z)
        c = cv2.warpAffine(c, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # --- giriş: karanlıktan aydınlanma + spot ışık ---
    fade = ease_out(prog(t, 0, .6))
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    spot = 1 - (1 - fade) * .0 - (1 - ease_out(prog(t, 0, 1.4))) * .35 * (1 - yy)
    c *= fade * spot
    # hafif vinyet
    return np.clip(c * VIG, 0, 1)

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = (1 - .06 * (((xx - W / 2) / (W * .75)) ** 2 + ((yy - H / 2) / (H * .75)) ** 2))[..., None]
VIG = np.clip(VIG, 0, 1)

# =====================================================================
# SİNEMATİK STİL: ürünle açılış -> yazılar ritimle -> fiyat çarpması
# Müzik vuruşları (video zamanı): her 0.5 sn, vurgular 1.5 sn arayla
# =====================================================================
VURGU = [1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5]

def cam_matrix(z, fx, fy, sx=0., sy=0.):
    hw, hh = W / (2 * z), H / (2 * z)
    fx = min(max(fx, hw), W - hw); fy = min(max(fy, hh), H - hh)
    return np.float32([[z, 0, W / 2 - z * fx + sx], [0, z, H / 2 - z * fy + sy]])

def shake(t, t0, amp, dur=.35):
    if not (t0 <= t < t0 + dur): return 0., 0.
    k = amp * math.exp(-(t - t0) * 12)
    return k * math.sin((t - t0) * 95), k * math.cos((t - t0) * 77)

def rise_mask(k, p):
    """Yazı, kendi kutusunun alt çizgisinin arkasından yükselir. dy ve alpha çarpanı döndürür."""
    x0, y0, x1, y1 = L[k]['box']
    h = y1 - y0
    dy = (1 - ease_out(p)) * h * .55
    cut = (PROF['kutular_px'][k][3] + 6) - y0 - dy  # kutunun alt kenarı (katman koordinatı)
    rows = np.clip(cut - np.arange(h, dtype=np.float32), 0, 1)[:, None]
    return round(dy), rows

def center_mask(k, p, edge=90.):
    x0, y0, x1, y1 = L[k]['box']
    w = x1 - x0
    d = np.abs(np.arange(w, dtype=np.float32) - w / 2)
    return np.clip((p * (w / 2 + edge) - d) / edge, 0, 1)[None, :]

# yeni fiyat merkezi ve halka için mesafe alanı
_yb = None
def ring_field():
    global _yb
    if _yb is None:
        x0, y0, x1, y1 = PROF['kutular_px']['yeni']
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
        _yb = (cx, cy, np.sqrt((gx - cx) ** 2 + ((gy - cy) * 1.15) ** 2))
    return _yb

brng = np.random.default_rng(21)
NB = 44
B_ANG = brng.uniform(0, 2 * np.pi, NB); B_V = brng.uniform(500, 1300, NB)
B_R = brng.uniform(1.8, 4.5, NB); B_SPR = [sprite(r) for r in B_R]

def burst(c, t, t0):
    dt = t - t0
    if not (0 <= dt < 1.1): return
    cx, cy, _ = ring_field()
    for i in range(NB):
        dist = B_V[i] * (1 - math.exp(-dt * 3.2)) / 3.2
        x = cx + math.cos(B_ANG[i]) * dist * 1.0; y = cy + math.sin(B_ANG[i]) * dist * .8
        a = (1 - dt / 1.1) ** 1.5 * .9
        sp = B_SPR[i]; s = sp.shape[0]; hs = s // 2
        xi, yi = int(x) - hs, int(y) - hs
        if xi < 0 or yi < 0 or xi + s > W or yi + s > H: continue
        c[yi:yi + s, xi:xi + s] += (sp * a)[..., None] * GOLD

def ring(c, t, t0, dur=.75):
    p = prog(t, t0, t0 + dur)
    if not (0 < p < 1): return
    cx, cy, D = ring_field()
    r = 40 + 760 * ease_out(p); th = 26 * (1 - p) + 5
    y0 = int(max(cy - r * .9 - 60, 0)); y1 = int(min(cy + r * .9 + 60, H))
    band = np.exp(-((D[y0:y1] - r) / th) ** 2) * .45 * (1 - p) ** 1.3
    c[y0:y1] += band[..., None] * np.array([.55, .85, 1.0], np.float32)

def stars_at(c, t, events):
    for ts, i in events:
        p = prog(t, ts, ts + .7)
        if 0 < p < 1 and i < len(YILDIZ):
            k = math.sin(math.pi * p) ** 1.5
            M = cv2.getRotationMatrix2D((STAR.shape[1] / 2, STAR.shape[0] / 2), 25 * p, .45 + .75 * k)
            sp = cv2.warpAffine(STAR, M, STAR.shape[::-1]) * k * 2.0
            x, y = YILDIZ[i]; r = STAR.shape[0] // 2
            x0, y0 = x - r, y - r
            ax0, ay0 = max(x0, 0), max(y0, 0); ax1, ay1 = min(x0 + STAR.shape[1], W), min(y0 + STAR.shape[0], H)
            c[ay0:ay1, ax0:ax1] += sp[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0, None] * STAR_COL

def gold_sweep(c, t, t0, t1, amt=.95):
    p = prog(t, t0, t1)
    if 0 < p < 1:
        pos = -250 + ease_io(p) * (W + .6 * H + 500)
        band = np.exp(-((AU - pos) / 110) ** 2) * ALTIN * amt
        c[AY0:] += band[..., None] * np.array([.55, .88, 1.0], np.float32)

STAR_EV1 = [(.35 + i * .26, i) for i in range(9)]
STAR_EV2 = [(9.35 + ((i * 4) % 9) * .22, i) for i in range(9)]

def frame_sinematik(t):
    c = bg.copy()
    ray_amt = .06 + .04 * math.sin(t * .9)
    c += np.roll(RAYS, int(30 * math.sin(t * .35)), axis=1)[..., None] * ray_amt * np.array([.55, .85, .95], np.float32)

    # ürün ışıltıları
    if ALTIN is not None:
        gold_sweep(c, t, .5, 1.9)
        gold_sweep(c, t, 9.0, 10.2)
        stars_at(c, t, STAR_EV1 + STAR_EV2)

    # --- logo: vuruşla iner (3.0) ---
    p = prog(t, 2.8, 3.0)
    if p > 0:
        e = p * p
        sh = prog(t, 3.5, 4.2); sh2 = prog(t, 9.0, 9.7)
        sp = sh if 0 < sh < 1 else (sh2 if 0 < sh2 < 1 else None)
        put(c, 'logo', t_opacity=clamp(p * 2.5), scale=1.55 - .55 * e, blur=7 * (1 - p), shine_p=sp)

    # --- tarih / çizgi / başlık: alttan yükselir ---
    p = prog(t, 3.3, 3.85)
    if p > 0:
        dy, am = rise_mask('tarih', p); put(c, 'tarih', dy=dy, alpha_mul=am)
    p = ease_io(prog(t, 3.45, 3.9))
    if p > 0: put(c, 'cizgi', vgrow=p)
    p = prog(t, 3.5, 4.1)
    if p > 0:
        dy, am = rise_mask('baslik', p); put(c, 'baslik', dy=dy, alpha_mul=am)

    # --- ürün başlığı: ortadan açılır (4.5) ---
    p = prog(t, 4.25, 4.75)
    if p > 0:
        e = ease_out(p)
        sh = prog(t, 4.9, 5.6)
        put(c, 'urun', scale=1.06 - .06 * e, alpha_mul=center_mask('urun', e),
            shine_p=sh if 0 < sh < 1 else None)

    # --- eski fiyat düşer, 6.0'da çizilir ---
    p = prog(t, 5.15, 5.6)
    if p > 0:
        dim = 1 - .25 * prog(t, 6.3, 7.3) + .25 * prog(t, 7.5, 7.9)
        put(c, 'eski', t_opacity=clamp(p * 3) * dim, dy=round(-60 * (1 - ease_back(p, 1.4))))
    p = prog(t, 5.85, 6.05)
    if p > 0:
        put(c, 'eski_cizgi', wipe=ease_out(p))

    # --- yeni fiyat: büyükten çarpar (7.5) ---
    p = prog(t, 7.22, 7.5)
    if p > 0:
        if t < 7.5:
            s = 2.3 - 1.3 * p * p; op = clamp(p * 2.2); bl = 9 * (1 - p)
        else:
            dt = t - 7.5
            s = 1 - .06 * math.exp(-dt * 7) * math.cos(2 * math.pi * dt * 3.2); op = 1; bl = 0
            # ritim nabzı: her vuruşta (0.5 sn) hafif büyüme
            b = round(t * 2) / 2
            s *= 1 + .022 * math.exp(-((t - b) / .07) ** 2) * prog(t, 8.0, 8.3)
        glow = .9 * math.exp(-max(t - 7.5, 0) * 2.5) * prog(t, 7.35, 7.5) \
             + .2 * (.5 + .5 * math.cos(2 * math.pi * t * 2)) * prog(t, 8.0, 8.5)
        sh = [prog(t, 8.1, 8.8), prog(t, 10.4, 11.1)]
        sp = next((x for x in sh if 0 < x < 1), None)
        put(c, 'yeni', t_opacity=op, scale=s, blur=bl, glow=glow, shine_p=sp)
    ring(c, t, 7.5)
    burst(c, t, 7.5)

    particles(c, t, .6 + .4 * prog(t, 7.5, 8.0))

    # --- flaşlar ---
    fl = .10 * math.exp(-max(t - 3.0, 0) * 18) * (t >= 3.0) + .2 * math.exp(-max(t - 7.5, 0) * 14) * (t >= 7.5)
    if fl > .003: c += fl * np.array([.85, .95, 1.0], np.float32)

    # --- kamera ---
    if t < 1.5:
        z = 1.36 - .04 * ease_io(t / 1.5); fx = 520 + 40 * t / 1.5; fy = 1500
    elif t < 3.0:
        e = ease_io(prog(t, 1.5, 2.95))
        z = 1.32 + (1 - 1.32) * e; fx = 560 + (W / 2 - 560) * e; fy = 1500 + (H / 2 - 1500) * e
    else:
        z = 1.0; fx, fy = W / 2, H / 2
        z += .02 * ease_io(prog(t, 6.2, 7.45)) * (t < 7.5)           # fiyat öncesi gerilim
        for a in (9.0, 10.5):
            z += .012 * math.exp(-((t - a) / .12) ** 2)               # vurgu nabzı
    sx, sy = shake(t, 3.0, 4)
    s2 = shake(t, 7.5, 9); sx += s2[0]; sy += s2[1]
    if abs(z - 1) > 1e-4 or abs(sx) + abs(sy) > .05:
        c = cv2.warpAffine(c, cam_matrix(z, fx, fy, sx, sy), (W, H), flags=cv2.INTER_LANCZOS4,
                           borderMode=cv2.BORDER_REFLECT)

    fade = ease_out(prog(t, 0, .5))
    return np.clip(c * fade * VIG, 0, 1)

# =====================================================================
# YUMUŞAK STİL: tüm sahneyle başlar, yazılar sırayla nazikçe belirir
# (sarsıntı / flaş / patlama yok)
# =====================================================================
def soft_in(k, t, t0, dur=.8, rise=22, blur0=4., **kw):
    p = prog(t, t0, t0 + dur)
    if p <= 0: return
    e = ease_out(p)
    put(c_ref[0], k, t_opacity=e, dy=rise * (1 - e), blur=blur0 * (1 - e), **kw)

c_ref = [None]
STAR_SOFT = [(7.1, 0), (7.7, 3), (8.3, 1), (8.9, 5), (9.6, 2)]

def frame_yumusak(t):
    c = bg.copy(); c_ref[0] = c
    c += np.roll(RAYS, int(25 * math.sin(t * .3)), axis=1)[..., None] * (.06 + .03 * math.sin(t * .7)) \
         * np.array([.55, .85, .95], np.float32)

    if ALTIN is not None:
        gold_sweep(c, t, 6.8, 8.6, amt=.45)
        for ts, i in STAR_SOFT:
            p = prog(t, ts, ts + .9)
            if 0 < p < 1 and i < len(YILDIZ):
                k = math.sin(math.pi * p) ** 2
                M = cv2.getRotationMatrix2D((STAR.shape[1] / 2, STAR.shape[0] / 2), 15 * p, .35 + .45 * k)
                sp = cv2.warpAffine(STAR, M, STAR.shape[::-1]) * k * 1.0
                x, y = YILDIZ[i]; r = STAR.shape[0] // 2
                x0, y0 = x - r, y - r
                ax0, ay0 = max(x0, 0), max(y0, 0); ax1, ay1 = min(x0 + STAR.shape[1], W), min(y0 + STAR.shape[0], H)
                c[ay0:ay1, ax0:ax1] += sp[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0, None] * STAR_COL

    def sh(*wins):
        for a, b in wins:
            q = prog(t, a, b)
            if 0 < q < 1: return q
        return None

    soft_in('logo', t, .5, 1.0, rise=18, blur0=6, shine_p=sh((7.6, 8.5)), shine_k=.45)
    soft_in('tarih', t, 1.3, .9)
    p = ease_io(prog(t, 1.5, 2.2))
    if p > 0: put(c, 'cizgi', vgrow=p)
    soft_in('baslik', t, 1.55, .9)
    soft_in('urun', t, 2.5, .9, shine_p=sh((8.6, 9.5)), shine_k=.4)
    soft_in('eski', t, 3.5, .7, rise=14)
    p = prog(t, 4.2, 4.75)
    if p > 0: put(c, 'eski_cizgi', wipe=ease_io(p))

    p = prog(t, 4.85, 5.75)
    if p > 0:
        e = ease_back(p, 1.15)
        breathe = 1 + .012 * math.sin(2 * math.pi * (t - 6.0) / 2.4) * prog(t, 6.0, 7.0)
        glow = .35 * math.sin(math.pi * prog(t, 5.0, 6.4)) + .1 * prog(t, 6.4, 7.0) * (.5 + .5 * math.sin(2 * math.pi * (t - 6.4) / 2.4))
        put(c, 'yeni', t_opacity=ease_out(clamp(p * 1.6)), scale=.86 + .14 * e, blur=3 * (1 - ease_out(p)),
            glow=glow, shine_p=sh((6.3, 7.2), (10.0, 10.9)), shine_k=.5)

    particles(c, t, .35 * prog(t, .3, 2.0))

    z = 1.035 - .035 * ease_io(prog(t, 0, 6.0))
    if z > 1.0003:
        c = cv2.warpAffine(c, cam_matrix(z, W / 2, H * .5), (W, H), flags=cv2.INTER_LANCZOS4,
                           borderMode=cv2.BORDER_REFLECT)
    fade = ease_out(prog(t, 0, .8))
    return np.clip(c * fade * VIG, 0, 1)

FRAME = {'sinematik': frame_sinematik, 'yumusak': frame_yumusak}.get(PROF.get('stil'), frame)

if PREVIEW:
    for ts in PREVIEW.split(','):
        f = FRAME(float(ts))
        cv2.imwrite(f'{OUT}_{ts}.png', (f * 255 + .5).astype(np.uint8))
    sys.exit()

ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                       '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                       *(['-t', str(DUR), '-i', MUZIK, '-map', '0:v', '-map', '1:a',
                          '-af', f'afade=t=in:d=0.3,afade=t=out:st={DUR-1.5}:d=1.5',
                          '-c:a', 'aac', '-b:a', '192k', '-ar', '44100', '-shortest'] if MUZIK else []),
                       '-c:v', 'libx264', '-preset', 'slow', '-crf', '14', '-tune', 'film', '-pix_fmt', 'yuv420p',
                       '-profile:v', 'high', '-movflags', '+faststart', OUT], stdin=subprocess.PIPE)
N = int(DUR * FPS)
for i in range(N):
    f = FRAME(i / FPS)
    ff.stdin.write((f * 255 + .5).astype(np.uint8).tobytes())
    if i % 60 == 0: print(f'{i}/{N}', flush=True)
ff.stdin.close(); ff.wait()
print('bitti', OUT)
