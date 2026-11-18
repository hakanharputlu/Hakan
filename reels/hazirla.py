"""Kaynak görseli katmanlara ayırır: temiz arka plan + her yazı öğesi ayrı RGBA katman.

Kullanım: python3 hazirla.py <cikti_klasoru>
"""
import cv2, numpy as np, os, json, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else 'katmanlar'
os.makedirs(OUT, exist_ok=True)
src = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kaynak.png'))
H0, W0 = src.shape[:2]
W, H = 1080, 1920
sx, sy = W / W0, H / H0
im = cv2.resize(src, (W, H), interpolation=cv2.INTER_LANCZOS4)
O = im.astype(np.float32)

# öğe kutuları (kaynak görsel koordinatı: x0,y0,x1,y1) - sıkı, birbirine taşmayan
R = {
    'logo':     (108, 138, 842, 258),
    'tarih':    (92, 282, 306, 468),
    'cizgi':    (318, 290, 338, 462),
    'baslik':   (346, 282, 864, 466),
    'cantalar': (98, 492, 864, 586),
    'eski':     (42, 606, 320, 698),
    'yeni':     (326, 588, 892, 740),
    'armine':   (58, 748, 364, 838),
    'tiffany':  (406, 748, 888, 838),
}
R = {k: (int(a*sx), int(b*sy), int(c*sx), int(d*sy)) for k, (a, b, c, d) in R.items()}
BAND = (int(125*sy), int(852*sy))  # yazıların bulunduğu dikey bant (çanta kulpunun üstünde biter)

lum = 0.299 * O[..., 2] + 0.587 * O[..., 1] + 0.114 * O[..., 0]
red = ((O[..., 2] - O[..., 1]) > 40) & (O[..., 2] > 110)
bright = (lum > 95) | red

glyph = {}
for k, (x0, y0, x1, y1) in R.items():
    g = np.zeros((H, W), np.uint8)
    g[y0:y1, x0:x1] = bright[y0:y1, x0:x1]
    if k == 'cizgi':
        g[y0:y1, x0:x1] = (lum[y0:y1, x0:x1] > 70)
    glyph[k] = g

# ---- temiz arka plan: yazı + gölge çevresini çevreden doldur ----
G = np.zeros((H, W), np.uint8)
for g in glyph.values(): G |= g
hole = cv2.dilate(G, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
hole[:BAND[0]] = 0; hole[BAND[1]:] = 0

def push_pull(img, m):
    known = (1 - m).astype(np.float32)
    f = img * known[..., None]
    levels = []; cf, ck = f, known
    while min(cf.shape[:2]) > 8:
        levels.append((cf, ck)); cf = cv2.pyrDown(cf); ck = cv2.pyrDown(ck)
    est = cf / np.maximum(ck[..., None], 1e-4)
    for lf, lk in reversed(levels):
        up = cv2.resize(est, (lf.shape[1], lf.shape[0]), interpolation=cv2.INTER_CUBIC)
        sm = cv2.GaussianBlur(lf, (0, 0), 2); sk = cv2.GaussianBlur(lk, (0, 0), 2)
        local = sm / np.maximum(sk[..., None], 1e-4)
        w = np.clip(sk * 1.5, 0, 1)[..., None]
        est = local * w + up * (1 - w)
    return est

hmb = (((O[..., 2] - O[..., 1]) > 20) & (O[..., 2] > 70)).astype(np.uint8)
hmb[:int(830*sy)] = 0
hmb = cv2.dilate(hmb, np.ones((9, 9), np.uint8))
fill = cv2.GaussianBlur(push_pull(O, hole | hmb), (0, 0), 10)  # kulp rengi dolguya karışmasın
fill += np.random.default_rng(3).normal(0, 1.4, fill.shape).astype(np.float32)
soft = cv2.GaussianBlur(cv2.dilate(hole, np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 6)[..., None]
# bandın alt kenarında yumuşakça sıfırla: altındaki çanta kulpu orijinal kalsın
vr = np.clip((BAND[1] - np.arange(H, dtype=np.float32)) / 14, 0, 1)[:, None, None]
soft = soft * vr
# çanta kulpu (turuncu-kahve pikseller) asla doldurulmasın
hm = (((O[..., 2] - O[..., 1]) > 20) & (O[..., 2] > 70)).astype(np.uint8)
hm[:int(830*sy)] = 0
hm = cv2.GaussianBlur(cv2.dilate(hm, np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 1.5)
soft = soft * (1 - np.clip(hm, 0, 1))[..., None]
C = O * (1 - soft) + fill * soft
C = np.clip(C, 0, 255)
cv2.imwrite(f'{OUT}/arkaplan.png', C.astype(np.uint8))
cv2.imwrite(f'{OUT}/tam.png', im)

# ---- yumuşak sahiplik: her piksel en yakın öğeye ait ----
keys = list(R)
D = np.stack([cv2.distanceTransform((1 - glyph[k]).astype(np.uint8), cv2.DIST_L2, 5) for k in keys])
Wk = np.exp(-(D - D.min(axis=0, keepdims=True)) / 3.0)
Wk /= Wk.sum(axis=0, keepdims=True)
bandm = np.zeros((H, W), np.float32); bandm[BAND[0]:BAND[1]] = 1
bandm = cv2.GaussianBlur(bandm, (0, 0), 4)

def matte(img):
    d = np.abs(img - C).max(axis=2)
    a = np.clip((d - 4) / 20, 0, 1)
    return cv2.GaussianBlur(a, (0, 0), .7) * bandm

def unpremult(img, a):
    F = (img - (1 - a[..., None]) * C) / np.maximum(a[..., None], 1e-3)
    return np.clip(np.where(a[..., None] > .02, F, img), 0, 255)

A = matte(O)
F = unpremult(O, A)

meta = {}
PAD = 50
def save(name, k, Fimg, alpha, gl):
    x0, y0, x1, y1 = R[k]
    ys, xs = np.nonzero(alpha > 0.004)
    X0, Y0 = max(min(xs.min(), x0) - PAD, 0), max(min(ys.min(), y0) - PAD, 0)
    X1, Y1 = min(max(xs.max(), x1) + PAD, W), min(max(ys.max(), y1) + PAD, H)
    rgba = np.dstack([Fimg[Y0:Y1, X0:X1], alpha[Y0:Y1, X0:X1] * 255])
    cv2.imwrite(f'{OUT}/{name}.png', np.clip(rgba + .5, 0, 255).astype(np.uint8))
    if gl is not None:
        g = cv2.GaussianBlur(gl.astype(np.float32), (0, 0), .8)[Y0:Y1, X0:X1]
        cv2.imwrite(f'{OUT}/{name}_parlak.png', (g * 255).astype(np.uint8))
    meta[name] = [int(X0), int(Y0), int(X1), int(Y1)]

for i, k in enumerate(keys):
    if k == 'eski':
        x0, y0, x1, y1 = R[k]
        rmask = np.zeros((H, W), np.uint8); rmask[y0:y1, x0:x1] = red[y0:y1, x0:x1]
        rmask = cv2.dilate(rmask, np.ones((5, 5), np.uint8))
        # çizgisiz eski fiyat
        Ob = O.copy()
        Ob[y0-20:y1+20, x0-20:x1+20] = cv2.inpaint(im[y0-20:y1+20, x0-20:x1+20], rmask[y0-20:y1+20, x0-20:x1+20], 5, cv2.INPAINT_TELEA)
        Ab = matte(Ob)
        save('eski', k, unpremult(Ob, Ab), Ab * Wk[i], glyph[k] & (1 - rmask))
        ra = cv2.GaussianBlur(rmask.astype(np.float32), (0, 0), 1.0)
        save('eski_cizgi', k, O, ra, None)
        continue
    save(k, k, F, A * Wk[i], glyph[k])

json.dump(meta, open(f'{OUT}/meta.json', 'w'), indent=1)
print('ok')
