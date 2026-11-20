"""Reels için özgün, telifsiz fon müziği üretir (sıcak, yumuşak, sevimli).

100 BPM, 4/4, 5 ölçü x 2.4 sn = 12 sn.  Akorlar: Fmaj7 | Em7 | Dm7 | G | Cmaj9
Ölçü 3'ün başı (4.8 sn) videoda 1.700 ₺'nin geldiği an: bas + hafif ritim orada girer.

Kullanım: python3 muzik_uret.py <cikti.wav> [baslangic_gecikmesi_sn]
"""
import numpy as np, sys, wave

OUT = sys.argv[1] if len(sys.argv) > 1 else 'fon_muzik_ozel.wav'
OFFSET = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
SR, DUR = 44100, 12.0
BEAT = 60 / 100
BAR = 4 * BEAT
N = int(SR * (DUR + 3))
rng = np.random.default_rng(42)
L = np.zeros(N); R = np.zeros(N)

def hz(m): return 440 * 2 ** ((m - 69) / 12)
def tt(d): return np.arange(int(SR * d)) / SR
def at(bar, beat): return OFFSET + bar * BAR + beat * BEAT

def add(x, t0, gain=1., pan=0.):
    i = int(t0 * SR); n = min(len(x), N - i)
    if n <= 0: return
    a = (pan + 1) * np.pi / 4
    L[i:i + n] += x[:n] * np.cos(a) * gain * 1.414
    R[i:i + n] += x[:n] * np.sin(a) * gain * 1.414

def lp(x, fc):
    a = 1 - np.exp(-2 * np.pi * fc / SR)
    y = np.empty_like(x); s = 0.
    for i in range(len(x)):
        s += a * (x[i] - s); y[i] = s
    return y

# ---------- enstrümanlar ----------
def epiano(m, d, vel=1.):
    """FM elektrik piyano (Rhodes benzeri), sıcak ve yumuşak."""
    f = hz(m); t = tt(d + 1.2)
    idx = 1.6 * np.exp(-t * 3.5) * vel
    mod = np.sin(2 * np.pi * f * t) * idx
    car = np.sin(2 * np.pi * f * t + mod) + .25 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 4)
    env = np.exp(-t * 1.1) * np.clip(t / .004, 0, 1)
    rel = np.clip((d + 1.2 - t) / 1.2, 0, 1) ** 2
    trem = 1 + .06 * np.sin(2 * np.pi * 4.5 * t)
    return car * env * rel * trem * vel * .22

def musicbox(m, d, vel=1.):
    """Müzik kutusu / celesta: parlak ama yumuşak."""
    f = hz(m); t = tt(min(d, 1.0) + 1.0)
    tone = np.sin(2 * np.pi * f * t) + .35 * np.sin(2 * np.pi * 3.01 * f * t) * np.exp(-t * 6) \
         + .12 * np.sin(2 * np.pi * 5.98 * f * t) * np.exp(-t * 10)
    env = np.exp(-t * 2.6) * np.clip(t / .002, 0, 1)
    return tone * env * vel * .16

def pad(ms, d):
    t = tt(d + 1.0)
    x = np.zeros(len(t))
    for m in ms:
        f = hz(m)
        for det in (-.12, .12):
            x += np.sin(2 * np.pi * f * (1 + det / 100) * t) + .2 * np.sin(2 * np.pi * 2 * f * t)
    env = np.clip(t / .8, 0, 1) * np.clip((d + 1.0 - t) / 1.0, 0, 1)
    return lp(x * env, 1800) * .035

def bass(m, d, vel=1.):
    f = hz(m); t = tt(d + .15)
    x = np.tanh(1.6 * (np.sin(2 * np.pi * f * t) + .3 * np.sin(4 * np.pi * f * t)))
    env = np.clip(t / .01, 0, 1) * np.exp(-t * 1.2) * np.clip((d + .15 - t) / .15, 0, 1)
    return lp(x * env, 900) * vel * .3

def kick(vel=1.):
    t = tt(.35)
    f = 48 + 70 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9) * vel * .4

def snap(vel=1.):
    t = tt(.25)
    n = rng.standard_normal(len(t))
    x = lp(n, 6000) - lp(n, 1500)
    return x * (np.exp(-t * 28) + .25 * np.exp(-t * 9)) * vel * .12

def shaker(vel=1.):
    t = tt(.09)
    n = rng.standard_normal(len(t))
    x = n - lp(n, 6500)
    return x * np.sin(np.pi * t / .09) ** 2 * vel * .03

# ---------- düzenleme ----------
CHORDS = [
    ([53, 57, 60, 64], 41),        # Fmaj7
    ([52, 55, 59, 62], 40),        # Em7
    ([50, 53, 57, 60], 38),        # Dm7
    ([55, 59, 62, 64], 43),        # G6
    ([48, 55, 59, 62, 64], 36),    # Cmaj9
]
ARP = [
    [72, 77, 81, 84, 88, 84, 81, 77],
    [71, 76, 79, 83, 86, 83, 79, 76],
    [69, 74, 77, 81, 84, 81, 77, 74],
    [71, 74, 79, 83, 86, 83, 79, 74],
]
MELODY = [  # (ölçü, vuruş, nota, süre)
    (2, 0, 81, 1.5), (2, 1.5, 79, .5), (2, 2, 77, 1), (2, 3, 76, 1),
    (3, 0, 79, 1), (3, 1, 81, .5), (3, 1.5, 83, .5), (3, 2, 86, 1), (3, 3, 84, .5), (3, 3.5, 83, .5),
    (4, 0, 84, 3),
]

for b, (notes, root) in enumerate(CHORDS):
    # pad: her ölçü boyunca
    add(pad(notes, BAR if b < 4 else BAR * 1.2), at(b, 0), 1., 0)
    # elektrik piyano ritmi
    hits = [(0, 1.0, 2.4), (2.5, .7, 1.4)] if b < 2 else ([(0, 1., 1.4), (1.5, .6, .9), (2.5, .8, 1.4)] if b < 4 else [(0, 1., 3.5)])
    for beat, v, d in hits:
        for j, m in enumerate(notes):
            add(epiano(m, d, v), at(b, beat) + j * .008, 1., -.25)
    # müzik kutusu arpej (melodi girince hafifler)
    if b < 4:
        av = .9 if b < 2 else .45
        for k, m in enumerate(ARP[b]):
            add(musicbox(m, .3, av * (1 if k % 2 == 0 else .8)), at(b, k * .5), 1., .35)
    # bas: ölçü 3'ten itibaren
    if b >= 2:
        if b < 4:
            for beat, d, v in ((0, 1.3, 1.), (1.5, .4, .6), (2.5, 1.3, .9)):
                add(bass(root, d * BEAT, v), at(b, beat))
        else:
            add(bass(root, 3.5 * BEAT, 1.), at(b, 0))
    elif b == 1:
        add(bass(root + 12, 2.0, .35), at(b, 0))  # girişte çok hafif bas

for b, beat, m, d in MELODY:
    add(musicbox(m, d * BEAT, 1.0), at(b, beat), 1.15, .15)
    add(musicbox(m + 12, d * BEAT, .18), at(b, beat) + .01, 1., .4)  # parıltı oktavı

# ritim: ölçü 3 ve 4, ölçü 5'in başı
for b in (2, 3):
    for beat in (0, 2): add(kick(1.), at(b, beat))
    add(kick(.5), at(b, 3.5))
    for beat in (1, 3): add(snap(1.), at(b, beat), 1., .1)
    for k in range(8):
        sw = .06 if k % 2 else 0.  # hafif swing
        add(shaker(1. if k % 2 else .6), at(b, k * .5 + sw), 1., -.3)
add(kick(.9), at(4, 0))
for k in range(6):
    add(shaker(.5 * (1 - k / 6)), at(4, k * .5 + (.06 if k % 2 else 0)), 1., -.3)
add(musicbox(96, 1.0, .5), at(4, 3.0), 1., .3)   # sonda küçük "ting"

# ---------- reverb + master ----------
def reverb_st(x, seed, dur=1.9):
    r = np.random.default_rng(seed)
    n = int(SR * dur)
    ir = r.standard_normal(n) * np.exp(-np.arange(n) / SR * 3.6)
    ir = lp(ir, 5000); ir /= np.sqrt((ir ** 2).sum())
    Lc = len(x) + n; nfft = 1 << (Lc - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[:len(x)]

wetL, wetR = reverb_st(L, 1), reverb_st(R, 2)
outL, outR = L * .78 + wetL * .32, R * .78 + wetR * .32
mix = np.stack([outL, outR])[:, :int(SR * DUR)]
t = np.arange(mix.shape[1]) / SR
mix *= np.clip(t / .05, 0, 1) * np.clip((DUR - t) / 1.2, 0, 1) ** 1.5
mix = np.tanh(mix / (np.abs(mix).max() + 1e-9) * 1.2) * .85
with wave.open(OUT, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix.T * 32767).astype('<i2').tobytes())
print('bitti', OUT)
