"""Reels için efekt seslerini üretir, müzikle karıştırır ve videoya ekler.

Kullanım: python3 ses.py <video.mp4> <muzik_kaynak.mp4> <cikti.mp4>
Efekt zamanları render.py'deki animasyon zamanlamasıyla eşleşir.
"""
import numpy as np, subprocess, sys

VIDEO, MUZIK, OUT = sys.argv[1:4]
SR, DUR = 44100, 12.0
N = int(SR * DUR)
rng = np.random.default_rng(5)

def tt(d): return np.arange(int(SR * d)) / SR

def lowpass_var(x, fc):
    """Zamanla değişen kesim frekanslı tek kutuplu alçak geçiren filtre."""
    a = 1 - np.exp(-2 * np.pi * np.asarray(fc) / SR)
    a = np.broadcast_to(a, x.shape)
    y = np.empty_like(x); s = 0.
    for i in range(len(x)):
        s += a[i] * (x[i] - s); y[i] = s
    return y

def lp2(x, fc):
    """İki kademe (12 dB/oktav) alçak geçiren: daha yumuşak, cızırtısız."""
    return lowpass_var(lowpass_var(x, fc), fc)

def bandnoise(d, f_lo, f_hi):
    n = rng.standard_normal(int(SR * d))
    return lp2(n, f_hi) - lp2(n, f_lo)

def reverb(x, dur=1.1, mix=.25):
    ir_n = int(SR * dur)
    ir = rng.standard_normal(ir_n) * np.exp(-np.arange(ir_n) / SR * 5.5)
    ir = lowpass_var(ir, 6000); ir /= np.sqrt((ir ** 2).sum())
    L = len(x) + ir_n
    nfft = 1 << (L - 1).bit_length()
    wet = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[:L]
    out = np.zeros(L); out[:len(x)] = x * (1 - mix)
    return out + wet * mix

# ---------- ses tasarımları ----------
def whoosh(d=.55, f0=300, f1=4500, peak=.55):
    t = tt(d); p = t / d
    env = np.sin(np.pi * np.clip(p / peak, 0, 1) * .5) ** 2 * np.where(p < peak, 1, np.exp(-(p - peak) / (1 - peak) * 4))
    fc = f0 + (f1 - f0) * np.sin(np.pi * p) ** 1.5
    n = rng.standard_normal(len(t))
    return (lp2(n, fc) - lp2(n, fc * .3)) * env * 2.2

def shimmer(d=1.6, base=1.0, notes=(2093, 2637, 3136, 4186, 5274), gap=.055):
    out = np.zeros(int(SR * d))
    for i, f in enumerate(notes):
        st = int(SR * i * gap); t = tt(d - i * gap)
        f = f * base
        tone = (np.sin(2 * np.pi * f * t + .002 * np.sin(2 * np.pi * 6 * t) * f)
                + .3 * np.sin(2 * np.pi * f * 2.01 * t))
        env = np.exp(-t * 3.2) * np.clip(t / .004, 0, 1)
        out[st:st + len(t)] += tone * env * (.5 - .05 * i)
    sp = bandnoise(d, 5000, 12000) * np.exp(-tt(d) * 6) * .15
    return reverb(out * .35 + sp, 1.4, .35)

def impact(d=1.6):
    t = tt(d)
    f = 42 + 110 * np.exp(-t * 14)
    ph = 2 * np.pi * np.cumsum(f) / SR
    boom = np.sin(ph) * np.exp(-t * 3.0) * np.clip(t / .003, 0, 1)
    click = bandnoise(.06, 800, 7000) * np.exp(-tt(.06) * 70)
    body = bandnoise(.5, 120, 1500) * np.exp(-tt(.5) * 9)
    out = boom * 1.0
    out[:len(click)] += click * 1.2
    out[:len(body)] += body * .6
    return reverb(np.tanh(out * 1.4) * .8, 1.6, .22)

def pop(f0=1100, f1=320, d=.09, amp=.6):
    t = tt(d)
    f = f1 + (f0 - f1) * np.exp(-t * 60)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return reverb(np.sin(ph) * np.exp(-t * 38) * amp, .5, .2)

def swipe(d=.38):
    """Üstü çizme: kalem/marker sürtünmesi."""
    t = tt(d); p = t / d
    fc = 2500 + 5000 * p
    n = rng.standard_normal(len(t)) * (1 + .5 * np.sin(2 * np.pi * 38 * t))
    x = lp2(n, fc) - lp2(n, 1500)
    env = np.clip(p / .08, 0, 1) * np.clip((1 - p) / .25, 0, 1)
    return x * env * 1.3

def tick(d=.05):
    t = tt(d)
    return np.sin(2 * np.pi * 2400 * t) * np.exp(-t * 120) * .35

def riser(d=1.0):
    t = tt(d); p = t / d
    n = rng.standard_normal(len(t))
    x = lp2(n, 400 + 4000 * p ** 2) - lp2(n, 200 + 1000 * p ** 2)
    return x * p ** 2 * 1.2

# ---------- yerleştirme ----------
sfx = np.zeros((2, N + SR * 2))
def place(x, t0, gain=1., pan=0., pan_to=None):
    """pan: -1 sol .. +1 sağ; pan_to verilirse ses boyunca kayar."""
    i = int(t0 * SR); n = min(len(x), sfx.shape[1] - i)
    pv = np.full(n, pan) if pan_to is None else np.linspace(pan, pan_to, n)
    ang = (pv + 1) * np.pi / 4
    sfx[0, i:i + n] += x[:n] * np.cos(ang) * gain * 1.414
    sfx[1, i:i + n] += x[:n] * np.sin(ang) * gain * 1.414

place(riser(.5), 0.0, .35)                                   # açılış
place(whoosh(.8, 200, 3000, .6), 0.25, .4)                  # logo yaklaşır
place(shimmer(1.8, 1.0), 0.95, .55)                          # logo belirir
place(whoosh(.5, 300, 4000, .5), 1.25, .32, -.7, -.1)        # 24 KASIM soldan
place(tick(), 1.75, .5)                                      # ayraç çizgisi
place(whoosh(.55, 300, 4200, .5), 1.55, .32, .7, .1)         # başlık sağdan
place(shimmer(1.2, .85, (2637, 3520, 4699)), 1.75, .25, -.3) # logo parıltısı
place(whoosh(.9, 250, 5000, .7), 2.45, .5, -.6, .6)          # TÜM ÇANTALAR açılır
place(shimmer(1.2, .9, (3136, 3951, 4699)), 2.7, .2, .3)     # başlık parıltısı
place(pop(700, 260, .1, .5), 3.38, 1.6, -.4)                  # 1.900 ₺
place(swipe(.4), 3.95, .55, -.7, .0)                         # üstü çizilir
place(riser(.45), 4.0, .5)                                   # 1.500 öncesi gerilim
place(impact(), 4.5, 1.0)                                    # 1.500 ₺ VURUŞ
place(shimmer(2.2, 1.0, (2093, 2637, 3136, 4186, 5274, 6272)), 4.62, .5)
place(shimmer(1.4, 1.12, (3136, 4186, 5274)), 5.25, .8, .4)  # fiyat parıltısı
place(whoosh(.4, 400, 3500, .5), 5.1, .3, -.4)
place(pop(1000, 380, .08, .55), 5.45, 1.6, -.5)               # Armine
place(pop(1250, 450, .08, .55), 5.65, 1.6, .5)                # Tiffany Tomato
place(shimmer(1.0, 1.25, (4186, 5274)), 6.55, .6, -.4)      # etiket parıltıları
place(shimmer(1.0, 1.3, (4699, 5588)), 6.75, .6, .4)
place(shimmer(1.4, .95, (2637, 3520, 4699)), 7.05, .7, -.2) # logo parıltısı
place(shimmer(1.4, 1.1, (3136, 4186, 5274)), 7.95, .75, .3)  # fiyat parıltısı
place(shimmer(1.2, .9, (2637, 3136, 3951)), 8.65, .6, -.3)  # TÜM ÇANTALAR parıltısı
place(shimmer(1.4, 1.1, (3136, 4186, 5274)), 10.05, .75, .3) # fiyat parıltısı
place(shimmer(1.4, .95, (2637, 3520, 4699)), 10.65, .6, -.2) # logo parıltısı
sfx = sfx[:, :N]

# ---------- müzik ----------
raw = subprocess.run(['ffmpeg', '-v', 'error', '-t', str(DUR), '-i', MUZIK, '-vn', '-ac', '2',
                      '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True, check=True).stdout
mus = np.frombuffer(raw, np.float32).reshape(-1, 2).T.astype(np.float64)
m = np.zeros((2, N)); m[:, :min(N, mus.shape[1])] = mus[:, :N]
t = np.arange(N) / SR
env = np.clip(t / .3, 0, 1) * np.clip((DUR - t) / 1.5, 0, 1)
# vuruş anında müziği hafifçe kıs (ducking)
duck = 1 - .35 * np.exp(-((t - 4.75) / .45) ** 2)
m *= env * duck * .8

SFX_GAIN = float(sys.argv[4]) if len(sys.argv) > 4 else .35
sfx *= SFX_GAIN
mix = m + sfx * np.clip((DUR - t) / .8, 0, 1)
peak = np.abs(mix).max()
# yumuşak limiter: -1 dBFS üstünü yuvarla, müzik seviyesine dokunma
th = .7
a = np.abs(mix)
mix = np.where(a > th, np.sign(mix) * (th + (.89 - th) * np.tanh((a - th) / (.89 - th))), mix)
pcm = (mix.T * 32767).astype('<i2').tobytes()
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', VIDEO, '-f', 's16le', '-ar', str(SR), '-ac', '2',
                '-i', '-', '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                '-movflags', '+faststart', '-shortest', OUT], input=pcm, check=True)
print('bitti', OUT, 'peak', peak)
