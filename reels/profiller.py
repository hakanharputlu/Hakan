"""Kampanya görselleri için ayarlar (koordinatlar kaynak görsel pikselidir: x0, y0, x1, y1)."""

PROFILLER = {
    # Öğretmenler Günü - Tüm Çantalar 1.500 ₺
    'canta': dict(
        kaynak='kaynak.png',
        kutular={
            'logo':    (108, 138, 842, 258),
            'tarih':   (92, 282, 306, 468),
            'cizgi':   (318, 290, 338, 462),
            'baslik':  (346, 282, 864, 466),
            'urun':    (98, 492, 864, 586),
            'eski':    (42, 606, 320, 698),
            'yeni':    (326, 588, 892, 740),
            'armine':  (58, 748, 364, 838),
            'tiffany': (406, 748, 888, 838),
        },
        bant=(125, 852),          # yazıların bulunduğu dikey bant
        koru_turuncu_y=830,       # bu y'nin altındaki turuncu-kahve pikseller (çanta kulpu) korunur
        markalar=['armine', 'tiffany'],
        altin_isilti=False,
    ),
    # Öğretmenler Günü - İkili Fincan Takımları 1.700 ₺
    'fincan': dict(
        kaynak='fincan/kaynak.png',
        kutular={
            'logo':    (110, 136, 840, 254),
            'tarih':   (104, 286, 306, 462),
            'cizgi':   (326, 292, 344, 458),
            'baslik':  (358, 286, 846, 450),
            'urun':    (58, 486, 886, 566),
            'eski':    (54, 596, 340, 690),
            'yeni':    (348, 572, 886, 732),
        },
        bant=(120, 764),          # fincanların üst kenarının üstünde biter
        koru_turuncu_y=None,
        markalar=[],
        altin_isilti=True,        # altın tepsi/fincanlarda parlama + yıldız ışıltıları
        stil='yumusak',           # sakin, yumuşak animasyon (alternatif: 'sinematik')
    ),
}


# Altın ışıltı zamanlamaları (render.py ve ses.py ortak kullanır)
ALTIN_SUPURME = [(5.25, 6.45), (9.0, 10.2)]   # ürünlerin üzerinden geçen parlama (başlangıç, bitiş)


def yildiz_zamanlari(n):
    """[(başlangıç saniyesi, yıldız no), ...]  her ışıltı ~0.7 sn sürer."""
    ev = [(5.55 + i * 0.4, i) for i in range(n)]                  # ilk tur: süpürmenin ardından sırayla
    ev += [(9.3 + ((i * 5) % n) * 0.24, i) for i in range(n)]     # ikinci tur: karışık sırayla
    return sorted(ev)
