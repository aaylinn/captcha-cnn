# -*- coding: utf-8 -*-
"""
CAPTCHA-CNN - TEK GORSEL ICIN TAHMIN
========================================
Bu klasordeki egitilmis modeli kullanarak tek bir CAPTCHA gorseli icin
metin tahmini yapar. Bagimsiz calisir - sadece bu klasordeki dosyalara
(model/model_en_iyi.pt, model/alfabe.json, captcha_model.py, onisleme.py)
ihtiyac duyar.

Test-time augmentation (TTA): tahmin sirasinda goruntunun 7 hafif
varyasyonu (kaydirma/olcek) uzerinden ayri ayri tahmin alinip
olasiliklar ortalanir - README'deki %97.02 dogruluk bu TTA ile olculdu.

Kurulum:
    pip install -r requirements.txt

Kullanim:
    python tahmin_et.py yol/to/captcha.png
"""
import sys
import os
import json

import numpy as np
import cv2
import torch
import torch.nn.functional as F

import onisleme as oi
from captcha_model import model_yukle, cihaz_sec

KLASOR = os.path.dirname(os.path.abspath(__file__))
MODEL_YOLU = os.path.join(KLASOR, "model", "model_en_iyi.pt")
ALFABE_YOLU = os.path.join(KLASOR, "model", "alfabe.json")

VARYASYONLAR = [(0, 0, 1.0), (-2, 0, 1.0), (2, 0, 1.0), (0, -1, 1.0),
                (0, 1, 1.0), (0, 0, 0.97), (0, 0, 1.03)]


def gorsel_oku(yol):
    """Windows'ta Turkce/unicode yol sorunu icin cv2.imread yerine
    np.fromfile + cv2.imdecode kullanilir."""
    veri = np.fromfile(yol, dtype=np.uint8)
    return cv2.imdecode(veri, cv2.IMREAD_GRAYSCALE)


def _kaydir(gri, dx, dy, olcek=1.0):
    h, w = gri.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, olcek)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(gri, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=int(np.median(gri[0, :])))


def tahmin_et(gorsel_yolu, model=None, cihaz=None, alfabe=None, gradient_kaldir=True):
    """Tek bir gorsel dosyasi icin (metin, guven) dondurur.
    guven = tum karakterlerin en dusuk softmax olasiligi (zayif halka)."""
    if model is None:
        cihaz = cihaz_sec()
        model = model_yukle(MODEL_YOLU, cihaz)
        with open(ALFABE_YOLU, encoding="utf-8") as f:
            alfabe = json.load(f)["alfabe"]

    gri = gorsel_oku(gorsel_yolu)
    if gri is None:
        raise ValueError(f"Gorsel okunamadi: {gorsel_yolu}")

    islenmis = oi.on_isle(gri, gradient_kaldir=gradient_kaldir)

    toplam = None
    for dx, dy, olcek in VARYASYONLAR:
        varyant = _kaydir(islenmis, dx, dy, olcek)
        x = torch.from_numpy((varyant.astype(np.float32) / 255.0)[None, None, :, :]).to(cihaz)
        with torch.no_grad():
            olasilik = F.softmax(model(x), dim=-1).cpu().numpy()[0]
        toplam = olasilik if toplam is None else toplam + olasilik
    olasilik = toplam / len(VARYASYONLAR)

    indeksler = olasilik.argmax(-1)
    karakter_guven = olasilik.max(-1)
    tahmin = "".join(alfabe[k] for k in indeksler)
    guven = float(karakter_guven.min())
    return tahmin, guven


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Kullanim: python tahmin_et.py yol/to/captcha.png")
        sys.exit(1)

    gorsel_yolu = sys.argv[1]
    if not os.path.exists(gorsel_yolu):
        print(f"HATA: dosya bulunamadi: {gorsel_yolu}")
        sys.exit(1)

    tahmin, guven = tahmin_et(gorsel_yolu)
    print(f"Tahmin: {tahmin}  (guven: %{guven*100:.1f})")
