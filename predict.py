# -*- coding: utf-8 -*-
"""
Tek gorsel icin tahmin - komut satirindan calistirilir.

Kullanim:
    python predict.py ornekler/6w5gb.png
"""

import os
import sys
import json

import numpy as np
import cv2
import torch
import torch.nn.functional as F

import onisleme as oi
from captcha_model import model_yukle, cihaz_sec

KLASOR = os.path.dirname(os.path.abspath(__file__))
MODEL_KLASORU = os.path.join(KLASOR, "model")
TTA_ACIK = True

VARYASYONLAR = [(0, 0, 1.0), (-2, 0, 1.0), (2, 0, 1.0), (0, -1, 1.0),
               (0, 1, 1.0), (0, 0, 0.97), (0, 0, 1.03)] if TTA_ACIK else [(0, 0, 1.0)]


def kaydir(gri, dx, dy, olcek=1.0):
    h, w = gri.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, olcek)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(gri, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=int(np.median(gri[0, :])))


def tahmin_et(model, alfabe, gradient_kaldir, h, w, cihaz, gri):
    islenmis = oi.on_isle(gri, gradient_kaldir=gradient_kaldir, hedef_h=h, hedef_w=w)
    toplam = None
    for dx, dy, olcek in VARYASYONLAR:
        varyant = kaydir(islenmis, dx, dy, olcek)
        girdi = torch.from_numpy((varyant.astype(np.float32) / 255.0)[None, None, :, :]).to(cihaz)
        with torch.no_grad():
            olasilik = F.softmax(model(girdi), dim=-1).cpu().numpy()[0]
        toplam = olasilik if toplam is None else toplam + olasilik
    olasilik = toplam / len(VARYASYONLAR)
    indeksler = olasilik.argmax(-1)
    karakter_guven = olasilik.max(-1)
    tahmin = "".join(alfabe[k] for k in indeksler)
    return tahmin, float(karakter_guven.min())


def main():
    if len(sys.argv) != 2:
        print("Kullanim: python predict.py <gorsel_yolu>")
        sys.exit(1)

    gorsel_yolu = sys.argv[1]

    cihaz = cihaz_sec()
    cfg = json.load(open(os.path.join(MODEL_KLASORU, "alfabe.json"), encoding="utf-8"))
    alfabe = cfg["alfabe"]
    h, w = cfg["giris_yukseklik"], cfg["giris_genislik"]
    gradient_kaldir = cfg["gradient_kaldir"]
    model = model_yukle(os.path.join(MODEL_KLASORU, "model_en_iyi.pt"), cihaz)

    # unicode yol guvenli okuma
    veri_baytlari = np.fromfile(gorsel_yolu, dtype=np.uint8)
    gri = cv2.imdecode(veri_baytlari, cv2.IMREAD_GRAYSCALE)
    if gri is None:
        print(f"HATA: gorsel okunamadi -> {gorsel_yolu}")
        sys.exit(1)

    tahmin, guven = tahmin_et(model, alfabe, gradient_kaldir, h, w, cihaz, gri)
    print(f"Tahmin: {tahmin}   (guven: {guven:.4f})")


if __name__ == "__main__":
    main()
