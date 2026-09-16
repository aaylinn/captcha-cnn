# -*- coding: utf-8 -*-
"""
PAYLASILAN MODEL TANIMI (PyTorch)
====================================
captcha_egit.py VE captcha_tahmin.py bu modulden import eder - boylece
egitimdeki ve tahmindeki mimari HER ZAMAN birebir ayni olur.

MIMARI: "PAYLASIMLI KOLON SINIFLANDIRICI"
-------------------------------------------
Ilk denemede "5 ayri sinifllandirici basi" olan klasik bir CNN kullanildi ve
model OGRENMEDI (40 epoch sonunda egitim karakter dogrulugu bile %7'de kaldi;
rastgele seviye %4.3). Sebep: 5 AYRI bas kullanildiginda, 1. pozisyondaki 'a'
ile 4. pozisyondaki 'a' AYRI AYRI ogrenilmek zorunda kaliyor -> her sinif icin
efektif veri 5 kat azaliyor.

Cozum: karakter siniflandiricisini POZISYONLAR ARASINDA PAYLASTIRMAK.
Konvolüsyon yiginindan sonra oznitelik haritasi tam 5 KOLONA indirilir (her
kolon bir karaktere denk gelir) ve AYNI Linear katmanlari butun 5 kolona
uygulanir (PyTorch'ta nn.Linear, girdinin son boyutuna otomatik + PAYLASIMLI
olarak uygulanir - Keras'taki TimeDistributed(Dense) ile ayni davranis).

Olculen fark (ayni veri, augmentasyon yok, 30 epoch):
    5 ayri bas        -> dogrulama kelime dogrulugu %0    (karakter %8)
    paylasimli kolon  -> dogrulama kelime dogrulugu %68   (karakter %92)
"""

import torch
import torch.nn as nn


class PaylasimliKolonCNN(nn.Module):
    def __init__(self, sinif_sayisi, karakter_sayisi=5):
        super().__init__()
        self.karakter_sayisi = karakter_sayisi
        self.sinif_sayisi = sinif_sayisi

        def blok(ic, dis):
            return nn.Sequential(
                nn.Conv2d(ic, dis, 3, padding=1, bias=False),
                nn.BatchNorm2d(dis),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        # Giris: (B,1,48,160) -> 4 blok sonrasi (B,128,3,10)
        # 48/16=3   160/16=10
        self.govde = nn.Sequential(blok(1, 32), blok(32, 64), blok(64, 128), blok(128, 128))
        self.kolon_havuzu = nn.MaxPool2d((1, 2))   # (B,128,3,10) -> (B,128,3,5) : tam 5 kolon = 5 karakter

        self.paylasimli_gizli = nn.Linear(3 * 128, 128)   # 384 -> 128, 5 kolonun HEPSINE paylasimli uygulanir
        self.aktivasyon = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.30)
        self.paylasimli_cikis = nn.Linear(128, sinif_sayisi)   # 5 kolonun HEPSINE paylasimli uygulanir

    def forward(self, x):
        """x: (B,1,48,160) -> logits: (B,5,sinif_sayisi)  (softmax YOK, kayip fonksiyonu icinde uygulanir)"""
        x = self.govde(x)                                   # (B,128,3,10)
        x = self.kolon_havuzu(x)                             # (B,128,3,5)
        b, c, h, w = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)       # (B,5,384) - kolon ekseni "dizi" (sequence) yapilir
        x = self.aktivasyon(self.paylasimli_gizli(x))        # Linear, son boyuta (384->128) PAYLASIMLI uygulanir
        x = self.dropout(x)
        x = self.paylasimli_cikis(x)                         # (B,5,sinif_sayisi)
        return x


def model_kaydet(model, yol, sinif_sayisi):
    torch.save({"model_state": model.state_dict(), "sinif_sayisi": sinif_sayisi}, yol)


def model_yukle(yol, cihaz):
    veri = torch.load(yol, map_location=cihaz, weights_only=False)
    model = PaylasimliKolonCNN(veri["sinif_sayisi"]).to(cihaz)
    model.load_state_dict(veri["model_state"])
    model.eval()
    return model


def cihaz_sec():
    if torch.cuda.is_available():
        ad = torch.cuda.get_device_name(0)
        print(f"GPU kullanilacak: {ad}")
        return torch.device("cuda")
    print("UYARI: GPU bulunamadi, CPU kullanilacak (yavas olur).")
    return torch.device("cpu")
