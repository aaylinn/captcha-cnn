# -*- coding: utf-8 -*-
"""
PAYLASILAN ON-ISLEME MODULU
=============================
captcha_egit.py VE captcha_tahmin.py bu modulden import eder. Boylece
egitimde ve tahminde FARKLI on-isleme yapilmasi riski (train/serve skew)
ortadan kalkar - daha once tam da bu yuzden hatalar yasamistik.

YENI ADIM: KUYRUK KIRPMA
--------------------------
Sorun: Model, goruntuyu 5 ESIT genislikte sabit dilime bolup her dilimi bir
karaktere atiyor (paylasimli kolon mimarisi). Cizgi HER ZAMAN sagdan tasiyor;
metin kisa/dar oldugunda cizginin ince kuyrugu son karakterden çok daha
sagda bitiyor. Sonuc: 5. (bazen 4.) dilim gercek harfi degil, buyuk olcude
BOS CIZGI KUYRUGUNU goruyor -> model o pozisyonda sistemik olarak hata
yapiyor (olculen: pozisyon 5 dogrulugu digerlerinden ~5-6 puan dusuk, ve
gozlemlenen hatalarin cogunda kuyrugun capraz sekli yanlislikla 'r' gibi
capraz-govdeli bir harfe benzetiliyor).

Cozum: Aga vermeden ONCE, goruntuyu GERCEK HARFLERIN bittigi yerde kirpip
(kuyrugu atip) SONRA sabit genislige yeniden olcekliyoruz. Boylece 5 dilim
her zaman gercek harflere denk gelir, kuyruk-kaynakli capitalcik ortadan
kalkar.

Nasil bulunuyor: distance-transform ile her murekkep pikselinin yerel
YARI KALINLIGI hesaplanir. Cizgi ince, harfler kalindir. "Kalin" (harf)
sayilan piksellerin en SAGDAKI sutunu = gercek metnin bittigi yer. Cizginin
ince kuyrugu bu esigin altinda kaldigi icin kirpmaya dahil olmaz.
"""

import numpy as np
import cv2

GIRIS_YUKSEKLIK = 48
GIRIS_GENISLIK = 160

KUYRUK_KIRPMA_ACIK = True
KALINLIK_YUZDELIGI = 88     # "harf govdesi" esigi (distance-transform percentile)
# NOT: 65. yuzdelik denendi ama TUM gorsellerde ayni sabit sinira (~121/135)
# takildi - cizginin kendisi de o esigin altinda "kalin" sayiliyordu (gradient
# duzlestirmenin bulanıklastirmasi cizgiye de ~2px yerel kalinlik kazandiriyor).
# 85-90 araligi, cizgiyi (sabit ince) harf govdesinden (degisken, daha kalin)
# gercekten ayirabiliyor - onceki stroke-width filtre denemelerinde de (Yontem 2,
# RefYuzde=95) benzer yuksek yuzdelikler ise yaramisti.
KIRPMA_PAYI = 6             # bulunan sinirin sagina eklenen guvenlik payi (piksel)
MIN_GENISLIK_ORANI = 0.55   # asiri agresif kirpmayi engelleyen taban (orijinal genisligin en az bu kadari kalir)


def gradient_duzlestir(gri):
    """Arka plan gradientini giderir: goruntuyu agir bulanik bir arka plan
    tahminine boler. GRI TONLAMADA kalir (binarizasyon yok -> bilgi kaybi yok)."""
    arkaplan = cv2.GaussianBlur(gri.astype(np.float32), (0, 0), sigmaX=25, sigmaY=25)
    arkaplan = np.maximum(arkaplan, 1.0)
    return np.clip((gri.astype(np.float32) / arkaplan) * 128.0, 0, 255).astype(np.uint8)


def _ink_maskesi(gri):
    _, ikili = cv2.threshold(gri, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if cv2.countNonZero(ikili) > ikili.size / 2:
        ikili = cv2.bitwise_not(ikili)
    return ikili


def kuyruk_kirp(gri, pay=KIRPMA_PAYI, min_oran=MIN_GENISLIK_ORANI):
    """Gercek (kalin) harflerin bittigi noktadan sonrasini kirpar - cizginin
    ince kuyrugunu atar. Tespit basarisiz olursa (cok az murekkep vb.)
    goruntuyu OLDUGU GIBI (kirpilmadan) doner - guvenli varsayilan."""
    w = gri.shape[1]
    ikili = _ink_maskesi(gri)
    dist = cv2.distanceTransform(ikili, cv2.DIST_L2, 5)
    degerler = dist[ikili > 0]
    if degerler.size < 10:
        return gri

    esik = np.percentile(degerler, KALINLIK_YUZDELIGI)
    kalin_kolonlar = np.where((dist >= esik).any(axis=0))[0]
    if kalin_kolonlar.size == 0:
        return gri

    sinir = int(kalin_kolonlar.max()) + 1 + pay
    taban = int(w * min_oran)
    kirpma_noktasi = max(min(sinir, w), taban)   # asiri kirpmayi da, kirpmamayi da sinirlar
    if kirpma_noktasi >= w:
        return gri
    return gri[:, :kirpma_noktasi]


def on_isle(gri, gradient_kaldir=True, kuyruk_kirp_ac=KUYRUK_KIRPMA_ACIK,
           hedef_h=GIRIS_YUKSEKLIK, hedef_w=GIRIS_GENISLIK):
    """EGITIM ve TAHMIN icin ORTAK on-isleme hattı. Sira onemli:
    1) gradient duzlestirme (varsa)
    2) kuyruk kirpma (varsa) - cizginin sag tasan ince kismini atar
    3) sabit boyuta yeniden olcekleme (kirpilmis/kisa goruntu tam genisligi
       doldurarak buyur -> 5 kolonluk siniflandirici artik gercek harflere hizalanir)
    """
    if gradient_kaldir:
        gri = gradient_duzlestir(gri)
    if kuyruk_kirp_ac:
        gri = kuyruk_kirp(gri)
    return cv2.resize(gri, (hedef_w, hedef_h), interpolation=cv2.INTER_AREA)
