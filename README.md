# Captcha Text Recognition — CNN

A CNN that reads 5-character alphanumeric text from gradient-background CAPTCHA images. Built to study a CNN architecture problem in practice: a naive "5 separate classifier heads" design failed to learn, and the fix (sharing the classifier across character positions) took character accuracy from ~7% to ~92%.

---

## Problem

Her görselde 5 karakterlik (küçük harf + rakam) bir metin var, soldan sağa koyu→açık bir gradyan arka plan ve sağdan sarkan rastgele bir çizgi ile zorlaştırılmış.

**İlk deneme:** Klasik yaklaşım — CNN gövdesi + 5 ayrı sınıflandırıcı "baş" (her karakter pozisyonu için bir tane). 40 epoch sonunda eğitim karakter doğruluğu **%7**'de kaldı (rastgele seviye %4.3 — model neredeyse hiçbir şey öğrenmedi).

**Kök neden:** 5 ayrı baş kullanıldığında, 1. pozisyondaki "a" ile 4. pozisyondaki "a" birbirinden bağımsız öğrenilmek zorunda kalıyor — her sınıf için efektif veri miktarı 5 kat azalıyor.

**Çözüm — "Paylaşımlı Kolon Sınıflandırıcı":** Konvolüsyon yığınından sonra öznitelik haritası tam 5 kolona indiriliyor (her kolon bir karaktere denk geliyor) ve **aynı** `Linear` katmanları tüm 5 kolona paylaşımlı olarak uygulanıyor (PyTorch'ta `nn.Linear`, Keras'taki `TimeDistributed(Dense)` ile aynı davranış).

| Mimari | Doğrulama kelime doğruluğu | Karakter doğruluğu |
|---|---|---|
| 5 ayrı baş | %0 | %8 |
| Paylaşımlı kolon | %68 | %92 |

Ön-işleme ve test-time augmentation (TTA) iyileştirmeleriyle birlikte nihai model **adil test setinde %97.02 doğruluk** elde etti (5919 eğitim / 304 doğrulama örneği).

---

## Ön-işleme

1. **Gradyan düzleştirme** — arka planın ağır bulanıklaştırılmış tahminine bölünerek gradyan giderilir (gri tonlamada kalınır, bilgi kaybı olmaz).
2. **Kuyruk kırpma** — sağdan sarkan çizginin ince kuyruğu, distance-transform ile bulunan "harf gövdesi" sınırının ötesinde kırpılır (aksi halde 5. karakter pozisyonu sistemik olarak çizgiyi harf sanıyordu).
3. Sabit boyuta ölçekleme (48×160).

Detaylar için [`onisleme.py`](onisleme.py).

## Mimari

Detaylar için [`captcha_model.py`](captcha_model.py) — 4 konvolüsyon bloğu (Conv+BN+ReLU+MaxPool) → 5 kolona indirgeme → paylaşımlı `Linear(384→128)` → `Linear(128→sınıf_sayısı)`.

## Kullanım

```bash
pip install -r requirements.txt
python predict.py ornekler/6w5gb.png
```

```
Tahmin: 6w5gb   (guven: 0.9993)
```

`ornekler/` klasöründe denemek için 10 örnek görsel var.

## Veri seti

Eğitim/doğrulama için kullanılan veri setinden 1000 örnek (ham + etiketli) Roboflow'da: **[gradient-captcha Dataset](https://universe.roboflow.com/aylins-workspace-i3lbm/gradient-captcha)**

## Not

Bu proje, CNN mimarisi ve captcha OCR problemi üzerinde çalışmak amacıyla geliştirilmiştir (eğitim/portföy amaçlı).
