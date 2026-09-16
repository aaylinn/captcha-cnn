# captcha-crnn

Gradyan (renk geçişi) arka planlı, **sabit 5 karakterli** CAPTCHA görsellerini
okuyan bir CNN modeli.

## Sonuç

**%97.02 doğruluk** (adil testte, test-time augmentation ile).

## Mimari — "Paylaşımlı Kolon Sınıflandırıcı"

Klasik yaklaşım (her karakter pozisyonu için ayrı bir sınıflandırıcı başı)
denendi ve öğrenemedi: 5 ayrı baş kullanıldığında, örneğin 1. pozisyondaki
"a" ile 4. pozisyondaki "a" birbirinden bağımsız öğrenilmek zorunda kalıyor
— her sınıf için etkili veri miktarı 5 kat azalıyor (40 epoch sonunda eğitim
karakter doğruluğu %7'de kaldı, rastgele seviye %4.3'tü).

**Çözüm:** karakter sınıflandırıcısını pozisyonlar arasında paylaştırmak.
Konvolüsyon yığınından sonra öznitelik haritası tam **5 koluna** indirilir
(her kolon bir karaktere denk gelir) ve **aynı** `Linear` katmanları tüm
5 kolona paylaşımlı uygulanır (PyTorch'ta `nn.Linear`, girdinin son
boyutuna otomatik ve paylaşımlı uygulanır — Keras'taki
`TimeDistributed(Dense)` ile aynı davranış).

Ölçülen fark (aynı veri, augmentasyon yok, 30 epoch):

| Mimari | Doğrulama kelime doğruluğu | Karakter doğruluğu |
|---|---|---|
| 5 ayrı sınıflandırıcı başı | %0 | %8 |
| **Paylaşımlı kolon** | **%68** | **%92** |

## Ön-işleme — Gradient Düzleştirme + Kuyruk Kırpma

1. **Gradient düzleştirme**: arka plandaki renk geçişi, görüntünün ağır
   bulanıklaştırılmış bir tahminine bölünerek giderilir (binarizasyon yok,
   gri tonlamada kalınır — bilgi kaybı olmaz).
2. **Kuyruk kırpma**: çizgi her zaman sağdan taşıyor; metin kısa/dar
   olduğunda çizginin ince kuyruğu son karakterden çok daha sağda bitiyor,
   bu da 5. kolonun gerçek harfi değil boş çizgi kuyruğunu görmesine (ve
   sistemik hataya) yol açıyordu. Distance-transform ile her mürekkep
   pikselinin yerel kalınlığı hesaplanır — çizgi ince, harfler kalın olduğu
   için "kalın" sayılan piksellerin en sağdaki sütunu gerçek metnin bittiği
   yer olarak alınır ve görüntü oradan kırpılır.

Detaylar için [`onisleme.py`](onisleme.py) içindeki modül docstring'ine bakın.

## Veri Seti

Eğitim/doğrulama görselleri Roboflow'da barındırılıyor:
[Roboflow — captcha veri seti](https://app.roboflow.com/aylins-workspace-i3lbm/gradient-captcha/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)

## Kullanım

```bash
pip install -r requirements.txt
python tahmin_et.py yol/to/captcha.png
```

Çıktı: `Tahmin: x7f2q  (guven: %98.3)`

Kod içinden çağırmak istersen:

```python
from tahmin_et import tahmin_et

metin, guven = tahmin_et("captcha.png")
print(metin, guven)
```

`guven`, tahmindeki **en zayıf karakterin** olasılığı (0-1 arası) — düşükse
o tahminin elle kontrol edilmesi önerilir.

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `model/model_en_iyi.pt` | Eğitilmiş ağırlıklar |
| `model/alfabe.json` | Alfabe (23 karakter) + ön-işleme ayarları |
| `captcha_model.py` | Model mimarisi (`PaylasimliKolonCNN`) + kaydet/yükle yardımcıları |
| `onisleme.py` | Ortak ön-işleme (gradient düzleştirme + kuyruk kırpma + boyutlandırma) |
| `tahmin_et.py` | Bağımsız, hazır çalışan tahmin scripti (test-time augmentation dahil) |
