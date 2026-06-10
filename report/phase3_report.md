# Faz 3 Raporu — Optimizasyon & Robustluk

*Tamamlandı: 10 Haziran 2026 · 77+ test · normal vs optimize her deneyde yan yana*

## Amaç

Faz 1–2 modellerini sağlamlaştırmak ve "optimize edersek ne kazanırız" sorusunu
**normal vs optimize** kıyasıyla net ölçmek: hiperparametre optimizasyonu, seed
duyarlılığı, yıl-bazlı CV, model-tarafı robustluk (anomali-augmente eğitim, uç
oversampling), aralık ensemble'ı, ve adaptive-CQR.

## Ne yaptık, ne çalıştı, ne çalışmadı

### 1. HPO (validation'da seçim) — **çalışmadı (öğretici negatif)**
20 eğitimlik küçük gridler, seçim yalnız validation'da, test'e yalnız kazanan
dokundu. Sonuç: kazançlar gürültü seviyesinde. **QT'de kazanan config =
zaten varsayılan** (d_model 64, 2 katman); LSTM/qLSTM'de 128-gizli/2-katman
kazandı ama validation farkı ~0.003. Bu, projenin baştan beri savunduğu "makul
sabit değerler adil kıyas için yeterli" duruşunu **doğruluyor** — rapora
güçlü bir "tuning bu problemde manşeti değiştirmiyor" mesajı.
→ [hpo_default_vs_best.png](../results/figures/phase3/hpo_default_vs_best.png)

### 2. Çoklu seed (4 model × 3 seed) — **çalıştı, gizli bir bulgu açtı**
| Model | RMSE (ort ± std) |
|---|---|
| GRU | **2.395 ± 0.007** |
| qLSTM | 2.402 ± 0.013 |
| LSTM | 2.446 ± **0.054** |
| qDLinear | 2.479 ± 0.004 |

LSTM'in std'si diğerlerinin ~5–8 katı (seed-7'de 2.52'ye fırlıyor) → Faz-1'deki
tek-seed LSTM skoru (2.429) **şanslı bir çekilişmiş**. GRU hem en iyi hem en
kararlı. Bu, Faz-2'deki "qLSTM ikizini geçiyor" sonucunu **güçlendiriyor**:
ortalama LSTM 2.446 vs qLSTM 2.402, artık varyans bilinerek.
→ [multiseed_rmse.png](../results/figures/phase3/multiseed_rmse.png)

### 3. Forward-chaining CV (yıl-bazlı) — **çalıştı, ana test yılını bağlamladı**
Test yılı 2013–2016, genişleyen train, önceki yıl validation:
| | 2013 | 2014 | 2015 | 2016 | std |
|---|---|---|---|---|---|
| qLSTM | 2.58 | 2.60 | 2.85 | **2.43** | 0.15 |
| LSTM | 2.58 | 2.61 | 2.93 | 2.45 | 0.18 |

**2016 (projenin ana test yılı) en KOLAY yıl, 2015 en zor.** "2016 ekstrem
miydi?" sorusunun cevabı: tersine — manşet sonuçlarımız hafifçe iyimser
tarafta; rapora dürüstlük notu. Fold std ~0.15 makul.
→ [cv_fold_variance.png](../results/figures/phase3/cv_fold_variance.png)
NOT: genişleyen train, yıl etkisini train-boyutuyla karıştırır (kayıt düşüldü).

### 4. Robust (anomali-augmente) eğitim — **EN GÜÇLÜ Faz-3 sonucu**
qLSTM'in eğitim batch'lerine uçuş-anında rastgele anomali enjekte edildi
(p=0.5). Ham (kalibrasyonsuz) sonuç:
| Ayar | PICP normal→robust | RMSE normal→robust |
|---|---|---|
| temiz | 0.927→0.938 | 2.38→**2.60** (bedel) |
| level shift 2× | 0.511→0.810 | 4.99→3.96 |
| level shift 4× | 0.237→**0.688** | 9.03→**5.41** |

Kritik fark: kalibrasyon yalnız *aralığı* onarıyordu; robust eğitim **nokta
tahminini de** kurtarıyor (RMSE 9→5.4). İkisi tamamlayıcı — robust eğitim +
ACI birlikte muhtemelen en güçlü kombinasyon (Faz-4 birleşik değerlendirme).
Bedeli: temizde RMSE +0.22 (klasik robustluk ödünleşimi).
→ [robust_training_picp.png](../results/figures/phase3/robust_training_picp.png)

### 5. Tail oversampling — **kısmen çalıştı, roadmap uyarısını doğruladı**
Uç-sıcaklık pencereleri ağırlıklandırıldı (power=3):
| Dilim | RMSE | bias | PICP |
|---|---|---|---|
| sıcak uç | 2.61→**2.25** | −1.15→**−0.12** | 0.917→0.918 |
| tüm test | 2.38→**2.96** | +0.10→+0.50 | 0.927→**0.868** |

Sıcak uçta gerçek kazanç (bias neredeyse sıfırlandı) ama **genel kalibrasyon ve
bias bozuldu** — roadmap'teki "genel kalibrasyonu bozabilir → kalibrasyon
yeniden-kontrolü" uyarısı birebir gerçekleşti. Net ödünleşim: uç-odaklı
uygulamalar için değerli, genel forecaster için zararlı.

### 6. Aralık ensemble'ı — **çalışmadı (Faz-1 sonucunun tersi)**
Probabilistik üyelerin kuantil eğrileri ortalandı (eşit + val-CRPS-ağırlıklı).
Temiz pinball: ensemble 0.493 vs **en iyi tekil üye QT-multi 0.467**. Faz-1'de
nokta-tahminde "combination beats selection" geçerliydi; aralıkta, çok güçlü bir
multivariate üye varken ensemble onu **geçemiyor**. Dürüst negatif.
→ [ensemble_intervals.png](../results/figures/phase3/ensemble_intervals.png)

### 7. Adaptive-CQR (online additive margin) — **çalıştı**
CQR'ın online versiyonu (offline margin ile warm-start, sonra coverage'a göre
adapte). DeepAR level shift 4×: PICP 0.27→0.86, MIS 85→31 — ACI-τ ile başa baş,
bazı modellerde daha iyi MIS. Roadmap'teki CQR→adaptive-CQR köprüsü kuruldu.

### 8. Bileşik anomali — **çalıştı, "birikmiyor" bulgusu**
flatline + spike üst üste: RMSE 3.78 ≈ flatline-tek 3.77. Baskın arıza
(flatline son bloğu donduruyor) sonucu belirliyor; spike onun içine düşünce ek
hasar yok. Anomaliler her zaman birikmez — rapora nüans.

## Atlananlar (gerekçeli)
- **N-BEATS / TFT:** opsiyoneldi; DLinear'ın bu kısa-horizon görevde recurrent'ın
  gerisinde kalması (Faz-2) "daha fazla derin mimari" getiriminin düşük olduğunu
  gösterdi. TFT'nin future-input leakage riski + maliyeti, marjinal kazanç
  beklentisini karşılamıyor. Faz-4 zaman kalırsa.
- **DeepAR 2-saat çözünürlük varyantı:** görev tanımını değiştiriyor; düşük
  öncelikliydi, atlandı. (DeepAR+takvim covariate varyantı ise Faz-2 sondasının
  doğal devamı — Faz-4 aday.)

## Faz-4'e devreden
Robust-eğitim + adaptif-kalibrasyon **birleşik** değerlendirmesi (ikisi
tamamlayıcı çıktı); `--catalog v2` ile tam arıza süpürmesi; manşet tabloların
donmuş kadroyla tek dalgada yenilenmesi.
