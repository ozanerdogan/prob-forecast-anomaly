# Faz 1 Raporu — Adaptif Kalibrasyon (Çekirdek Katkı)

*Tamamlandı: 10 Haziran 2026 · Commit aralığı: `9ce45e9…1562b7e` · 61 test*

## Amaç

Projenin ana bulgusu, belirsizliğin tam ihtiyaç anında güvenilmez kalmasıydı:
level shift 4×'te DeepAR'ın %90 aralığı gerçeğin yalnız %27'sini kapsıyor ve
validation'da fit edilen **statik** spread-temperature (τ) bunu ancak %32'ye
çekebiliyordu. Faz 1'in işi bu soruna cevap vermekti: belirsizliği girdiye/zamana
göre ayarlayan **adaptif kalibrasyon** katmanları kurmak ve aynı dondurulmuş
tahminler üzerinde adil biçimde kıyaslamak.

## Yapılanlar

- **İki-aşamalı mimari:** Aşama-1 scriptleri dondurulmuş tahminleri
  `results/predictions/`e yazar (val+test × temiz + 12 anomali ayarı +
  hampel'li ikizler); Aşama-2 kalibrasyon scriptleri yalnız bu dosyaları okur,
  hiçbir model koşturmaz → before/after farkı **saf olarak onarım yöntemine**
  atfedilir. Eski `anomaly_eval.json` bit-bit aynı kaldı (regresyon kontrolü).
- **Dört onarım rejimi** (`scripts/calibrate_*.py`, model-agnostik kütüphane
  `src/calibrators.py`): statik τ (mevcut pratik), **CQR** (offline conformal
  margin), **ACI** (online; pencere *t* yalnız *t*-öncesi gerçekleşen coverage'ı
  kullanır — deploy'da da mevcut bilgi; γ + warm-start yalnız validation'da),
  **girdi-koşullu τ** (tamamen offline: validation + *enjekte edilmiş validation*
  çiftlerinde eğitilir; test gerçeğine hiç dokunmaz → "test verisi gördü"
  itirazı yapısal olarak kapalı).
- **Detect-and-clean kontrast baseline'ı:** hampel filtreli girdi onarımı
  (inference-only) — kalibrasyon-tarafı onarımın alternatifi olarak.
- **İstatistiksel anlamlılık:** Diebold–Mariano (HAC) + pencere-bazlı paired
  bootstrap, 359-pencere grid'inde (`run_significance.py`).
- Sigorta: LightGBM-quantile öne çekilip kalibrasyon makinesi üçüncü, yapısal
  olarak farklı profile karşı doğrulandı.

## Ana sonuçlar

| Level shift 4× (DeepAR) | PICP | MIS |
|---|---|---|
| Onarımsız | 0.268 | 85.1 |
| Statik τ | 0.323 | 76.3 |
| CQR | 0.320 | 75.2 |
| **ACI (online)** | **0.848** | **46.5** |
| **Girdi-koşullu τ (offline)** | **0.718** | **49.8** |

- Temiz veride ACI hedefe en yakın kalibre (0.898/0.903); girdi-koşullu hafif
  geniş (0.95 — keskinlik bedeli MIS'te raporlanıyor).
- **Anlamlılık:** LSTM vs QT nokta doğruluğu tam beraberlik (ΔRMSE −0.001,
  p=0.976); LSTM+QT ensemble kazancı **anlamlı** (−0.045, CI [−0.081, −0.010],
  p=0.006) — "combination beats selection" istatistiksel destekli.

## Figürler

- [PICP vs şiddet — manşet](../results/figures/phase1/calibration_picp_vs_intensity.png):
  statik/CQR ham eğriyle çöküyor; ACI 0.90 hattında, girdi-koşullu yüksekte.
- [MIS, level shift 4×](../results/figures/phase1/calibration_mis_ls4.png):
  adaptif onarım aralık skorunu yarılıyor.
- [Anlamlılık](../results/figures/phase1/significance_headline.png).

## Ne çalıştı / ne çalışmadı

**Çalıştı:** ACI (en dengeli: temizde hedefte, kaymada 0.85'e toparlıyor);
girdi-koşullu τ (anomaliye ACI'den hızlı tepki, FGSM'de en iyi); iki-aşamalı
mimari (üçüncü modelde sıfır rework ile doğrulandı).

**Çalışmadı (dürüst negatifler, rapora girecek):**
- **CQR** kaymada statikle aynı kaderde (0.32) — exchangeability kırılınca
  offline conformal garantisi anlamını yitiriyor; bu *beklenen* ve öğretici
  bir başarısızlık.
- **Detect-and-clean** level shift'i tasarımı gereği geçiriyor (yerel medyan
  şifti izler: 0.27→0.29) ve temiz/spike'ta zarar veriyor (DeepAR temiz PICP
  0.85→0.79) — girdi-tarafı onarımın false-alarm maliyeti ölçüldü; kalibrasyon-
  tarafı onarımın gerekliliğini kanıtlıyor.

**Kayıt düşülen sınırlar:** ACI'nin kapsama garantisi α-uzayındaki orijinal
kurala ait; sabit 7-kuantil ızgara nedeniyle τ-uzayında uyguladık (raporda
açıkça etiketlenecek). Kalıcı kayma altında her onarım coverage'ı genişlik
pahasına kurtarır — PICP yanında MIS/MPIW raporlanıyor.
