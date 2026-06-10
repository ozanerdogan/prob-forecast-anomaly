# Genel Özet — Probabilistic Forecasting with Anomaly Injection (Faz 1–4)

CENG 463 · Jena Climate · Deterministik vs probabilistik tahmin, anomali altında
*Son güncelleme: 11 Haziran 2026*

> Her madde **Ne yaptık · Ne için · Ne bekliyorduk · Sonuç** dörtlüsüyle.
> Tüm sayılar `results/` altındaki gerçek JSON'lardan; figürler
> `results/figures/phase{1,2,3,4}/`, faz raporları `report/phase{1,2,3,4}_report.md`.

---

## 0. Başlangıç durumu (bu çalışma öncesi)

Faz öncesi repoda zaten vardı: 6 model (naive, ARIMA, SARIMA, LSTM, DeepAR,
quantile-Transformer), 4 anomali tipi (spike, outlier, level shift, FGSM), statik
spread-temperature kalibrasyon, ablation ve hata analizi. **Ana bulgu:** statik
kalibrasyon anomalide çöküyor (level shift 4×'te DeepAR PICP 0.27 → τ ile ancak
0.32, hedef 0.90). Bu dört faz, o bulgunun *üstüne* çözüm inşa etti.

---

## Faz 1 — Adaptif kalibrasyon (çekirdek katkı)

**Ne yaptık.** İki-aşamalı mimari kurduk (Aşama-1 modeller tahminleri diske
dondurur, Aşama-2 kalibrasyon yalnız o dondurulmuş tahminleri okur) ve dört
onarım rejimi yazdık: statik τ, CQR, ACI (online), girdi-koşullu τ (offline);
artı hampel detect-and-clean kontrast baseline'ı ve DM/bootstrap anlamlılık.

**Ne için.** "Belirsizlik tam ihtiyaç anında güvenilmez" sorununa doğrudan
cevap. İki-aşamalı yapı, normal-vs-onarımlı farkının **saf olarak yönteme**
atfedilebilmesi için (forecast'ler bit-bit aynı).

**Ne bekliyorduk.** Adaptif yöntemlerin statik τ'nun çöktüğü yerde coverage'ı
kurtarması; "test verisi gördü" itirazına karşı offline bir varyant.

**Sonuç ✓.** Level shift 4× (DeepAR): PICP 0.27 → **ACI 0.85**, girdi-koşullu
0.72; MIS 85 → 47 (yarı). Girdi-koşullu τ yalnız validation + enjekte anomaliyle
eğitildiği için itiraz kökten kapalı. Dürüst negatifler: **CQR** kaymada
statikle çöküyor (exchangeability kırılması), **detect-and-clean** level shift'i
geçiriyor ve temizde zarar veriyor — kalibrasyon-tarafı onarımın gerekçesi.
Anlamlılık: LSTM≈QT beraberlik (p=0.98), ensemble kazancı gerçek (p=0.006).

---

## Faz 2 — Kadro genişletme + girdi + arıza kataloğu

**Ne yaptık.** Kadroyu 13 modele çıkardık (üç ailede *eşleştirilmiş* det/prob
ikizler: LSTM↔qLSTM, DLinear↔qDLinear, LightGBM↔QRF; + GRU, DeepAR, QT,
QT-multi). Multivariate base'i model-bazında karara bağladık, permütasyon önemi
çıkardık, anomali kataloğunu 5 yeni gerçekçi arıza tipiyle (flatline, drift,
noise burst, gap+imputation, clock skew) genişlettik, doğal-uç dilimi ekledik.

**Ne için.** "Deterministik vs probabilistik" sorusunu *kontrollü* yanıtlamak
(aynı gövde, sadece çıktı başlığı değişir → başlığın etkisi mimariden izole);
covariate'lerin gerçekten yardım edip etmediğini ve hangi değişkenin önemli
olduğunu ölçmek.

**Ne bekliyorduk.** Probabilistik başlığın nokta doğruluğuna mal olmamasını;
multivariate'ın yardım etmesini; tree'lerin anomaliye daha dayanıklı olmasını.

**Sonuç ✓.** **qLSTM medyanı kendi deterministik ikizini anlamlı geçiyor**
(p=0.024) — probabilistik başlık bedava değil, kazandırıyor. **QT-multi tüm
metriklerde en güçlü tekil model** (RMSE 2.21, CRPS 0.934). **DeepAR covariate
gizemi çözüldü:** suçlu past-covariate'in donmuş-hava kanalları (calendar-only
sonda kanıtladı). VPmax baskın önemli değişken. **Tree hipotezi doğrulandı:**
QRF/LightGBM ham haliyle kadronun en dayanıklıları. Negatif: DLinear bu kısa
horizon'da recurrent'ın gerisinde.

---

## Faz 3 — Optimizasyon & robustluk (normal vs optimize)

**Ne yaptık.** Her deneyi *normal vs optimize* yan yana koştuk: HPO (val'de
seçim), çoklu seed (4 model × 3), forward-chaining yıl-CV, robust (anomali-
augmente) eğitim, tail oversampling, adaptive-CQR, aralık ensemble'ı, bileşik
anomali.

**Ne için.** "Optimize edersek ne kazanırız" sorusunu dürüstçe ölçmek;
mevcut iddiaların sağlamlığını (seed gürültüsü, yıl etkisi) sınamak.

**Ne bekliyorduk.** HPO'nun küçük kazanç vermesini; robust eğitimin anomalide
yardım etmesini; ensemble'ın iyileştirmesini.

**Sonuç (karışık, hepsi öğretici).**
- **Robust eğitim — en güçlü:** LS4× ham RMSE 9.0 → **5.4**, PICP 0.24 → 0.69.
  Kalibrasyonun ulaşamadığı *nokta-tahmin* onarımı. Bedel: temizde +0.22 RMSE.
- **Çoklu seed gizli bulgu açtı:** LSTM 2.446±**0.054** (oynak), GRU
  2.395±0.007 → Faz-2'deki tek-seed LSTM şanslıymış; qLSTM>LSTM güçlendi.
- **CV bağlam verdi:** 2016 (ana test yılı) **en kolay**, 2015 en zor →
  manşetlerimiz hafif iyimser (dürüstlük notu).
- **Dürüst negatifler:** HPO neredeyse hiçbir şey kazandırmadı (QT'de kazanan =
  varsayılan — "adil sabit config" duruşu doğrulandı); **aralık ensemble'ı en
  güçlü tekil üyeyi geçemedi** (Faz-1 nokta-tahmin sonucunun tersi); tail
  oversampling sıcak uçta iyileştirdi ama genel kalibrasyonu bozdu (roadmap
  uyarısı birebir); bileşik anomaliler birikmiyor (baskın arıza belirler).

---

## Faz 4 — Konsolidasyon + birleşik sentez

**Ne yaptık.** v2 arıza kataloğunun (8 tip) manşet kadroyla tam süpürmesi;
model-tarafı (robust) × aralık-tarafı (ACI) **birleşik** 4-köşe değerlendirmesi;
rapor tablolarının dondurulmuş dump'lardan üretimi.

**Ne için.** Genişletilmiş arızaların önceki hikayeyi bozup bozmadığını görmek;
Faz-3'ün iki ayrı onarımının tamamlayıcı mı redundant mı olduğunu sınamak.

**Ne bekliyorduk.** Yeni arızaların üç-rejime oturmasını; robust+ACI'nin ayrı
ayrıdan iyi olmasını.

**Sonuç ✓✓.** Yeni 5 arıza temiz üç-rejime oturdu; **drift, level_shift ile aynı
yıkıcı sınıfta** ("yavaş kayma = hızlı kayma" doğrulandı). **Birleşik sonuç
ana sentez:** LS4×'te robust-tek 0.69, ACI-tek 0.75, **robust+ACI birlikte
0.87** — tamamlayıcılar. Projenin pratik reçetesi: *modeli anomaliyle eğit +
üstüne online adaptif kalibrasyon.*

## Faz 4+ — derinleştirme (girdi analizi + genelleme + ek ablation)

**Ne yaptık.** Recenzör/hoca sorularını kapatan ek deneyler.

**Sonuçlar ✓.**
- **Robust eğitim GENEL reçete:** üç mimaride de işe yarıyor (LS4× ham PICP
  qLSTM 0.24→0.69, QT 0.30→0.72, DeepAR 0.27→0.46); robust+ACI iki mimaride
  birleşik en iyi (qLSTM 0.87, QT 0.89).
- **Girdinin değeri + LEAKAGE düzeltmesi:** T-türevleri (VPmax, Tpot...)
  sıcaklığı analitik veriyor (VPmax→T RMSE 0.05°C) → "exogenous-only iyi"
  bir sızıntı artefaktıymış; gerçek bağımsız sensörlerle (basınç/nem/rüzgâr)
  RMSE 3.68, naive'in (3.21) altında → **direkt sıcaklık vazgeçilmez**.
  Bağımsız önem: T ≫ mevsim ≫ rüzgâr > basınç > nem.
- **Horizon ablation:** RMSE 12h 1.86 / 24h 2.43 / 48h 3.15 — 24h seçimi
  gerekçeli. **Ekstrem quantile:** %90/95/98 aralık hepsi kalibre (PICP
  0.90/0.95/0.98), crossing yok.
- **Dürüst negatifler (rapora):** **10 dk çözünürlük kazandırmadı**
  (RMSE 2.40 vs 2.38, seed gürültüsü içinde); HPO bulgusuyla tutarlı —
  problem HP/veri-miktarına yapısal duyarsız; DeepAR oracle leakage
  rosterdan çıkarıldı.

---

## Genel sentez — projenin söylediği

1. **Sorun gerçek ve yaygın.** Statik kalibrasyon, dağılımsal anomali (level
   shift, drift, FGSM) altında çöküyor — 8 arızanın 3'ü "yıkıcı" sınıfta.
2. **Çözüm var ve iki katmanlı.** Aralık-tarafı (ACI/girdi-koşullu kalibrasyon)
   coverage'ı kurtarıyor; model-tarafı (robust eğitim) nokta tahminini kurtarıyor;
   **ikisi tamamlayıcı** ve birlikte en iyisi.
3. **Probabilistik olmak kazandırıyor** — eşleştirilmiş ikizlerde pinball başlığı
   nokta doğruluğuna mal olmadan belirsizlik veriyor (qLSTM ikizini anlamlı geçti).
4. **Dürüstlük diskini döndürdük:** HPO kazandırmadı, ensemble aralıkta
   geçemedi, CQR kaymada çöktü, 2016 kolay yıl, tail oversampling kalibrasyonu
   bozdu — hepsi raporlandı.

## Metodolojik sağlamlık
- Tek global seed (42), kronolojik sızıntısız split, scaler yalnız train'e fit.
- İki-aşamalı dondurulmuş-tahmin mimarisi → onarım deltaları saf atfedilebilir.
- 79 unit test (metrikler, injektörler + doğrusallık, kalibratörler, ACI'nin
  kendi-sonucunu-görmemesi, anlamlılık, yeni modeller).
- Offline/online etiketi açık; ACI'nin τ-uzayı kullanımı not edildi.

## Gelecek iş (yapılmadı, gerekçeli)
- Robust+ACI'yi tüm kadroya yaymak (3 mimaride doğrulandı; kalan modeller marjinal); QRF/linear için v2 tam süpürme.
- N-BEATS/TFT (DLinear sonucu getiriyi düşük gösterdi), DeepAR 2h varyantı,
  10dk çözünürlük (görev tanımını değiştirir) — en spekülatif, atlandı.
- Literatür taraması künyelerinin DOI doğrulaması (adaylar
  `implementation.md` "Rapora yazılacaklar"da).

## Yol haritası
Detaylı iş listesi ve kararlar: `implementation.md` (gitignore'lu, yerel).
Faz raporları: `report/phase{1,2,3,4}_report.md`.
