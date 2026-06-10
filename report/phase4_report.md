# Faz 4 Raporu — Konsolidasyon

*Tamamlandı: 11 Haziran 2026 · 79 test · tüm sonuçlar dondurulmuş dump'lardan*

## Amaç

Donmuş kadro üstünde işi kapatmak: (1) genişletilmiş arıza kataloğunun (v2, 8
arıza tipi) tam süpürmesi, (2) Faz-3'ün açtığı **model-tarafı × aralık-tarafı**
birleşik değerlendirmesi, (3) rapor tablolarının dondurulmuş tahminlerden
üretilmesi (yeniden eğitim YOK).

## 1. v2 arıza kataloğu — tam süpürme

Manşet modeller (DeepAR, QT, QT-multi, qLSTM, LightGBM + LSTM/naive)
`--catalog v2` ile 8 arıza × 3 şiddette değerlendirildi. QRF/GRU/DLinear/qDLinear
v2'ye **alınmadı** (QRF predict'i ~25 dk × 5 yeni arıza = saatler; v1'de zaten
temsil ediliyorlar — kayıt düşüldü).

**Sonuç: yeni 5 arıza, Faz-2'nin üç-rejim hikayesine temiz oturdu**
(QT-multi ham PICP, 4× şiddet):

| Rejim | Arızalar | PICP (4×) | Yorum |
|---|---|---|---|
| **Yıkıcı** | drift **0.31**, level_shift 0.37, fgsm 0.19 | <0.40 | kalıcı/adversarial seviye kayması |
| **Orta** | flatline 0.67, clock_skew 0.66 | ~0.66 | bağlam yapısını bozar, seviyeyi değil |
| **Hafif** | noise_burst 0.88, gap_imputation 0.89, point_spike/outlier 0.90 | >0.87 | 168h bağlamda lokalize, kayboluyor |

**En değerli doğrulama:** *drift* (yavaş doğrusal kayma) ham PICP'si 0.31 ile
*level_shift*'in (0.37) yanında — "drift = level shift'in yavaş kardeşi"
hipotezi (Faz-2'de yazılmıştı) sayıyla doğrulandı: kaymanın hızlı mı yavaş mı
olduğu önemli değil, **kalıcı seviye hatası** belirleyici. Gap+imputation'ın
hafif kalması da önemli — pipeline'ın kendi interpolasyonu bağlamı düzleştirse
bile 168h pencerede fark yutuluyor.
→ [fault_catalog_heatmap.png](../results/figures/phase4/fault_catalog_heatmap.png)

## 2. Birleşik değerlendirme: model-tarafı × aralık-tarafı — **ANA SENTEZ**

Faz-3 iki ayrı onarım gösterdi: robust eğitim *nokta*yı, ACI *aralığı* onarıyor.
Soru: tamamlayıcı mı? qLSTM'in dört köşesi (ham/ACI × normal/robust), hepsi
dondurulmuş dump'lardan:

| Ayar | normal ham | normal+ACI | robust ham | **robust+ACI** |
|---|---|---|---|---|
| temiz | 0.93 | 0.91 | 0.94 | 0.91 |
| level shift 2× | 0.51 | 0.83 | 0.81 | **0.88** |
| level shift 4× | 0.24 | 0.75 | 0.69 | **0.87** |
| fgsm 4× | 0.25 | 0.77 | 0.50 | **0.84** |

**Sonuç: tamamlayıcılar, redundant değil.** Level shift 4×'te tek başına robust
0.69, tek başına ACI 0.75, **birlikte 0.87** — her ikisi de tek başına ulaşamadığı
yere birlikte ulaşıyor. Dahası robust+ACI nokta-tahmin RMSE'si de korunuyor
(LS4× medyan RMSE 5.41 vs normal-ham 9.0). Projenin pratik reçetesi: **modeli
anomaliyle eğit + üstüne online adaptif kalibrasyon** — belirsizlik kirli girdide
hem dürüst (PICP≈0.87) hem keskin kalıyor.
→ [robust_plus_cal.png](../results/figures/phase4/robust_plus_cal.png)

## 3. Rapor tabloları (dondurulmuş dump'lardan, eğitimsiz)

`make_report_tables.py` üç tabloyu dump'lardan üretir → `report/tables/`:
- **Temiz sıralama** ([clean_leaderboard.md](tables/clean_leaderboard.md)):
  QT-multi RMSE 2.21 (en iyi) → qLSTM 2.38 → GRU 2.39 → … → naive 3.21.
- **Ham dayanıklılık** ([robustness_picp.md](tables/robustness_picp.md)):
  qlstm_robust her ağır ayarda kadronun en dayanıklısı (LS4× 0.69 vs ≤0.39).
- **Kalibrasyon kurtarması** ([calibration_recovery.md](tables/calibration_recovery.md)):
  her model × 5 yöntem, LS4× PICP.
→ [final_leaderboard.png](../results/figures/phase4/final_leaderboard.png)

## Atlananlar (gerekçeli)
- **QRF/GRU/DLinear/qDLinear v2 süpürmesi** — maliyet; v1 temsil yeterli.
- **10dk çözünürlük** — görev tanımını değiştiriyor (naive s=24→144, ARIMA refit
  ~6×), lookback platosu kazancı şüpheli → en spekülatif madde, atlandı.
- **24h per-horizon / sezon-sıcaklık yeni modellerle** — mevcut `error_analysis`
  iki modelle var; tam kadro genişletmesi dump altyapısıyla artık mümkün ama
  marjinal getiri (ileride istenirse `make_report_tables` genişletilir).

## Durum
Dört faz tamamlandı. Genel sentez: [SUMMARY.md](SUMMARY.md).
