# Faz 2 Raporu — Genişletme: Kadro, Girdi, Anomali Kataloğu

*Tamamlandı: 10 Haziran 2026 · Commit aralığı: `7cdf114…b2b064d` · 77 test*

## Amaç

Kadroyu "deterministik vs probabilistik" sorusunu *kontrollü* yanıtlayacak
biçimde büyütmek (üç ailede eşleştirilmiş ikizler), multivariate base kararını
veriyle vermek, anomali kataloğunu gerçekçi arıza tipleriyle genişletmek.

## Yapılanlar

- **Kadro 13 modele çıktı** (7 deterministik + 6 probabilistik). Eşleştirilmiş
  aileler: LSTM↔qLSTM (recurrent), DLinear↔qDLinear (linear),
  LightGBM-point↔LightGBM-quantile + QRF (tree); klasikler ve DeepAR/QT yerinde.
  Tümü `src/model_eval.py` ortak koşucusuyla **bit-bit aynı seed'li anomali
  grid'ine** girdi (bağlam eşitliği doğrulandı), türevlenebilenler white-box
  FGSM sütunu aldı, hepsi kalibrasyon süitinden geçti.
- **Multivariate base kararı** model-bazında: QT multi'ye terfi etti (tam
  bütçe), DeepAR uni kaldı; neden ayrıştıkları ayrımcı deneyle çözüldü
  (`probe_deepar_covariates.py`).
- **Permütasyon önemi** tablosu (val+test) — `feature_importance.json`.
- **Anomali kataloğu v2:** 5 yeni arıza injektörü (flatline, drift, noise
  burst, gap+imputation, clock skew) + doğrusallık/locality testleri;
  `run_anomaly_eval.py --catalog v2` bayrağı Faz 4 dalgasında açılacak.
- **Doğal uç dilimi** (enjeksiyonsuz): gerçek soğuk cephe / ani ısınma
  onluklarında kalibratörlerin false-alarm maliyeti ölçüldü.

## Ana sonuçlar

| Bulgu | Sayılar |
|---|---|
| **qLSTM ikizini anlamlı geçiyor** | medyan RMSE 2.384 vs LSTM 2.429; ΔRMSE −0.045, DM **p=0.024**; tek başına LSTM+QT ensemble'ına denk (p=0.99) |
| **QT-multi en güçlü tekil model** | CRPS 0.934, PICP 0.899, RMSE 2.21 (uni QT: 1.022/0.905/2.43) |
| GRU > LSTM (anlamlı) | ΔRMSE −0.044, p=0.010 — Bari 2025 ile uyumlu |
| Linear çiftte yön aynı, anlamsız | dlinear−qdlinear +0.032, p=0.081 → "başlık kazancı" mimariye bağlı |
| **Tree'ler ham haliyle en dayanıklı** | LS4× onarımsız PICP: QRF 0.391, LGBM 0.339 vs sinir ağları 0.24–0.37 (satürasyon hipotezi, 2 veri noktası) |
| DeepAR covariate gizemi çözüldü | calendar-only CRPS 1.145 < target-only 1.463 ≪ past-cov 1.904 → **suçlu donmuş-hava kanalları**, covariate yolu değil |
| Covariate'ler bozulmayı kısmen telafi ediyor | hedef-kanal LS4×'te QT-multi ham 0.368 vs uni 0.302 |
| Permütasyon önemi | VPmax ΔCRPS +1.18 (baskın), doy_cos +0.59, p/wv ≈ +0.10; hour kanalları stride-24'te pencere-arası özdeş → ölçülemez (dipnot) |
| False-alarm maliyeti düşük | girdi-koşullu τ gerçek cephelerde MPIW +%8–11, dilim PICP 0.85–0.95 |

## Figürler

- [Kadro genel bakış](../results/figures/phase2/roster_overview.png)
- [Eşleştirilmiş aileler](../results/figures/phase2/paired_families.png)
- [Ham dayanıklılık (LS4×)](../results/figures/phase2/raw_robustness_ls4.png)
- [Permütasyon önemi](../results/figures/phase2/permutation_importance.png)
- [Doğal uçlarda false-alarm](../results/figures/phase2/natural_extremes_falsealarm.png)

## Ne çalıştı / ne çalışmadı

**Çalıştı:** eşleştirilmiş tasarım (tek mimari değişkenle det-vs-prob cevabı);
ortak koşucu (yeni model eklemek ≈ 1 modül + 1 ince script); QT-multi terfisi;
ayrımcı sonda (tek deneyle iki hipotezi ayırdı).

**Çalışmadı / sınırlı:**
- **DLinear ailesi** bu görevde recurrent'ın gerisinde (RMSE 2.51) — Zeng'in
  uzun-horizon bulgusu 24h kısa horizonda tekrarlamıyor; yine de FGSM hasarının
  analitik sınırı (ε·‖w‖₁, unit-testli) rapora değerli.
- **QRF maliyeti:** leaf-tabanlı predict çok yavaş (~25 dk dump fazı) — bütçe
  stride-3+100 ağaçla sınırlandı; nokta doğruluğu DeepAR seviyesinde (p=0.48),
  kadroda "ikinci görüş" rolünde kalmalı.
- **LightGBM/QRF takvimsiz** (bağlam-yalnız tasarım gereği) → nokta doğrulukları
  LSTM/QT'nin 0.31 gerisinde; bilinçli bir kıyaslanabilirlik ödünleşimi.
- hour_sin/cos önem ölçümü stride-24 grid'de yapısal olarak kör (pencere-arası
  özdeş kanal) — yöntemsel dipnot.

## Faz 3'e devreden

HPO + çoklu seed (yakın band artık belli: LSTM≈GRU≈qLSTM≈QT), CV ablation,
robust/uç-odaklı eğitim, ensemble aralık birleştirme; normal-vs-optimize net
kıyas tabloları.
