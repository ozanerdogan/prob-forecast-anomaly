# Detect-then-adapt — policy comparison (qlstm)

Coverage repair per regime; clean row shows the width cost (MIS, lower = sharper).

| Setting | raw | static | aci | input_tau | detect_adapt |
|---|---|---|---|---|---|
| clean MIS | 9.57 | 9.69 | 9.52 | 12.42 | 10.10 |
| level_shift_4.0 PICP | 0.237 | 0.256 | 0.746 | 0.714 | 0.651 |
| drift_4.0 PICP | 0.223 | 0.242 | 0.752 | 0.729 | 0.650 |
| fgsm_4.0 PICP | 0.250 | 0.266 | 0.772 | 0.837 | 0.881 |
| flatline_4.0 PICP | 0.643 | 0.665 | 0.847 | 0.889 | 0.888 |
| clock_skew_4.0 PICP | 0.569 | 0.589 | 0.828 | 0.810 | 0.883 |
| noise_burst_4.0 PICP | 0.888 | 0.901 | 0.902 | 0.931 | 0.918 |

# Detection quality by fault kind and intensity (qlstm, AUC vs clean-test)

| Fault | 1.0 | 2.0 | 4.0 |
|---|---|---|---|
| clock_skew | 0.59 | 0.82 | 1.00 |
| contextual_outlier | 0.52 | 0.56 | 0.72 |
| drift | 0.56 | 0.64 | 0.82 |
| fgsm | 0.92 | 0.99 | 1.00 |
| flatline | 1.00 | 1.00 | 1.00 |
| gap_imputation | 0.51 | 0.54 | 0.60 |
| level_shift | 0.55 | 0.66 | 0.83 |
| noise_burst | 0.75 | 0.89 | 0.98 |
| point_spike | 0.51 | 0.56 | 0.69 |

Clean false-alarm rate (any repair engagement): 0.142
