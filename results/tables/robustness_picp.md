# Raw robustness — PICP at severe settings (uncalibrated)

| Model | clean | level_shift_4.0 | fgsm_4.0 | flatline_4.0 |
|---|---|---|---|---|
| deepar | 0.852 | 0.268 | 0.421 | 0.589 |
| qtransformer | 0.905 | 0.302 | 0.310 | 0.589 |
| qtransformer_multi | 0.841 | 0.203 | 0.164 | 0.552 |
| qlstm | 0.927 | 0.237 | 0.250 | 0.643 |
| qdlinear | 0.935 | 0.271 | 0.272 | - |
| lgbm | 0.866 | 0.339 | - | 0.510 |
| qrf | 0.916 | 0.391 | - | - |
| qlstm_robust | 0.938 | 0.688 | 0.496 | 0.729 |
