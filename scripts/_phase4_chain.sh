#!/usr/bin/env bash
# Phase-4 GPU chain: v2 fault sweep for the headline models + robust dump.
# QRF/gru/dlinear/qdlinear are NOT re-swept on v2 (cost; v1 represents them).
set -u
cd /home/ozan/Desktop/prob-forecast-anomaly || exit 2
PY=.venv/bin/python
echo "PHASE4-CHAIN start $(date +%H:%M:%S)"
$PY scripts/run_anomaly_eval.py --catalog v2 && echo "ANOMALY-V2-OK $(date +%H:%M:%S)" || { echo "ANOMALY-V2-FAIL"; exit 1; }
$PY scripts/run_qlstm.py --catalog v2        && echo "QLSTM-V2-OK $(date +%H:%M:%S)"   || { echo "QLSTM-V2-FAIL"; exit 1; }
$PY scripts/run_qtransformer_multi.py --catalog v2 && echo "QTM-V2-OK $(date +%H:%M:%S)" || { echo "QTM-V2-FAIL"; exit 1; }
$PY scripts/run_lgbm.py --catalog v2         && echo "LGBM-V2-OK $(date +%H:%M:%S)"    || { echo "LGBM-V2-FAIL"; exit 1; }
$PY scripts/run_qlstm_robust.py              && echo "ROBUST-DUMP-OK $(date +%H:%M:%S)" || { echo "ROBUST-DUMP-FAIL"; exit 1; }
echo "PHASE4-CHAIN done $(date +%H:%M:%S)"
