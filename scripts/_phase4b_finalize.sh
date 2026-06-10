#!/usr/bin/env bash
# Phase-4b finalize: horizon ablation, extreme quantiles, robust+cal (QT+qLSTM),
# refreshed report tables/figures. Run AFTER _robust_chain.sh completes.
set -u
cd /home/ozan/Desktop/prob-forecast-anomaly || exit 2
PY=.venv/bin/python
echo "PHASE4B start $(date +%H:%M:%S)"
$PY scripts/run_horizon_ablation.py      && echo "HORIZON-OK $(date +%H:%M:%S)"   || { echo "HORIZON-FAIL"; exit 1; }
$PY scripts/run_qt_extreme_quantiles.py  && echo "EXTREME-OK $(date +%H:%M:%S)"   || { echo "EXTREME-FAIL"; exit 1; }
$PY scripts/run_robust_plus_cal.py       && echo "ROBUST-CAL-OK $(date +%H:%M:%S)" || { echo "ROBUST-CAL-FAIL"; exit 1; }
$PY scripts/make_report_tables.py        && echo "TABLES-OK"                       || echo "TABLES-FAIL"
echo "PHASE4B done $(date +%H:%M:%S)"
