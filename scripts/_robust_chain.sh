#!/usr/bin/env bash
# Robust-generalisation chain: 3-architecture normal-vs-robust + QT robust dump.
set -u
cd /home/ozan/Desktop/prob-forecast-anomaly || exit 2
PY=.venv/bin/python
echo "ROBUST-CHAIN start $(date +%H:%M:%S)"
$PY scripts/run_robust_generalize.py && echo "GENERALIZE-OK $(date +%H:%M:%S)" || { echo "GENERALIZE-FAIL"; exit 1; }
$PY scripts/run_qt_robust.py          && echo "QT-ROBUST-DUMP-OK $(date +%H:%M:%S)" || { echo "QT-ROBUST-FAIL"; exit 1; }
$PY scripts/calibrate_aci.py --models qtransformer_robust >/dev/null       && echo "CAL-ACI-OK"       || echo "CAL-ACI-FAIL"
$PY scripts/calibrate_input_tau.py --models qtransformer_robust >/dev/null && echo "CAL-INPUT-OK"     || echo "CAL-INPUT-FAIL"
echo "ROBUST-CHAIN done $(date +%H:%M:%S)"
