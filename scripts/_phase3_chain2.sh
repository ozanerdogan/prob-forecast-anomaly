#!/usr/bin/env bash
# Phase-3 GPU chain #2: model-side optimization experiments (after chain #1).
set -u
cd /home/ozan/Desktop/prob-forecast-anomaly || exit 2
PY=.venv/bin/python
echo "PHASE3-CHAIN2 start $(date +%H:%M:%S)"
$PY scripts/run_robust_training.py    && echo "ROBUST-OK $(date +%H:%M:%S)"    || { echo "ROBUST-FAIL"; exit 1; }
$PY scripts/run_tail_oversampling.py  && echo "TAIL-OK $(date +%H:%M:%S)"      || { echo "TAIL-FAIL"; exit 1; }
$PY scripts/run_composite_anomaly.py  && echo "COMPOSITE-OK $(date +%H:%M:%S)" || { echo "COMPOSITE-FAIL"; exit 1; }
echo "PHASE3-CHAIN2 done $(date +%H:%M:%S)"
