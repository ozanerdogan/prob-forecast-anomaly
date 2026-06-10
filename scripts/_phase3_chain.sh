#!/usr/bin/env bash
# Phase-3 GPU chain orchestrator (newline-safe; one step per line with &&).
set -u
cd /home/ozan/Desktop/prob-forecast-anomaly || exit 2
PY=.venv/bin/python
echo "PHASE3-GPU-CHAIN start $(date +%H:%M:%S)"
$PY scripts/run_hpo.py        && echo "HPO-OK $(date +%H:%M:%S)"       || { echo "HPO-FAIL"; exit 1; }
$PY scripts/run_multiseed.py  && echo "MULTISEED-OK $(date +%H:%M:%S)" || { echo "MULTISEED-FAIL"; exit 1; }
$PY scripts/run_cv.py         && echo "CV-OK $(date +%H:%M:%S)"        || { echo "CV-FAIL"; exit 1; }
echo "PHASE3-GPU-CHAIN done $(date +%H:%M:%S)"
