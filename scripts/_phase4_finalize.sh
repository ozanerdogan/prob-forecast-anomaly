#!/usr/bin/env bash
# Phase-4 finalize: calibrate v2-swept + robust models, build tables/figures.
# Run AFTER _phase4_chain.sh completes.
set -u
cd /home/ozan/Desktop/prob-forecast-anomaly || exit 2
PY=.venv/bin/python
echo "PHASE4-FINALIZE start $(date +%H:%M:%S)"
# refit every calibrator over the full roster (now incl. qlstm_robust + v2 settings)
for meth in static cqr aci input_tau aci_margin; do
  $PY scripts/calibrate_${meth}.py >/dev/null && echo "CAL-${meth}-OK" || echo "CAL-${meth}-FAIL"
done
$PY scripts/calibrate_detect_clean.py >/dev/null && echo "CAL-detect-OK" || echo "CAL-detect-FAIL"
$PY scripts/run_robust_plus_cal.py     && echo "ROBUST-CAL-OK"   || { echo "ROBUST-CAL-FAIL"; exit 1; }
$PY scripts/run_significance.py >/dev/null && echo "SIG-OK"       || echo "SIG-FAIL"
$PY scripts/make_report_tables.py      && echo "TABLES-OK"       || { echo "TABLES-FAIL"; exit 1; }
$PY scripts/make_phase_figures.py --phase 4 && echo "FIG4-OK"    || { echo "FIG4-FAIL"; exit 1; }
echo "PHASE4-FINALIZE done $(date +%H:%M:%S)"
