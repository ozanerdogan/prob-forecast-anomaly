#!/usr/bin/env bash
set -u
cd /home/ozan/Desktop/prob-forecast-anomaly || exit 2
PY=.venv/bin/python
echo "QTMULTI-REFRESH start $(date +%H:%M:%S)"
$PY scripts/run_qtransformer_multi.py --catalog v2 && echo "QTMULTI-OK $(date +%H:%M:%S)" || { echo "QTMULTI-FAIL"; exit 1; }
for m in static cqr aci input_tau aci_margin; do
  $PY scripts/calibrate_${m}.py --models qtransformer_multi >/dev/null && echo "CAL-${m}-OK" || echo "CAL-${m}-FAIL"
done
$PY scripts/calibrate_detect_clean.py >/dev/null && echo "DETECT-OK" || echo "DETECT-FAIL"
$PY scripts/make_error_tables.py    >/dev/null && echo "ERRTAB-OK" || echo "ERRTAB-FAIL"
$PY scripts/make_report_tables.py   >/dev/null && echo "REPTAB-OK" || echo "REPTAB-FAIL"
$PY scripts/run_significance.py     >/dev/null && echo "SIG-OK"    || echo "SIG-FAIL"
$PY scripts/run_natural_extremes.py >/dev/null && echo "NATEXT-OK" || echo "NATEXT-FAIL"
echo "QTMULTI-REFRESH done $(date +%H:%M:%S)"
