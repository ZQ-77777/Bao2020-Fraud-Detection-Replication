#!/usr/bin/env bash
# Full pipeline. Quick smoke test first:  bash run_all.sh --trees 300
set -e
cd "$(dirname "$0")"
EXTRA="$@"
python scripts/01_check_data.py $EXTRA
python scripts/02_replicate.py $EXTRA
python scripts/03_serial_fraud.py --seeds 3 $EXTRA
python scripts/04_extension_gbm.py --seeds 3 $EXTRA
python scripts/05_tune_rusboost.py $EXTRA
python scripts/06_interpret.py --model gbm $EXTRA
python scripts/07_collect_results.py
python scripts/11_make_report_figures.py
