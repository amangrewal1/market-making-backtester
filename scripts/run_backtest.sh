#!/usr/bin/env bash
set -euo pipefail

# Reproduce the 50k-episode CLOB run
python run_backtest.py --episodes 50000 --horizon 1000 --venue clob
python plot_results.py
