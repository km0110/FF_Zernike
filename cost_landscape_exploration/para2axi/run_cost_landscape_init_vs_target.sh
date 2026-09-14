#!/bin/bash
set -e

PARAMS="params.txt"

python save_init_and_target_coeffs.py "$PARAMS"
python plot_cost_landscape.py "$PARAMS" coeffs_init.csv coeffs_target.csv cost_landscape_init_vs_target.pdf
