#!/bin/bash

# Runs make_plots.py over every run folder in <directory>. Shared across all
# experiment directories -- lives in common/. Any arguments after
# <directory> are passed straight through to make_plots.py.
#
# Usage: $0 <directory with runs> [make_plots.py args...]
#   e.g. $0 int_para2axi_n_f20/runs --fom-module zernike
#        $0 int_para_to_para/runs --fom-module local
#        $0 int_para2axi_n_f20/runs --fom-module zernike --generations

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <directory with runs> [make_plots.py args...]"
    exit 1
fi

directory=$1
shift

if [ ! -d "$directory" ]; then
    echo "Error: Directory $directory does not exist."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for folder in "$directory"/*; do
if [ -d "$folder" ]; then
        echo "Running make_plots.py with $folder"
        python "$SCRIPT_DIR/make_plots.py" "$folder" "$@"
        echo "$folder"
	wait
    fi
done
