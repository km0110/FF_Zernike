#!/bin/bash

# Runs <python_script> (a create_video.py-style script) over every run folder
# in <directory>. Shared across all experiment directories -- lives in
# common/. <python_script> is resolved relative to the caller's cwd (same
# convention as dir_run.sh), so each directory passes whichever video-creation
# script is right for it -- its own local one, or ../common/create_video.py.

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <python_script> <directory with runs> <frame rate>"
    exit 1
fi

python_script="$1"
directory="$2"
framerate="$3"

if [ ! -d "$directory" ]; then
    echo "Error: Directory $directory does not exist."
    exit 1
fi

for folder in "$directory"/*; do
if [ -d "$folder" ]; then
        echo "Running $python_script with $folder"
        python "$python_script" "$folder" "$framerate"
        echo "$folder"
	wait
    fi
done
