#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <python_script> <directory>"
    exit 1
fi


# Assign command-line arguments to variables
python_script="$1"
directory="$2"

if [ ! -d "$directory" ]; then
    echo "Error: Directory $directory does not exist."
    exit 1
fi


# Iterate through each .txt file and run the Python script
for file in "$directory"/*.txt; do
if [ -f "$file" ]; then
        echo "Running $python_script with $file"
        full_path="$directory/$(basename "$file")"
        python "$python_script" "$full_path"
        wait
    fi
done
