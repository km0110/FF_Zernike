#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <directory with runs> "
    exit 1
fi


# Assign command-line arguments to variables
directory=$1

if [ ! -d "$directory" ]; then
    echo "Error: Directory $directory does not exist."
    exit 1
fi


# Iterate through each .txt file and run the Python script
for folder in "$directory"/*; do
if [ -d "$folder" ]; then
        echo "Running make_plots.py with $folder"
        python make_plots.py "$folder" 
        echo "$folder"
	wait
    fi
done
