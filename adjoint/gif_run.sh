#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <directory with runs> <frame rate>"
    exit 1
fi


# Assign command-line arguments to variables
directory=$1
framerate=$2

if [ ! -d "$directory" ]; then
    echo "Error: Directory $directory does not exist."
    exit 1
fi


# Iterate through each .txt file and run the Python script
for folder in "$directory"/*; do
if [ -d "$folder" ]; then
        echo "Running create_gif.py with $folder"
        python create_gif.py "$folder" $framerate
        echo "$folder"
	wait
    fi
done
