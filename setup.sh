#!/bin/bash

set -e

ROOT_DIR=$(pwd)

echo "Setting up project structure under: $ROOT_DIR"

mkdir -p "$ROOT_DIR/data" \
         "$ROOT_DIR/output" 




Real_file="FitMRI_fitbit_intraday_steps_trainingData.csv"
DATA_DIR="$ROOT_DIR/data"

echo "Project successfully set up."

if [ -f "$Real_file" ]; then
    echo "Found $Real_file in current directory."
    echo "Copying $Real_file to data directory."
    cp "$Real_file" "$DATA_DIR"

else
    echo "Error: $Real_file not found in current directory."
    exit 1
fi

echo "Setup complete."
