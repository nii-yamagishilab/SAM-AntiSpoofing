#!/bin/bash

# Ensure the script exits on errors and uninitialized variables
set -euo pipefail

# Function to print usage information
usage() {
    echo "Usage: $0 <exp_dir> [epoch]"
    echo "  exp_dir: Directory containing experiment results"
    echo "  epoch: Specific epoch to evaluate (default: 'best')"
    exit 1
}

# Check for at least one argument
if [ $# -lt 1 ]; then
    echo "Error: Missing required argument <exp_dir>."
    usage
fi

# Get arguments
exp_dir=$1
epoch=${2:-best} # Default to 'best' if not provided

# Validate that the experiment directory exists
if [ ! -d "$exp_dir" ]; then
    echo "Error: Directory '$exp_dir' does not exist."
    exit 1
fi

# Define the evaluation datasets
evals=('19LA' 'ITW' 'FOR' 'WF' 'ADDE' 'SCE' '21LA' '21DF')

# Loop through the evaluation datasets and run the Python script
for eval in "${evals[@]}"; do
    echo "Running evaluation for: $eval"
    python evaluate.py --exp_dir "$exp_dir" --epoch "$epoch" --eval "$eval"
done

echo "All evaluations completed successfully."

