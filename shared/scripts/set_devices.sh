#!/bin/bash

# Check if CUDA is available
if ! command -v nvcc &> /dev/null
then
    echo "CUDA is not installed. Please install CUDA Toolkit."
    exit 1
fi
# Prompt the user to select GPU devices
echo "Available GPU devices:"
gpustat

read -p "Enter the device IDs you want to use (comma-separated, e.g., 0,1): " selected_devices


# Set CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=${selected_devices:-0,1}

echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
