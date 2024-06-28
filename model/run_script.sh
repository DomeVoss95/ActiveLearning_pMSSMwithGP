#!/bin/bash

# Activate the conda environment
# source ~/miniconda3/etc/profile.d/conda.sh

# Initialize Conda (replace <SHELL_NAME> with your shell, e.g., bash)
conda init bash

conda activate ALenv

# Run the Python script
~/.conda/envs/ALenv/bin/python GPmodel2D.py
