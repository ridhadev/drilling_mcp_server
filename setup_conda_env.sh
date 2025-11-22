#!/bin/bash
# Setup script for drilling_mcp conda environment
# This script creates a conda environment with Python 3.10 and installs dependencies

# Create conda environment with Python 3.10
conda create -n drilling_mcp python=3.10 -y

# Activate the environment
conda activate drilling_mcp

# Install packages from requirements.txt
pip install -r requirements.txt

# Verify installation
echo "Environment setup complete!"
echo "To activate the environment in the future, run: conda activate drilling_mcp"

