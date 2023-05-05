#!/bin/bash
# Start Xvfb
Xvfb :1 -screen 0 1400x900x24 &

# Set the DISPLAY environment variable
export DISPLAY=:1

# Run the Python script
python train.py