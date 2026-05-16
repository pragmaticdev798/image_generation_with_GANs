#!/bin/bash

# Exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Clean up PyTorch cache to reduce memory usage
if [ -d "/opt/render/.cache/torch" ]; then
  rm -rf /opt/render/.cache/torch
fi
