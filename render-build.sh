#!/usr/bin/env bash
# exit on error
set -o errexit

python -m pip install --upgrade pip

# Install CPU-only PyTorch and torchvision (saves ~3GB of CUDA libraries on Render)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt

# Pre-download MobileNetV3-Small weights so the server starts instantly
python -c "from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights as W; mobilenet_v3_small(weights=W.DEFAULT)"
