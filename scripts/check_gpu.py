"""Run this once the GPU is free to confirm PyTorch picks it up.

Not run as part of Phase 0 setup because another model was occupying the GPU
at the time — avoid contending with it.
"""
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
