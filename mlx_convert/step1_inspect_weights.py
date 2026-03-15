"""
Step 1: Load the GigaAM v3_ctc checkpoint and inspect weight keys/shapes.
This helps us understand the exact mapping needed for MLX conversion.
"""
import sys
sys.path.insert(0, "..")

import warnings
import torch

# Download the model using GigaAM's own download mechanism
import gigaam

model = gigaam.load_model("v3_ctc", device="cpu", fp16_encoder=False, use_flash=False)

print("=" * 80)
print("Model type:", type(model).__name__)
print("=" * 80)

# Get state dict
sd = model.state_dict()
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Total keys: {len(sd)}")
print()

# Print all keys with shapes
for k, v in sorted(sd.items()):
    print(f"  {k:70s} {str(list(v.shape)):>25s}  {v.dtype}")

# Also print the config
print("\n" + "=" * 80)
print("Model config:")
print("=" * 80)
from omegaconf import OmegaConf
print(OmegaConf.to_yaml(model.cfg))
