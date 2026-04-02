"""
Convert GigaAM v3 PyTorch checkpoint to MLX safetensors format.

Handles weight shape transpositions required by MLX conventions:
- PyTorch Conv1d: [out_ch, in_ch, kernel] → MLX Conv1d: [out_ch, kernel, in_ch]
- PyTorch Linear: same (no change needed, MLX nn.Linear uses same layout)
- RNNT LSTM weights properly mapped to MLX layout
- BatchNorm running stats need special handling

Usage:
    python convert_gigaam_to_mlx.py --model v3_ctc --output ./gigaam-v3-ctc-mlx
    python convert_gigaam_to_mlx.py --model v3_rnnt --output ./gigaam-v3-rnnt-mlx
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "..")

import numpy as np
import torch
from safetensors.numpy import save_file

VOCABULARY_V3 = [
    " ", "а", "б", "в", "г", "д", "е", "ж", "з", "и", "й", "к", "л", "м",
    "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ",
    "ы", "ь", "э", "ю", "я"
]


def transpose_conv1d_weight(w: np.ndarray) -> np.ndarray:
    """PyTorch Conv1d: [out, in, kernel] → MLX Conv1d: [out, kernel, in]"""
    return np.transpose(w, (0, 2, 1))


def sanitize_weights(state_dict: dict) -> dict:
    """
    Convert PyTorch state_dict to MLX-compatible weight dict.
    
    Key transformations:
    1. Conv1d weights: transpose [out, in, K] → [out, K, in]
    2. Skip preprocessor (mel filterbank computed at runtime in MLX)
    3. Handle LayerNorm batch_norm naming
    """
    mlx_weights = {}
    
    skipped = []
    for key, tensor in state_dict.items():
        w = tensor.detach().cpu().float().numpy()
        
        # Keep preprocessor mel filterbank and window for exact reproduction
        if key == "preprocessor.featurizer.0.mel_scale.fb":
            mlx_weights["mel_filterbank"] = w  # [n_fft//2+1, n_mels]
            skipped.append(key + " → mel_filterbank")
            continue
        if key == "preprocessor.featurizer.0.spectrogram.window":
            mlx_weights["stft_window"] = w  # [win_length]
            skipped.append(key + " → stft_window")
            continue
        if key.startswith("preprocessor."):
            skipped.append(key)
            continue
        
        # ALL 3D weights are Conv1d weights and need transposition
        # PyTorch Conv1d: [out, in, kernel] → MLX Conv1d: [out, kernel, in]
        # This covers: encoder.pre_encode.conv.*, encoder.layers.*.conv.*, head.decoder_layers.*
        if "weight" in key and len(w.shape) == 3:
            w = transpose_conv1d_weight(w)
            
        # RNNT LSTM conversions
        if "lstm.weight_ih" in key:
            mlx_weights[key.replace(".weight_ih_l0", ".Wx")] = w
            continue
        if "lstm.weight_hh" in key:
            mlx_weights[key.replace(".weight_hh_l0", ".Wh")] = w
            continue
        if "lstm.bias_ih" in key:
            # We must add bias_ih and bias_hh
            hh_key = key.replace(".bias_ih_l0", ".bias_hh_l0")
            hh_w = state_dict[hh_key].detach().cpu().float().numpy()
            mlx_weights[key.replace(".bias_ih_l0", ".bias")] = w + hh_w
            continue
        if "lstm.bias_hh" in key:
            # Already handled with bias_ih
            continue
            
        # RNNT Joint conversions
        if "joint.joint_net.1.weight" in key:
            mlx_weights[key.replace("joint_net.1.weight", "joint_net_linear.weight")] = w
            continue
        if "joint.joint_net.1.bias" in key:
            mlx_weights[key.replace("joint_net.1.bias", "joint_net_linear.bias")] = w
            continue
            
        # BatchNorm / LayerNorm:
        # GigaAM v3 uses layer_norm for conv_norm_type, so batch_norm is actually LayerNorm
        # The keys already use "batch_norm" name but the module is nn.LayerNorm
        # In MLX, this maps to nn.LayerNorm with weight/bias — same key structure works
        
        mlx_weights[key] = w
    
    if skipped:
        print(f"Skipped {len(skipped)} preprocessor keys: {skipped}")
    
    return mlx_weights


def build_config(model_name: str, cfg) -> dict:
    """Build config.json for the MLX model."""
    from omegaconf import OmegaConf
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    config = {
        "model_type": "gigaam",
        "model_name": model_name,
        "sample_rate": cfg_dict.get("sample_rate", 16000),
        "preprocessor": {
            "sample_rate": cfg_dict["preprocessor"].get("sample_rate", 16000),
            "features": cfg_dict["preprocessor"].get("features", 64),
            "win_length": cfg_dict["preprocessor"].get("win_length", 320),
            "hop_length": cfg_dict["preprocessor"].get("hop_length", 160),
            "n_fft": cfg_dict["preprocessor"].get("n_fft", 320),
            "center": cfg_dict["preprocessor"].get("center", False),
        },
        "encoder": {
            "feat_in": cfg_dict["encoder"].get("feat_in", 64),
            "n_layers": cfg_dict["encoder"].get("n_layers", 16),
            "d_model": cfg_dict["encoder"].get("d_model", 768),
            "subsampling": cfg_dict["encoder"].get("subsampling", "conv1d"),
            "subs_kernel_size": cfg_dict["encoder"].get("subs_kernel_size", 5),
            "subsampling_factor": cfg_dict["encoder"].get("subsampling_factor", 4),
            "ff_expansion_factor": cfg_dict["encoder"].get("ff_expansion_factor", 4),
            "self_attention_model": cfg_dict["encoder"].get("self_attention_model", "rotary"),
            "pos_emb_max_len": cfg_dict["encoder"].get("pos_emb_max_len", 5000),
            "n_heads": cfg_dict["encoder"].get("n_heads", 16),
            "conv_kernel_size": cfg_dict["encoder"].get("conv_kernel_size", 5),
            "conv_norm_type": cfg_dict["encoder"].get("conv_norm_type", "layer_norm"),
        },
    }
    
    # Model-specific head config
    if "ctc" in model_name:
        config["head_type"] = "ctc"
        config["head"] = {
            "feat_in": cfg_dict["head"].get("feat_in", 768),
            "num_classes": cfg_dict["head"].get("num_classes", 34),
        }
        config["vocabulary"] = cfg_dict["decoding"].get("vocabulary", VOCABULARY_V3)
    elif "rnnt" in model_name:
        config["head_type"] = "rnnt"
        config["head"] = {
            "decoder": cfg_dict["head"]["decoder"],
            "joint": cfg_dict["head"]["joint"],
        }
        if "vocabulary" in cfg_dict["decoding"]:
            config["vocabulary"] = cfg_dict["decoding"]["vocabulary"]
        else:
            config["vocabulary"] = VOCABULARY_V3
        # RNNT uses tokenizer
        config["tokenizer_model"] = "tokenizer.model"
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Convert GigaAM to MLX format")
    parser.add_argument("--model", type=str, default="v3_ctc",
                        help="Model name: v3_ctc, v3_rnnt, v3_e2e_ctc, v3_e2e_rnnt, v3_ssl")
    parser.add_argument("--output", type=str, default="./gigaam-v3-ctc-mlx",
                        help="Output directory")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Output dtype")
    args = parser.parse_args()
    
    print(f"Loading GigaAM model: {args.model}")
    import gigaam
    model = gigaam.load_model(args.model, device="cpu", fp16_encoder=False, use_flash=False)
    
    print("Extracting state dict...")
    state_dict = model.state_dict()
    print(f"  Total PyTorch keys: {len(state_dict)}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("Sanitizing weights for MLX...")
    mlx_weights = sanitize_weights(state_dict)
    print(f"  Total MLX keys: {len(mlx_weights)}")
    
    # Convert dtype
    np_dtype = {
        "float16": np.float16,
        "bfloat16": np.float16,  # safetensors doesn't support bf16 natively in numpy
        "float32": np.float32,
    }[args.dtype]
    
    mlx_weights = {k: v.astype(np_dtype) for k, v in mlx_weights.items()}
    
    # Build config
    print("Building config.json...")
    config = build_config(args.model, model.cfg)
    
    # Save
    os.makedirs(args.output, exist_ok=True)
    
    safetensors_path = os.path.join(args.output, "model.safetensors")
    print(f"Saving weights to {safetensors_path}...")
    save_file(mlx_weights, safetensors_path)
    
    config_path = os.path.join(args.output, "config.json")
    print(f"Saving config to {config_path}...")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # Copy tokenizer if rnnt
    if "rnnt" in args.model:
        import shutil
        tokenizer_src = os.path.join(os.path.expanduser("~/.cache/gigaam"), 
                                      f"{args.model}_tokenizer.model")
        if os.path.exists(tokenizer_src):
            tokenizer_dst = os.path.join(args.output, "tokenizer.model")
            shutil.copy2(tokenizer_src, tokenizer_dst)
            print(f"Copied tokenizer to {tokenizer_dst}")
    
    # Summary
    total_bytes = os.path.getsize(safetensors_path)
    print(f"\n✅ Conversion complete!")
    print(f"   Output: {args.output}")
    print(f"   Weights: {total_bytes / 1024 / 1024:.1f} MB ({args.dtype})")
    print(f"   Keys: {len(mlx_weights)}")
    print(f"   Config: {config_path}")


if __name__ == "__main__":
    main()
