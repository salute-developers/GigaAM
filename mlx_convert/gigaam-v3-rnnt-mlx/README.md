---
license: mit
language:
  - ru
library_name: mlx
tags:
  - mlx
  - speech-recognition
  - asr
  - rnnt
  - conformer
  - russian
  - apple-silicon
base_model: ai-sage/GigaAM
pipeline_tag: automatic-speech-recognition
---

# GigaAM v3 RNNT — MLX (Apple Silicon)

GigaAM v3 RNNT (Conformer, 16 layers, 768d + RNNT Joint & Decoder) converted to [MLX](https://github.com/ml-explore/mlx) for native inference on Apple Silicon.

**48× realtime** on M4 — transcribes 11 seconds of Russian speech in 230ms.
Compared to the CTC version, RNNT offers **~9% lower Word Error Rate (WER)** across benchmarks due to the autoregressive joint language modeling loop, with slightly slower sequential decoding.

Original model: [ai-sage/GigaAM](https://huggingface.co/ai-sage/GigaAM)

## Quick Start

```bash
pip install mlx safetensors numpy
```

```python
from huggingface_hub import snapshot_download

model_dir = snapshot_download("al-bo/gigaam-v3-rnnt-mlx")
```

Or use with the inference code from [GigaAM MLX](https://github.com/salute-developers/GigaAM/tree/main/mlx_convert):

```python
from gigaam_mlx import load_model, load_audio

model = load_model("./gigaam-v3-rnnt-mlx")
text = model.transcribe(load_audio("audio.wav"))
print(text)
# → ничьих не требуя похвал счастлив уж я надеждой сладкой
```

## Architecture

```
Audio (16kHz) → Log-Mel Spectrogram (64 bins)
             → Conv1d Subsampling (4× stride)
             → 16× Conformer Layers:
                  ├─ FFN₁ (half-step residual)
                  ├─ RoPE Multi-Head Self-Attention (16 heads)
                  ├─ Convolution Module (GLU + depthwise conv)
                  └─ FFN₂ (half-step residual)
             → RNNT Head (Joint + LSTM Decoder)
             → Greedy Decode
```

## Performance (Apple M4)

| Metric | Value |
|--------|-------|
| Batch (11s audio) | **230ms** (48× realtime) |
| Model size | 423 MB (fp16) |
| Parameters | ~222M |

## Files

- `model.safetensors` — weights (fp16, 423 MB)
- `config.json` — model config + vocabulary (34 Russian characters)

## Conversion

Converted from PyTorch using `convert_gigaam_to_mlx.py`.
LSTM weights are transformed from PyTorch `(weight_ih, weight_hh, bias_ih, bias_hh)` to MLX layout `(Wx, Wh, bias)`.

## License

MLX conversion code: MIT.
Model weights: see [ai-sage/GigaAM](https://huggingface.co/ai-sage/GigaAM) license.