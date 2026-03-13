# GigaAM v3 MLX — Russian ASR on Apple Silicon

GigaAM v3 (Conformer encoder, 16 layers, 768d) converted to [MLX](https://github.com/ml-explore/mlx) for fast inference on Apple Silicon.
Supports both **CTC** and **RNNT** models.

**139x realtime** for CTC on M4 — transcribes 11 seconds of Russian speech in 81ms.
**48x realtime** for RNNT on M4 — transcribes 11 seconds in 230ms (higher quality, sequential decode).

## Quick Start

### 1. Install dependencies

```bash
uv venv .venv
uv pip install mlx safetensors numpy
# For streaming from microphone:
uv pip install sounddevice
```

### 2. Convert the model (one-time)

```bash
# Downloads GigaAM v3 CTC from HuggingFace and converts to MLX safetensors
python convert_gigaam_to_mlx.py --model v3_ctc --output ./gigaam-v3-ctc-mlx

# Optional: fp32 version
python convert_gigaam_to_mlx.py --model v3_ctc --output ./gigaam-v3-ctc-mlx-fp32 --dtype float32
```

This creates a directory with:
- `model.safetensors` — weights (421 MB fp16, 842 MB fp32)
- `config.json` — model configuration + vocabulary

### 3. Transcribe

```bash
# Single file
python gigaam-cli -f audio.wav

# Streaming from file
python gigaam-stream --file audio.wav

# Live microphone streaming
python gigaam-stream
```

---

## Python API

### Basic transcription

```python
from gigaam_mlx import load_model, load_audio

model = load_model("./gigaam-v3-ctc-mlx")
audio = load_audio("audio.wav")  # any format, resampled to 16kHz via ffmpeg
text = model.transcribe(audio)
print(text)
# → ничьих не требуя похвал счастлив уж я надеждой сладкой
```

### Streaming (pre-recorded file)

Process audio incrementally, yielding results every N seconds:

```python
from gigaam_mlx import load_model, load_audio, StreamingConfig

model = load_model("./gigaam-v3-ctc-mlx")
audio = load_audio("audio.wav")

config = StreamingConfig(step_duration=1.0)  # yield every 1s

for result in model.stream_generate(audio, config):
    print(f"[{result.audio_position:.1f}s] {result.cumulative_text}")
    # [1.0s] ничьих не требуя
    # [2.0s] ничьих не требуя похвал
    # [3.0s] ничьих не требуя похвал счастлив уж я надеж
    # ...
```

`StreamingResult` fields:

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | New text since last emission |
| `cumulative_text` | `str` | Full transcription so far |
| `is_final` | `bool` | `True` if last chunk |
| `audio_position` | `float` | Current position in seconds |
| `audio_duration` | `float` | Total audio duration |
| `progress` | `float` | 0.0–1.0 |
| `language` | `str` | Always `"ru"` |

### Streaming (live microphone)

For real-time transcription, call `stream_live()` with a growing audio buffer:

```python
import numpy as np
import mlx.core as mx
from gigaam_mlx import load_model

model = load_model("./gigaam-v3-ctc-mlx")

# Accumulate audio from microphone (16kHz float32 mono)
buffer = np.zeros(0, dtype=np.float32)

# Called every N ms with new audio
def on_audio_chunk(chunk: np.ndarray):
    global buffer
    buffer = np.concatenate([buffer, chunk])

    result = model.stream_live(mx.array(buffer))
    print(f"\r{result.cumulative_text}", end="", flush=True)
```

### StreamingConfig options

```python
from gigaam_mlx import StreamingConfig

config = StreamingConfig(
    step_duration=1.0,      # process every 1s (default: 2s)
    chunk_duration=2.0,     # unused for stream_generate (kept for compat)
    context_duration=3.0,   # unused for stream_generate (kept for compat)
)
```

---

## mlx-audio Integration

GigaAM MLX is designed to be compatible with the [mlx-audio](https://github.com/Blaizzy/mlx-audio) ecosystem.

### Using with mlx-audio's generate interface

The `StreamingResult` dataclass follows the same contract as mlx-audio's Parakeet and Whisper streaming:

```python
# mlx-audio style usage
for result in model.stream_generate(audio, config):
    if result.is_final:
        print(f"Final: {result.cumulative_text}")
    else:
        print(f"[{result.progress:.0%}] {result.text}")
```

### Adding GigaAM to mlx-audio as a model

To register GigaAM as an mlx-audio STT model, create the following structure in `mlx_audio/stt/models/gigaam/`:

```
mlx_audio/stt/models/gigaam/
├── __init__.py          # exports Model, ModelConfig
├── gigaam.py            # Model class with generate() method
├── conformer.py         # ConformerEncoder, layers
├── audio.py             # log_mel_spectrogram
└── README.md
```

**`__init__.py`:**

```python
from .gigaam import Model, ModelConfig
```

**`gigaam.py`** (key parts):

```python
import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Optional, Union, Generator
from pathlib import Path

from mlx_audio.stt.models.base import STTOutput


@dataclass
class ModelConfig:
    """Wraps GigaAM config.json."""
    def __init__(self, config: dict):
        self._config = config

    @classmethod
    def from_dict(cls, config: dict) -> "ModelConfig":
        return cls(config)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        cfg = config._config
        # ... build ConformerEncoder + CTCHead from cfg ...
        self.sample_rate = cfg.get("sample_rate", 16000)
        self.vocabulary = cfg.get("vocabulary", [])

    def generate(
        self,
        audio: Union[str, Path, mx.array],
        *,
        stream: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Union[STTOutput, Generator]:
        """mlx-audio compatible generate interface."""
        from mlx_audio.stt.utils import load_audio

        if isinstance(audio, (str, Path)):
            audio = load_audio(str(audio), self.sample_rate)

        if stream:
            return self._stream_generate(audio, **kwargs)

        text = self._transcribe(audio)

        return STTOutput(
            text=text,
            segments=[{"text": text, "start": 0.0, "end": len(audio) / self.sample_rate}],
            language="ru",
        )

    def _transcribe(self, audio: mx.array) -> str:
        """CTC forward + greedy decode."""
        # ... mel → encoder → ctc_head → argmax → collapse ...
        pass

    def _stream_generate(self, audio, **kwargs):
        """Yields StreamingResult compatible with mlx-audio."""
        # ... growing buffer approach ...
        pass
```

Then register in `mlx_audio/stt/models/__init__.py`:

```python
from . import gigaam  # add this line
```

And in `mlx_audio/stt/utils.py` add the remapping:

```python
MODEL_REMAPPING = {
    # ... existing entries ...
    "gigaam": "gigaam",
}
```

The model will then work with:

```python
from mlx_audio.stt.utils import load_model

model = load_model("path/to/gigaam-v3-ctc-mlx")  # or HuggingFace repo
result = model.generate("audio.wav")
print(result.text)

# Streaming
for chunk in model.generate("audio.wav", stream=True):
    print(chunk.text, end="", flush=True)
```

### HuggingFace model card

To upload to HuggingFace for `load_model("your-name/gigaam-v3-ctc-mlx")`:

```
gigaam-v3-ctc-mlx/
├── config.json          # must include "model_type": "gigaam"
├── model.safetensors    # 421 MB (fp16)
└── README.md            # model card
```

The `config.json` already has `"model_type": "gigaam"` so mlx-audio's auto-detection will route to the correct model class.

---

## CLI Tools

### `gigaam-cli` — Single-file transcription

```bash
python gigaam-cli -f audio.wav                    # default model
python gigaam-cli -f audio.wav -m /path/to/model  # custom model path
python gigaam-cli -f audio.wav --no-prints         # only text to stdout
```

### `gigaam-stream` — Real-time streaming

```bash
# Live microphone
python gigaam-stream
python gigaam-stream --step 1000              # update every 1s
python gigaam-stream --step 500               # update every 0.5s

# File streaming (simulates real-time)
python gigaam-stream --file audio.wav
python gigaam-stream --file audio.wav --step 1000 --no-overwrite
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--step N` | 2000 | Process every N ms |
| `--file PATH` | — | File mode instead of microphone |
| `--model PATH` | auto | Model directory |
| `--no-overwrite` | off | Print incrementally (don't clear line) |
| `--vad-threshold` | 0.003 | Energy threshold for speech detection |

### `gigaam-transcribe` — Shell wrapper

```bash
# Uses bundled Python venv automatically
gigaam-transcribe -f audio.wav --no-prints

# Symlink for PATH access
ln -s /path/to/mlx_convert/gigaam-transcribe /usr/local/bin/gigaam-transcribe
```

---

## OpenWispr Integration

GigaAM can be used as a transcription engine in [OpenWispr](https://github.com/nicklama/open-wispr) (push-to-talk macOS app):

```bash
# Switch to GigaAM engine
open-wispr set-engine gigaam
open-wispr status
# Engine: gigaam
# GigaAM: ready
```

Or in `~/.config/open-wispr/config.json`:

```json
{
  "engine": "gigaam",
  "language": "ru"
}
```

---

## Benchmarks (Apple M4)

| | GigaAM MLX (fp16) | GigaAM PyTorch | Whisper CPP (small) |
|--|---|---|---|
| **Batch (11s audio)** | **81ms** | 400ms | 1130ms |
| **Realtime factor** | **139x** | 28x | 10x |
| **Stream (1s step)** | **57ms/step** | — | ~300ms/step |
| **Model size** | 421 MB | 842 MB | 465 MB |
| **Language** | Russian | Russian | Multilingual |

## Architecture

```
Audio (16kHz) → Log-Mel Spectrogram (64 bins)
             → Conv1d Subsampling (4x stride)
             → 16× Conformer Layers:
                  ├─ FFN₁ (half-step residual)
                  ├─ RoPE Multi-Head Self-Attention (16 heads)
                  ├─ Convolution Module (GLU + depthwise conv)
                  └─ FFN₂ (half-step residual)
             → CTC Head (Conv1d → 35 classes)
             → Greedy Decode (collapse repeated + remove blanks)
```

Key implementation details:
- **RoPE before projections**: GigaAM applies rotary embeddings to raw input *before* Q/K/V linear projections (non-standard)
- **Exact mel filterbank**: Saved from PyTorch to avoid HTK recomputation differences
- **All Conv1d weights transposed**: `[out, in, K]` → `[out, K, in]` for MLX convention

## License

GigaAM model weights: [ai-sage/GigaAM](https://huggingface.co/ai-sage/GigaAM) — check their license.
MLX conversion code: MIT.
