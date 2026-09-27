# GigaAM v3 MLX — Russian ASR on Apple Silicon

GigaAM v3 (Conformer encoder, 16 layers, 768d) converted to [MLX](https://github.com/ml-explore/mlx) for fast inference on Apple Silicon.
Supports **CTC**, **RNNT** and the end-to-end **E2E CTC / E2E RNNT** models (punctuated, normalized text).

On 200 Golos Crowd clips MLX transcripts are identical to upstream PyTorch for all four models;
speed, WER and parity numbers are in [Evaluation](#evaluation) and are reproducible with `eval/`.

## Quick Start

### 1. Install dependencies

```bash
uv venv .venv
uv pip install mlx safetensors numpy
# For streaming from microphone:
uv pip install sounddevice
# For conversion (step 2) the PyTorch package is needed once:
uv pip install -e "..[torch]"
```

### 2. Convert the model (one-time)

```bash
# CTC model (fastest)
python convert_gigaam_to_mlx.py --model v3_ctc --output ./gigaam-v3-ctc-mlx

# RNNT model (lower WER, sequential decode)
python convert_gigaam_to_mlx.py --model v3_rnnt --output ./gigaam-v3-rnnt-mlx

# End-to-end models (SentencePiece output with punctuation/normalization)
python convert_gigaam_to_mlx.py --model v3_e2e_ctc --output ./gigaam-v3-e2e-ctc-mlx
python convert_gigaam_to_mlx.py --model v3_e2e_rnnt --output ./gigaam-v3-e2e-rnnt-mlx
```

This creates a directory with:
- `model.safetensors` — weights (≈421–425 MB, fp16)
- `config.json` — model configuration + vocabulary (for E2E: all SentencePiece pieces)
- `tokenizer.model` — SentencePiece model (E2E only; the runtime decodes from `config.json` and does not need the `sentencepiece` package)

The upstream encoder weights are stored in fp16, so `--dtype float32` doubles the file without changing them (only the small CTC head is fp32 upstream).

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

## mlx-audio Compatibility

The `StreamingResult` dataclass follows the same contract as [mlx-audio](https://github.com/Blaizzy/mlx-audio) Parakeet/Whisper streaming, making it straightforward to integrate GigaAM as an mlx-audio STT model.

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

## Evaluation

Measured on Apple M4 Pro (24 GB), macOS 26.6.2, mlx 0.32.2, torch 2.14.0, on 200 Golos Crowd test clips
(823.85 s) — the sample of [garrrikkotua/gigaam-vs-whisper-russian](https://github.com/garrrikkotua/gigaam-vs-whisper-russian),
verified by SHA-256. Full tables and method: [`eval/results/RESULTS.md`](eval/results/RESULTS.md);
to reproduce: [`eval/README.md`](eval/README.md).

**Parity with PyTorch** (upstream model on CPU, fp32): identical transcripts on 200/200 clips for all four
models; encoder output max |Δ| ≤ 9e-5 per clip (outputs are O(1)); CTC frame labels 100% identical.

**Accuracy** (corpus WER; E2E output normalized — lowercase, ё→е, numbers → words, no punctuation):

| | v3_ctc | v3_rnnt | v3_e2e_ctc | v3_e2e_rnnt |
|---|---:|---:|---:|---:|
| WER | 1.70% | 1.50% | 6.49% | 5.69% |
| WER, 168 clips without digits/Latin in E2E output | 1.35% | 1.23% | 2.70% | 2.09% |

E2E models write "HD", "4K", "15-й", phone numbers as digit groups where Golos references spell words out,
so their full-corpus WER is dominated by formatting. The v3_rnnt result (15 errors / 1001 words) matches the
archived PyTorch result of the garrrikkotua benchmark, hypothesis for hypothesis.

**Speed** (RTFx = audio seconds / wall seconds, median of 3 passes after warm-up; `short` = the 200 clips one
by one, median 3.9 s; `long` = 38 segments of ≈20 s):

| | MLX | PyTorch MPS | PyTorch CPU |
|---|---:|---:|---:|
| v3_ctc short / long | 201 / 285 | 201 / 268 | 10 / 41 |
| v3_rnnt short / long | 91 / 102 | 37 / 41 | 10 / 39 |
| v3_e2e_ctc short / long | 205 / 289 | 202 / 271 | 10 / 41 |
| v3_e2e_rnnt short / long | 99 / 112 | 45 / 48 | 10 / 41 |

For CTC models MLX and PyTorch MPS are on par; for RNNT models MLX is 2.3–2.5× faster (the greedy
decode loop is cheaper). MLX peak memory ≈1.8 GB with the default fp32 upcast at load, ≈1.35 GB with
`load_model(..., dtype=None)` at 5–20% lower speed (see RESULTS.md).

**Memory note.** MLX caches freed buffers for every new input shape; with varying audio lengths (and
the growing buffer of `gigaam-stream`) the cache can grow to most of unified memory. Cap it in
long-running processes: `mx.set_cache_limit(1 << 30)` (≈4% slower on short clips). `gigaam-stream`
does this.

## Architecture

All four models share the same Conformer encoder:

```
Audio (16kHz) → Log-Mel Spectrogram (64 bins)
             → Conv1d Subsampling (4x stride)
             → 16× Conformer Layers:
                  ├─ FFN₁ (half-step residual)
                  ├─ RoPE Multi-Head Self-Attention (16 heads)
                  ├─ Convolution Module (GLU + depthwise conv)
                  └─ FFN₂ (half-step residual)
             → CTC Head (Conv1d → vocabulary + blank → greedy decode)
                or
             → RNNT Head (Joint + LSTM Decoder → greedy decode)
```

Key implementation details:
- **RoPE before projections**: GigaAM applies rotary embeddings to raw input *before* Q/K/V linear projections (non-standard)
- **RoPE base 5000**: upstream calls `RotaryPositionalEmbedding(d_model // n_heads, pos_emb_max_len)`, whose second positional parameter is `base`, so the base is 5000 rather than the usual 10000
- **Exact mel filterbank**: Saved from PyTorch to avoid HTK recomputation differences
- **All Conv1d weights transposed**: `[out, in, K]` → `[out, K, in]` for MLX convention
- **RNNT LSTM weights**: PyTorch `(weight_ih, weight_hh, bias_ih, bias_hh)` mapped to MLX `(Wx, Wh, bias)` layout
- **SentencePiece without sentencepiece**: E2E pieces are exported to `config.json`; `decode_sentencepiece` reproduces `SentencePieceProcessor.decode` (byte fallback, `<unk>`, dummy prefix) — checked by `eval/test_decode.py`

## License

GigaAM model weights: [ai-sage/GigaAM](https://huggingface.co/ai-sage/GigaAM) — check their license.
MLX conversion code: MIT.
