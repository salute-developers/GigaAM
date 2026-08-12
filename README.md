# GigaAM: the family of open-source acoustic models for speech processing

<div align="center" style="line-height: 1;">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2506.01192-b31b1b.svg)](https://arxiv.org/abs/2506.01192)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Models-yellow.svg)](https://huggingface.co/collections/ai-sage/gigaam)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salute-developers/GigaAM/blob/main/colab_example.ipynb)

</div>

<hr>

![plot](./assets/gigaam_scheme.svg)

## Latest News
* 2026/11 — GigaAM-v4 (ru-en, surpassing Whisper-large-v3 on English) and GigaAM-Multilingual-v2 (**21** languages, SOTA for CIS): new Zapformer architecture, streaming ASR (160/320/640 ms), dual-head decoding, and native GigaAM VAD for long-form inference.
* 2026/06 — GigaAM Multilingual: 220M / 600M encoders pre-trained on **2M hours** across **70+ languages**; charwise CTC ASR with best-in-class WER on Russian, Kazakh, Kyrgyz, and Uzbek (moderate on English). Our [research paper](https://arxiv.org/abs/2607.10371) was accepted to InterSpeech 2026!
* 2026/04 — [model fine-tuning](#model-fine-tuning) (CTC / RNNT), word-level timestamps, [Triton Inference Server](#triton-inference-server-and-tensorrt)
* 2025/11 — GigaAM-v3: **30%** WER reduction on new data domains; GigaAM-v3-e2e: end-to-end transcription support (**70:30** win in Side-by-Side vs Whisper-large-v3)
* 2025/06 — Our [research paper on GigaAM](https://arxiv.org/abs/2506.01192) was accepted to InterSpeech 2025!
* 2024/12 — [MIT License](./LICENSE), GigaAM-v2 (**-15%** and **-12%** WER Reduction for CTC and RNN-T models, respectively), [ONNX export support](#onnx-export-and-inference)
* 2024/05 — GigaAM-RNNT (**-19%** WER Reduction), [long-form inference using external Voice Activity Detection](#model-inference)
* 2024/04 — GigaAM Release: GigaAM-CTC ([SoTA Speech Recognition model for the Russian language](#model-performance)), [GigaAM-Emo](#model-performance)

---

## Setup

### Requirements
- Python ≥ 3.10
- [ffmpeg](https://ffmpeg.org/) installed and added to your system's PATH

### Install the GigaAM Package

```bash
# Clone the repository
git clone https://github.com/salute-developers/GigaAM.git
cd GigaAM

# Install the package requirements
pip install -e .[torch]

# (optionally) Verify the installation:
pip install -e ".[tests]"
pytest -v tests/test_loading.py -m partial  # or `-m full` to test all models
```

---

## GigaAM overview

GigaAM is a family of [Zapformer](https://github.com/k2-fsa/icefall/pull/2082)- and [Conformer](https://arxiv.org/pdf/2005.08100.pdf)-based foundational models (220M–780M parameters) pre-trained on up to 10M hours of multilingual speech data. It serves as the backbone for the entire GigaAM family, enabling state-of-the-art fine-tuned performance in speech recognition and emotion recognition. We fine-tuned the GigaAM encoder for ASR using [CTC](https://www.cs.toronto.edu/~graves/icml_2006.pdf) and [RNNT](https://arxiv.org/abs/1211.3711) decoders. The GigaAM family includes six lines of models:

| | Pretrain Method | Pretrain (hours) | ASR (hours) | Available Versions |
| :--- | :--- | :--- | :--- | :---: |
| **v1** | [Wav2vec 2.0](https://arxiv.org/abs/2006.11477) | 50,000 | 2,000 | `v1_ssl`, `emo`, `v1_ctc`, `v1_rnnt` |
| **v2** | [HuBERT–CTC](https://arxiv.org/abs/2506.01192) | 50,000 | 2,000 | `v2_ssl`, `v2_ctc`, `v2_rnnt` |
| **v3** | HuBERT–CTC | 700,000 | 4,000 | `v3_ssl`, `v3_ctc`, `v3_rnnt`, `v3_e2e_ctc`, `v3_e2e_rnnt` |
| **v4** | HuBERT-style | 7,000,000 | 58,000  | `v4_ssl`, `v4_ctc`, `v4_rnnt` |
| **multilingual** | HuBERT–style | 2,000,000 | 50,000 | `multilingual_ssl`, `multilingual_large_ssl`, `multilingual_ctc`, `multilingual_large_ctc` |
| **multilingual_v2** | HuBERT–style | 10,000,000 | 155,000 | `multilingual_v2_ssl`, `multilingual_v2_rnnt` |

## Model Performance

`GigaAM-v4` outperforms `GigaAM-v3-E2E` by **35%** on Russian ASR and achieves best-in-class performance for its model size on English, surpassing `Whisper-large-v3`. `GigaAM-Multilingual-v2` achieves state-of-the-art results on CIS languages and outperforms leading open-weight models on European languages. Both ASR models support streaming with `160, 320, and 640 ms` chunk sizes and feature a dual-head architecture for E2E and raw transcriptions.

For detailed results, see [here](./evaluation.md).

---

## Usage

### Model inference

```python
import gigaam

# Load test audio
audio_path = gigaam.utils.download_short_audio()
long_audio_path = gigaam.utils.download_long_audio()

# Audio embeddings
model_name = "v4_ssl"       # Or `multilingual_v2_ssl`
model = gigaam.load_model(model_name)
embedding, _ = model.embed_audio(audio_path)
print(embedding)

# ASR
model_name = "v4_rnnt"      # Or `v4_ctc` / `multilingual_v2_rnnt`
model = gigaam.load_model(model_name)
transcription = model.transcribe(audio_path)
print("E2E:", transcription["e2e"].text)
print("Raw:", transcription["raw"].text)

# Streaming ASR with 160, 320, or 640 ms future context
audio = gigaam.load_audio(audio_path)
stream = gigaam.load_model("v4_rnnt", streaming=True, future_context_ms=160)
for packet in audio.split(16000):
    partial = stream.transcribe_partial(packet)
    print({head: result.text for head, result in partial.items()})
tail = stream.flush()
print({head: result.text for head, result in tail.items()})

# ASR with word-level timestamps
result = model.transcribe(audio_path, word_timestamps=True)
for word in result["e2e"].words:
    print(f"  [{word.start:.2f} - {word.end:.2f}] {word.text}")

# and long-form ASR
result = model.transcribe_longform(long_audio_path)
for segment in result["e2e"].segments:
   print(f"[{gigaam.format_time(segment.start)} - {gigaam.format_time(segment.end)}]: {segment.text}")

# Voice activity detection
vad = gigaam.load_model("vad")
probs = vad.speech_probs(gigaam.load_audio(audio_path))
print("Speech probabilities:", tuple(probs.shape), probs[:5].tolist())

# Emotion recognition
model = gigaam.load_model("emo")
emotion2prob = model.get_probs(audio_path)
print(", ".join([f"{emotion}: {prob:.3f}" for emotion, prob in emotion2prob.items()]))
```

#### Confidence scores (native PyTorch)

Pass `return_confidence=True` to score a short utterance or each long-form segment. Add `word_timestamps=True` for per-word scores. Models v1–v3 return a single result. In v4 each head scores its own transcript.

```python
model = gigaam.load_model("v4_rnnt")
short = model.transcribe(
    audio_path, return_confidence=True, word_timestamps=True
)
print(short["e2e"].confidence)
for word in short["e2e"].words:
    print(word.text, word.confidence)

long = model.transcribe_longform(long_audio_path, return_confidence=True)
for segment in long["e2e"].segments:
    print(segment.text, segment.confidence)
```

The score is the geometric mean of the probabilities of non-blank tokens emitted by the greedy decoder. It can help rank passages for review. It is uncalibrated, so even a high-scoring word can be wrong. The field is `None` when scoring is off or there are no emitted tokens.

### Model Fine-tuning

CTC, RNNT, and SSL models can be fine-tuned on custom data using PyTorch Lightning. For a detailed description of all training arguments, see [`train_utils/README.md`](./train_utils/README.md). End-to-end examples with different VRAM constraints are available in [`train_utils/example.ipynb`](./train_utils/example.ipynb).

### Loading from Hugging Face

> **Note:** Install requirements from the [example](./colab_example.ipynb).

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "ai-sage/GigaAM-v4", revision="rnnt", trust_remote_code=True,
    streaming=True, future_context_ms=160,
)

model = AutoModel.from_pretrained(
    "ai-sage/GigaAM-Multilingual-v2", revision="rnnt",
    trust_remote_code=True,
)
```

### ONNX Export and Inference

1. **Full-context export and inference:**
   ```python
   from gigaam.onnx_utils import load_onnx, infer_onnx

   onnx_dir = "onnx"
   model_name = "v4_rnnt"
   gigaam.load_model(model_name).to_onnx(onnx_dir)
   sessions, cfg = load_onnx(onnx_dir, model_name)
   result = infer_onnx([audio_path], cfg, sessions)
   print("E2E:", result["e2e"][0])
   print("Raw:", result["raw"][0])
   ```

2. **Streaming export and inference** use the same API and support 160, 320, or 640 ms future contexts:
   ```python
   from gigaam.onnx_utils import OnnxGigaAMStreaming, load_onnx

   stream = gigaam.load_model("v4_rnnt", streaming=True, future_context_ms=160)
   prefix = stream.to_onnx("onnx_streaming")
   sessions, cfg = load_onnx("onnx_streaming", prefix)
   runner = OnnxGigaAMStreaming(cfg, sessions)
   for packet in audio.split(16000):
       partial = runner.transcribe_partial(packet)
       print({head: result.text for head, result in partial.items()})
   tail = runner.flush()
   print({head: result.text for head, result in tail.items()})
   ```

These and more advanced examples can be found in the [Colab notebook](https://colab.research.google.com/github/salute-developers/GigaAM/blob/main/colab_example.ipynb).

### Triton Inference Server and TensorRT

Dual-head models can be served in ONNX/TRT format through Triton Inference Server. For setup instructions, model conversion, and deployment details, see the [Triton Inference Server documentation](./triton_scripts/README.md).

---

## Citation

If you use GigaAM in your research, please cite our papers:

```bibtex
@misc{gigaam_multilingual,
      title={GigaAM Multilingual: Foundation Model for Underrepresented Languages}, 
      author={Andrei Kuzmenko and Alexandr Maximenko and Aleksandr Kutsakov and Georgii Gospodinov and Dmitrii Bolotov and Oleg Kutuzov and Pavel Bogomolov and Fyodor Minkin},
      year={2026},
      eprint={2607.10371},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2607.10371}
}

@inproceedings{kutsakov25_interspeech,
  title     = {{GigaAM: Efficient Self-Supervised Learner for Speech Recognition}},
  author    = {Aleksandr Kutsakov and Alexandr Maximenko and Georgii Gospodinov and Pavel Bogomolov and Fyodor Minkin},
  year      = {2025},
  booktitle = {{Interspeech 2025}},
  pages     = {1213--1217},
  doi       = {10.21437/Interspeech.2025-1616},
  issn      = {2958-1796},
}
```

## Links
* [[arxiv] GigaAM Multilingual: Foundation Model for Underrepresented Languages](https://arxiv.org/abs/2607.10371)
* [[arxiv] GigaAM: Efficient Self-Supervised Learner for Speech Recognition](https://arxiv.org/abs/2506.01192)
* [[habr] GigaAM-v4 и GigaAM Multilingual-v2: потоковое распознавание, 21 язык и новый уровень качества](https://habr.com/ru/companies/sberbank/articles/1112846/)
* [[habr] GigaAM-v3: открытая SOTA-модель распознавания речи на русском](https://habr.com/ru/companies/sberdevices/articles/973160/)
* [[habr] GigaAM: класс открытых моделей для обработки звучащей речи](https://habr.com/ru/companies/sberdevices/articles/805569)
* [[youtube] Как научить LLM слышать: GigaAM 🤝 GigaChat Audio](https://www.youtube.com/watch?v=O7NSH2SAwRc)
* [[youtube] GigaAM: Семейство акустических моделей для русского языка](https://youtu.be/PvZuTUnZa2Q?t=26442)
* [[youtube] Speech-only Pre-training: обучение универсального аудиоэнкодера](https://www.youtube.com/watch?v=ktO4Mx6UMNk)
