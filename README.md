# GigaAM: the family of open-source acoustic models for speech processing

![plot](./gigaam_scheme.svg)

## Latest News
* 2024/12 — [MIT License](./LICENSE), GigaAM-v2 (**-15%** and **-12%** WER Reduction for CTC and RNN-T models, respectively), [ONNX export support](#onnx-inference-example)
* 2024/05 — GigaAM-RNNT (**-19%** WER Reduction), [long-form inference using external Voice Activity Detection](#long-form-audio-transcribation)
* 2024/04 — GigaAM Release: GigaAM-CTC ([SoTA Speech Recognition model for the Russian language](#performance-metrics-word-error-rate)), [GigaAM-Emo](#gigaam-emo-emotion-recognition)
---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [GigaAM: The Foundational Model](#gigaam-the-foundational-model)
- [GigaAM for Speech Recognition](#gigaam-for-speech-recognition)
  - [GigaAM-CTC](#gigaam-ctc)
  - [GigaAM-RNNT](#gigaam-rnnt)
- [GigaAM-Emo: Emotion Recognition](#gigaam-emo-emotion-recognition)
- [License](#license)
- [Links](#links)

---

## Overview

GigaAM (**Giga** **A**coustic **M**odel) is a family of open-source models for Russian speech processing tasks, including speech recognition and emotion recognition. The models are built on top of the [Conformer](https://arxiv.org/pdf/2005.08100.pdf) architecture and leverage self-supervised learning ([wav2vec2](https://arxiv.org/abs/2006.11477)-based for GigaAM-v1 and [HuBERT](https://arxiv.org/pdf/2106.07447)-based for GigaAM-v2).

GigaAM models are state-of-the-art open-source solutions for their respective tasks in the Russian language.

This repository includes:

- **GigaAM**: A foundational self-supervised model pre-trained on massive Russian speech datasets.
- **GigaAM-CTC** and **GigaAM-RNNT**: Fine-tuned models for automatic speech recognition (ASR).
- **GigaAM-Emo**: A fine-tuned model for emotion recognition.

## Installation

### Requirements
- Python ≥ 3.8
- [ffmpeg](https://ffmpeg.org/) installed and added to your system's PATH

### Install the GigaAM Package

1. Clone the repository:
   ```bash
   git clone https://github.com/salute-developers/GigaAM.git
   cd GigaAM
   ```

2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

3. Verify the installation:
   ```python
   import gigaam
   model = gigaam.load_model("ctc")
   print(model)
   ```

---

## GigaAM: The Foundational Model

GigaAM is a [Conformer](https://arxiv.org/pdf/2005.08100.pdf)-based foundational model (240M parameters) pre-trained on 50,000+ hours of diverse Russian speech data. 

It serves as the backbone for the entire GigaAM family, enabling state-of-the-art fine-tuned performance in speech recognition and emotion recognition.

There are 2 available versions:

* GigaAM-v1 was trained with a [wav2vec2](https://arxiv.org/abs/2006.11477)-like approach and can be used by loading the `v1_ssl` model version.
* GigaAM-v2 was trained with a [HuBERT](https://arxiv.org/pdf/2106.07447)-like approach and allows us to get GigaAM-v2 ASR model with better quality. It can be used by loading the `v2_ssl` or `ssl` model version.

More information about GigaAM-v1 can be found in our [post on Habr](https://habr.com/ru/companies/sberdevices/articles/805569).

### GigaAM Usage Example

```python
import gigaam
model = gigaam.load_model('ssl') # Options: "ssl", "v1_ssl"
embedding, _ = model.embed_audio(audio_path)
```

---

## GigaAM for Speech Recognition

We fine-tuned the GigaAM encoder for ASR using two different architectures:

- GigaAM-CTC was fine-tuned with [Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) and a character-based tokenizer.
- GigaAM-RNNT was fine-tuned with [RNN Transducer loss](https://arxiv.org/abs/1211.3711).

Fine-tuning was done for both GigaAM-v1 and GigaAM-v2 SSL models, so we have 4 ASR models: `v1` and `v2` versions for both CTC and RNNT.

### Training Data
The models were trained on publicly available Russian datasets:

| Dataset                | Size (hours) | Weight |
|------------------------|--------------|--------|
| Golos                 | 1227         | 0.6    |
| SOVA                  | 369          | 0.2    |
| Russian Common Voice  | 207          | 0.1    |
| Russian LibriSpeech   | 93           | 0.1    |

### Performance Metrics (Word Error Rate)
| Model              | Parameters | Golos Crowd | Golos Farfield | OpenSTT YouTube | OpenSTT Phone Calls | OpenSTT Audiobooks | Mozilla Common Voice 12 | Mozilla Common Voice 19 | Russian LibriSpeech |
|--------------------|------------|-------------|----------------|-----------------|----------------------|--------------------|-------|-------|---------------------|
| Whisper-large-v3   | 1.5B       | 13.9        | 16.6           | 18.0            | 28.0                 | 14.4               | 5.7   | 5.5   | 9.5                 |
| NVIDIA FastConformer | 115M       | 2.2         | 6.6            | 21.2            | 30.0                 | 13.9               | 2.7   | 5.7   | 11.3                |
| **GigaAM-CTC-v1**  | 242M       | 3.0         | 5.7            | 16.0            | 23.2                 | 12.5               | 2.0   | 10.5  | 7.5                 |
| **GigaAM-RNNT-v1** | 243M       | 2.3         | 5.0            | 14.0            | 21.7                 | 11.7               | 1.9   | 9.9   | 7.7                 |
| **GigaAM-CTC-v2**  | 242M       | 2.5         | 4.3            | 14.1            | 21.1                 | 10.7               | 2.1   | 3.1   | 5.5                 |
| **GigaAM-RNNT-v2** | 243M       | **<span style="color:green">2.2</span>**         | **<span style="color:green">3.9</span>**            | **<span style="color:green">13.3</span>**            | **<span style="color:green">20.0</span>**                | **<span style="color:green">10.2</span>**               | **<span style="color:green">1.8</span>**   | **<span style="color:green">2.7</span>**   | **<span style="color:green">5.5</span>**               |


### Speech Recognition Example (GigaAM-ASR)

   #### Basic usage: short audio transcribation (up to 30 seconds)

   ```python
   import gigaam
   model_name = "rnnt"  # Options: "v2_ctc" or "ctc", "v2_rnnt" or "rnnt", "v1_ctc", "v1_rnnt"
   model = gigaam.load_model(model_name)
   transcription = model.transcribe(audio_path)
   ```

   #### Long-form audio transcribation
   1. Install external VAD dependencies ([pyannote.audio](https://github.com/pyannote/pyannote-audio) library) with 
      ```bash
      pip install gigaam[longform]
      ```
   2. 
      * Generate [Hugging Face API token](https://huggingface.co/docs/hub/security-tokens)
      * Accept the conditions to access [pyannote/voice-activity-detection](https://huggingface.co/pyannote/voice-activity-detection) files and content.
      * Accept the conditions to access [pyannote/segmentation](https://huggingface.co/pyannote/segmentation) files and content.
   3. Use the `model.transcribe_longform` method:
      ```python
      import os
      import gigaam

      os.environ["HF_TOKEN"] = "<HF_TOKEN>"

      model = gigaam.load_model("ctc")
      recognition_result = model.transcribe_longform("long_example.wav")

      for utterance in recognition_result:
         transcription = utterance["transcription"]
         start, end = utterance["boundaries"]
         print(f"[{gigaam.format_time(start)} - {gigaam.format_time(end)}]: {transcription}")
      ```   

   #### ONNX inference example

   1. Export the model to ONNX using the `model.to_onnx` method:
      ```python
      onnx_dir = "onnx"
      model_type = "rnnt" # or "ctc"

      model = gigaam.load_model(
         model_type,
         fp16_encoder=False,  # only fp32 tensors
         use_flash=False,  # disable flash attention
      )
      model.to_onnx(dir_path=onnx_dir)
      ```
   2. Run ONNX inference:
      ```python
      from gigaam.onnx_utils import load_onnx_sessions, transcribe_sample

      sessions = load_onnx_sessions(onnx_dir, model_type)
      transcribe_sample("example.wav", model_type, sessions)
      ```


All these examples can also be found in [inference_example.ipynb](./inference_example.ipynb) notebook.

---


## GigaAM-Emo: Emotion Recognition

GigaAM-Emo is a fine-tuned model for emotion recognition trained on the [Dusha](https://arxiv.org/pdf/2212.12266.pdf) dataset. It significantly outperforms existing models on several metrics.

### Performance Metrics
|  |  | Crowd |  |  | Podcast |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Unweighted Accuracy | Weighted Accuracy | Macro F1-score | Unweighted Accuracy | Weighted Accuracy | Macro F1-score |
| [DUSHA](https://arxiv.org/pdf/2212.12266.pdf) baseline <br/> ([MobileNetV2](https://arxiv.org/abs/1801.04381) + [Self-Attention](https://arxiv.org/pdf/1805.08318.pdf)) | 0.83 | 0.76 | 0.77 | 0.89 | 0.53 | 0.54 |
| [АБК](https://aij.ru/archive?albumId=2&videoId=337) ([TIM-Net](https://arxiv.org/pdf/2211.08233.pdf)) | 0.84 | 0.77 | 0.78 | <span style="color:green">0.90</span> | 0.50 | 0.55 |
| GigaAM-Emo | <span style="color:green">0.90</span> | <span style="color:green">0.87</span> | <span style="color:green">0.84</span> | <span style="color:green">0.90</span> | <span style="color:green">0.76</span> | <span style="color:green">0.67</span> |

### Emotion Recognition Example (GigaAM-Emo)

```python
import gigaam
model = gigaam.load_model('emo')
emotion2prob: Dict[str, int] = model.get_probs("example.wav")

print(", ".join([f"{emotion}: {prob:.3f}" for emotion, prob in emotion2prob.items()]))
```

---

## License

GigaAM's code and model weights are released under the [MIT License](./LICENSE).

---

## Links
* [[habr] GigaAM: класс открытых моделей для обработки звучащей речи](https://habr.com/ru/companies/sberdevices/articles/805569)
* [[youtube] Как научить LLM слышать: GigaAM 🤝 GigaChat Audio](https://www.youtube.com/watch?v=O7NSH2SAwRc)
* [[youtube] GigaAM: Семейство акустических моделей для русского языка](https://youtu.be/PvZuTUnZa2Q?t=26442)
* [[youtube] Speech-only Pre-training: обучение универсального аудиоэнкодера](https://www.youtube.com/watch?v=ktO4Mx6UMNk)
