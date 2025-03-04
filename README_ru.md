# GigaAM: семейство акустических моделей для обработки звучащей речи

![plot](./gigaam_scheme.svg)

## Последние Обновления
* 2024/12 — [MIT-лицензия](./LICENSE), GigaAM-v2 (улучшение качества распознавания речи на **15%** и **12%** WER для CTC и RNN-T моделей), [поддержка экспорта в ONNX](#пример-распознавания-с-помощью-onnx)
* 2024/05 — GigaAM-RNNT (улучшение WER на **-19%**), [распознавание речи на длинных аудиозаписях с помощью внешей VAD-модели](#распознавание-речи-на-длинных-аудиозаписях)
* 2024/04 — Релиз GigaAM: GigaAM-CTC ([Лучшая открытая модель для распознавания речи на русском языке](#метрики-качества-word-error-rate)), [GigaAM-Emo](#gigaam-emo)
---

## Содержание

- [Обзор](#обзор)
- [Установка](#установка)
- [GigaAM](#gigaam)
- [GigaAM для распознавания речи](#gigaam-для-распознавания-речи)
  - [GigaAM-CTC](#gigaam-ctc)
  - [GigaAM-RNNT](#gigaam-rnnt)
- [GigaAM-Emo](#gigaam-emo)
- [Лицензионное соглашение](#лицензия)
- [Ссылки](#ссылки)

---

## Обзор

GigaAM (**Giga** **A**coustic **M**odel) - семейство акустических моделей для обработки звучащей речи на русском языке. Среди решаемых задач - задачи распознавания речи, распознавания эмоций и извлечения эмбеддингов из аудио. Модели построены на основе архитектуры [Conformer](https://arxiv.org/pdf/2005.08100.pdf) с использованием методов self-supervised learning ([wav2vec2](https://arxiv.org/abs/2006.11477)-подход для GigaAM-v1 и [HuBERT](https://arxiv.org/pdf/2106.07447)-подход для GigaAM-v2).

Модели GigaAM с отрывом являются лучшими по качеству моделями в открытом доступе для соответствующих задач.

Репозиторий включает:
- **GigaAM**: фундаментальная акустическая модель, обученная на большом объеме неразмеченных русскоязычных аудиозаписей.
- **GigaAM-CTC и GigaAM-RNNT**: модели, дообученные на задачу автоматического распознавания речи.
- **GigaAM-Emo**: модель, дообученная на задачу распознавания эмоций. 

## Установка

### Требования
- Python ≥ 3.8
- установленный и добавленный в PATH [ffmpeg](https://ffmpeg.org/)

### Установка пакета GigaAM

1. Скачивание репозитория:
  ```bash
   git clone https://github.com/salute-developers/GigaAM.git
   cd GigaAM
   ```
2. Установка пакета:
  ```bash
   pip install -e .
   ```

3. Проверка установленного пакета:
  ```python
   import gigaam
   model = gigaam.load_model("ctc")
   print(model)
   ```

## GigaAM

GigaAM (**Giga** **A**coustic **M**odel) — фундаментальная акустическая модель, основанная на [Conformer](https://arxiv.org/pdf/2005.08100.pdf)-энкодере (около 240M параметров) и обученная на 50 тысячах часов разнообразных русскоязычных данных. 

Доступны 2 версии модели, отличающиеся алгоритмом предобучения:
* GigaAM-v1 была обучена на основе подхода [wav2vec2](https://arxiv.org/abs/2006.11477). Версия модели для использования - `v1_ssl`.
* GigaAM-v2 была обучена на основе подхода [HuBERT](https://arxiv.org/pdf/2106.07447) и позволила улучшить качество распознавания речи. Версия модели для использования - `v2_ssl` или `ssl`.

Больше информации про обучение GigaAM-v1 можно найти в [нашей статье на Хабре](https://habr.com/ru/companies/sberdevices/articles/805569).

### Пример использования GigaAM

```python
import gigaam
model = gigaam.load_model('ssl') # Options: "ssl", "v1_ssl"
embedding, _ = model.embed_audio(audio_path)
```

---


## GigaAM для распознавания речи (GigaAM-ASR)
Мы дообучали GigaAM энкодер для задачи распознавания речи с двумя разными декодерами:
* Модели GigaAM-CTC были дообучены с [CTC](https://www.cs.toronto.edu/~graves/icml_2006.pdf) функцией потерь.
* Модели GigaAM-RNNT была дообучена с [RNN-T](https://arxiv.org/abs/1211.3711) функцией потерь.

Мы проводили дообучения моделей с обеих версий GigaAM: `v1` и `v2`, таким образом для каждой из GigaAM-CTC и GigaAM-RNNT моделей доступны 2 версии: `v1` и `v2`.

### Данные для обучения

| dataset | size, hours | weight |
| --- | --- | --- |
| [Golos](https://arxiv.org/pdf/2106.10161.pdf) | 1227 | 0.6 |
| [SOVA](https://github.com/sovaai/sova-dataset) | 369 | 0.2 |
| [Russian Common Voice](https://arxiv.org/pdf/1912.06670.pdf) | 207 | 0.1 |
| [Russian LibriSpeech](https://arxiv.org/pdf/2012.03411.pdf) | 93 | 0.1 |

### Метрики качества (Word Error Rate)

| Model              | Parameters | Golos Crowd | Golos Farfield | OpenSTT YouTube | OpenSTT Phone Calls | OpenSTT Audiobooks | Mozilla Common Voice 12 | Mozilla Common Voice 19 | Russian LibriSpeech |
|--------------------|------------|-------------|----------------|-----------------|----------------------|--------------------|-------|-------|---------------------|
| Whisper-large-v3   | 1.5B       | 13.9        | 16.6           | 18.0            | 28.0                 | 14.4               | 5.7   | 5.5   | 9.5                 |
| NVIDIA FastConformer | 115M       | 2.2         | 6.6            | 21.2            | 30.0                 | 13.9               | 2.7   | 5.7   | 11.3                |
| **GigaAM-CTC-v1**  | 242M       | 3.0         | 5.7            | 16.0            | 23.2                 | 12.5               | 2.0   | 10.5  | 7.5                 |
| **GigaAM-RNNT-v1** | 243M       | 2.3         | 5.0            | 14.0            | 21.7                 | 11.7               | 1.9   | 9.9   | 7.7                 |
| **GigaAM-CTC-v2**  | 242M       | 2.5         | 4.3            | 14.1            | 21.1                 | 10.7               | 2.1   | 3.1   | 5.5                 |
| **GigaAM-RNNT-v2** | 243M       | **<span style="color:green">2.2</span>**         | **<span style="color:green">3.9</span>**            | **<span style="color:green">13.3</span>**            | **<span style="color:green">20.0</span>**                | **<span style="color:green">10.2</span>**               | **<span style="color:green">1.8</span>**   | **<span style="color:green">2.7</span>**   | **<span style="color:green">5.5</span>**               |

### Использование моделей распознавания речи (GigaAM-ASR)

  #### Базовое использование - распознование речи на коротких аудиозаписях (до 30 секунд)
  ```python
   import gigaam
   model_name = "rnnt"  # Options: "v2_ctc" or "ctc", "v2_rnnt" or "rnnt", "v1_ctc", "v1_rnnt"
   model = gigaam.load_model(model_name)
   transcription = model.transcribe(audio_path)
   ```

  #### Распознавание речи на длинных аудиозаписях
  1. Установите зависимости для внешней VAD-модели ([pyannote.audio](https://github.com/pyannote/pyannote-audio) library):
      ```bash
      pip install gigaam[longform]
      ```
  2. 
      * Сгенерируйте [Hugging Face API token](https://huggingface.co/docs/hub/security-tokens)
      * Примите условия для получения доступа к контенту [pyannote/voice-activity-detection](https://huggingface.co/pyannote/voice-activity-detection)
      * Примите условия для получения доступа к контенту [pyannote/segmentation](https://huggingface.co/pyannote/segmentation)
  
  3. Используйте метод ```model.transcribe_longform```:
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

  #### Пример распознавания с помощью ONNX

  1. Экспортируйте модель в onnx-формат с помощью метода ```model.to_onnx```:
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
   2. Используйте полученную модель для транскрибации:
      ```python
      from gigaam.onnx_utils import load_onnx_sessions, transcribe_sample

      sessions = load_onnx_sessions(onnx_dir, model_type)
      transcribe_sample("example.wav", model_type, sessions)
      ```

Все приведенные примеры также могут быть найдены в jupyter-ноутбуке [inference_example.ipynb](./inference_example.ipynb).

---


## GigaAM-Emo

GigaAM-Emo — акустическая модель для определения эмоций. Мы доучивали GigaAM на датасете [Dusha](https://arxiv.org/pdf/2212.12266.pdf).

В таблице ниже приведены метрики качества открытых моделей на датасете [Dusha](https://arxiv.org/pdf/2212.12266.pdf):

|  |  | Crowd |  |  | Podcast |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Unweighted Accuracy | Weighted Accuracy | Macro F1-score | Unweighted Accuracy | Weighted Accuracy | Macro F1-score |
| [DUSHA](https://arxiv.org/pdf/2212.12266.pdf) baseline <br/> ([MobileNetV2](https://arxiv.org/abs/1801.04381) + [Self-Attention](https://arxiv.org/pdf/1805.08318.pdf)) | 0.83 | 0.76 | 0.77 | 0.89 | 0.53 | 0.54 |
| [АБК](https://aij.ru/archive?albumId=2&videoId=337) ([TIM-Net](https://arxiv.org/pdf/2211.08233.pdf)) | 0.84 | 0.77 | 0.78 | <span style="color:green">0.90</span> | 0.50 | 0.55 |
| GigaAM-Emo | <span style="color:green">0.90</span> | <span style="color:green">0.87</span> | <span style="color:green">0.84</span> | <span style="color:green">0.90</span> | <span style="color:green">0.76</span> | <span style="color:green">0.67</span> |

### Пример использвания GigaAM-Emo для распознавания эмоций

```python
import gigaam
model = gigaam.load_model('emo')
emotion2prob: Dict[str, int] = model.get_probs("example.wav")

print(", ".join([f"{emotion}: {prob:.3f}" for emotion, prob in emotion2prob.items()]))
```

---

## Лицензия

Код и веса моделей семества GigaAM доступны для использования с [MIT-лицензией](./LICENSE).

## Ссылки
* [[habr] GigaAM: класс открытых моделей для обработки звучащей речи](https://habr.com/ru/companies/sberdevices/articles/805569)
* [[youtube] Как научить LLM слышать: GigaAM 🤝 GigaChat Audio](https://www.youtube.com/watch?v=O7NSH2SAwRc)
* [[youtube] GigaAM: Семейство акустических моделей для русского языка](https://youtu.be/PvZuTUnZa2Q?t=26442)
* [[youtube] Speech-only Pre-training: обучение универсального аудиоэнкодера](https://www.youtube.com/watch?v=ktO4Mx6UMNk)
