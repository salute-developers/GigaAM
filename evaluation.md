## Evaluation Protocol

WER metrics follow the [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard) protocol, with language-specific normalization adjustments (Russian `ё`, Uzbek apostrophes, Azerbaijani `I`/`İ` casing) and substitutions between equivalent Russian and English words gathered primarily for the Golos dataset ([TSV](assets/word_substitutions.tsv)).

## Russian and English ASR

### Full-context ASR

#### Russian

Word Error Rate (%) on open-source and in-house test sets:

| Test set or group       | GigaAM Multilingual-v2 RNN-T (auto) | GigaAM-v4 RNN-T | GigaAM-v4 CTC | GigaAM-v3 RNN-T | GigaAM-v3 CTC | Parakeet Ultra | Nemotron 30B-A4B (auto) | Whisper large-v3 (auto) | Qwen3-ASR 1.7B (auto) |
| :---------------------- | ----------------------------------: | --------------: | ------------: | --------------: | ------------: | -------------: | ----------------------: | ----------------------: | --------------------: |
| Params                  |                                774M |            238M |          234M |            222M |          220M |           627M |                     34B |                    1.5B |                    2B |
| **Open-source data**    |                                     |                 |               |                 |               |                |                         |                         |                       |
| Golos Crowd             |                                4.03 |            4.56 |          5.05 |            7.43 |          8.12 |       **2.86** |                    4.36 |                   12.22 |                 13.14 |
| Golos Farfield          |                            **2.84** |            3.62 |          4.07 |            4.97 |          5.71 |           6.47 |                    8.25 |                   16.11 |                 20.22 |
| Russian LibriSpeech     |                            **3.22** |            3.69 |          4.06 |            5.77 |          5.93 |           9.19 |                    9.16 |                    8.61 |                 15.16 |
| Mozilla Common Voice 27 |                            **1.87** |            2.29 |          2.68 |            2.24 |           2.51 |           5.41 |                    4.81 |                    5.88 |                  7.55 |
| **In-house data**       |                                     |                 |               |                 |               |                |                         |                         |                       |
| Disordered Speech       |                           **11.99** |           12.88 |         12.78 |           18.68 |         16.55 |          32.13 |                   35.45 |                   33.69 |                 44.29 |
| Callcenter              |                            **5.96** |            6.71 |          7.20 |           11.04 |         11.92 |          20.77 |                   26.70 |                   25.96 |                 23.95 |
| Natural Speech          |                            **7.78** |            8.32 |          8.43 |           10.37 |         11.48 |          13.99 |                   14.26 |                   16.22 |                 17.53 |
| Synth IT Terms          |                                6.49 |        **5.07** |          5.61 |           15.27 |         15.23 |          12.57 |                   14.63 |                    9.63 |                 15.31 |
| Education               |                            **5.93** |            6.80 |          7.04 |            9.83 |          10.04 |          12.19 |                   12.71 |                   15.72 |                 15.65 |
| **Average**             |                            **5.57** |            5.99 |          6.33 |            9.51 |            9.72 |          12.84 |                   14.48 |                   16.01 |                 19.20 |

#### English

Seven public English Open ASR Leaderboard test sets. English was explicitly specified (`en`) where supported. GigaAM Multilingual-v2 was also evaluated with automatic language detection (`auto`).

| Model                  |            Params | Mode |  Average | AMI Cleaned | Earnings22 Cleaned | GigaSpeech Cleaned | SPGISpeech | URGENT24 | Voice Arena Monsoon | VoxPopuli Cleaned |
| :--------------------- | ----------------: | :--- | -------: | ----------: | -----------------: | -----------------: | ---------: | -------: | ------------------: | ----------------: |
| GigaAM Multilingual-v2 |              774M | auto |     7.35 |        7.16 |               7.93 |               8.42 |       1.94 |    16.89 |                3.78 |              5.32 |
| GigaAM Multilingual-v2 |              774M | en   |     6.93 |    **7.12** |               7.76 |               8.27 |       1.94 |    14.63 |                3.75 |              5.02 |
| GigaAM-v4 RNN-T        |              238M |      |     7.28 |        7.47 |               7.91 |               8.38 |       2.00 |    15.57 |                4.88 |              4.79 |
| GigaAM-v4 CTC          |              234M |      |     7.83 |       11.34 |               7.71 |               9.78 |       2.15 |    14.84 |                4.44 |              4.55 |
| Nemotron 30B-A4B       | 34B (4.3B active) | en   | **5.55** |        8.26 |           **4.83** |           **7.15** |       2.83 |    10.68 |            **3.39** |          **1.73** |
| Qwen3-ASR 1.7B         |                2B | en   |     5.75 |        8.31 |               5.96 |               7.22 |       2.58 | **9.69** |                3.47 |              3.02 |
| Parakeet Ultra         |              627M |      |     6.03 |        8.43 |               5.71 |               7.66 |       3.34 |    10.09 |                3.86 |              3.12 |
| Rumble-2-Flash         |              508M | en   |     6.09 |        9.15 |               6.44 |               7.56 |       3.16 |    10.28 |                3.55 |              2.49 |
| Qwen3-ASR 0.6B         |              938M | en   |     6.67 |        9.47 |               8.07 |               7.62 |       2.74 |    11.59 |                4.13 |              3.06 |
| Whisper large-v3       |              1.5B | en   |     7.70 |       13.66 |               7.81 |               8.37 |       2.69 |    13.65 |                3.95 |              3.79 |
| Parakeet TDT-CTC 1.1B  |              1.1B |      |     7.85 |       12.19 |               8.36 |               8.43 |       1.99 |    14.96 |                5.10 |              3.95 |
| Canary Flash 1B        |              883M |      |     7.96 |       11.76 |               6.99 |               8.65 |   **1.69** |    17.66 |                4.58 |              4.42 |
| Parakeet TDT-CTC 110M  |              114M |      |     8.26 |       12.73 |               9.23 |               8.86 |       2.25 |    15.34 |                5.66 |              3.74 |
| Omni LLM 7B v2         |              7.8B | en   |     8.67 |       12.50 |              11.45 |               9.10 |       3.34 |    15.21 |                4.96 |              4.12 |
| Canary Flash 180M      |              182M |      |    13.82 |       21.98 |               9.12 |              10.81 |       1.96 |    30.37 |               18.57 |              3.93 |

### Streaming ASR

#### Russian

Evaluated on nine Russian test sets: Golos Crowd, Golos Farfield, Russian LibriSpeech, Common Voice, Disordered Speech, Callcenter, Natural Speech, Synth IT Terms, and Education.

| Model             | Chunk (ms) | Average WER (%) ↓ |
| :---------------- | ---------: | ----------------: |
| GigaAM-v4 RNN-T   |        160 |              9.53 |
|                   |        320 |              8.52 |
|                   |        640 |              7.85 |
| T-one CTC+LM      |        300 |             17.93 |
| Nemotron ASR 0.6B |         80 |             28.27 |
|                   |        320 |             24.95 |
|                   |        560 |             23.72 |
|                   |       1120 |             22.41 |

#### English

Evaluated on the seven public English Open ASR Leaderboard test sets: AMI Cleaned, Earnings22 Cleaned, GigaSpeech Cleaned, SPGISpeech, URGENT24, Voice Arena Monsoon, and VoxPopuli Cleaned.

| Model                  | Mode              | Chunk (ms) | Average WER (%) ↓ |
| :--------------------- | :---------------- | ---------: | ----------------: |
| GigaAM-v4 RNN-T        | E2E               |        160 |              9.98 |
|                        | E2E               |        320 |              9.10 |
|                        | E2E               |        640 |              8.49 |
| GigaAM Multilingual-v2 | Auto language     |        160 |             10.41 |
|                        | Auto language     |        320 |              9.75 |
|                        | Auto language     |        640 |              9.60 |
| GigaAM Multilingual-v2 | English specified |        160 |              9.72 |
|                        | English specified |        320 |              9.05 |
|                        | English specified |        640 |              8.77 |
| Nemotron ASR 0.6B      | Auto language     |         80 |             18.84 |
|                        | Auto language     |        320 |             17.02 |
|                        | Auto language     |        560 |             16.55 |
|                        | Auto language     |       1120 |             16.14 |
| Nemotron ASR 0.6B      | English specified |         80 |             12.23 |
|                        | English specified |        320 |             10.57 |
|                        | English specified |        560 |              9.99 |
|                        | English specified |       1120 |              9.51 |

## Multilingual ASR

Word Error Rate (%) is reported on [Common Voice 27](https://commonvoice.mozilla.org), [FLEURS](https://huggingface.co/datasets/google/fleurs), and internal scripted speech datasets.

| Language                    | GigaAM Multilingual-v2 | Omni LLM 7B v2 | Whisper large-v3 | Rumble 2 Flash | Parakeet Ultra | Qwen3-ASR 1.7B | Nemotron 30B-A4B |
| :-------------------------- | ---------------------: | -------------: | ---------------: | -------------: | -------------: | -------------: | ---------------: |
| Params                      |                   774M |           7.8B |             1.5B |           508M |           627M |             2B |              34B |
| **High-resource languages** |                        |                |                  |                |                |                |                  |
| Russian                     |               **2.30** |           6.75 |             4.81 |           8.12 |           4.83 |           6.05 |             4.69 |
| English                     |                   5.42 |           7.86 |            11.04 |           6.23 |           5.62 |       **4.55** |             5.02 |
| German                      |               **3.16** |           4.92 |             4.42 |           5.49 |           3.93 |           4.16 |             3.33 |
| Spanish                     |                   3.06 |           3.73 |             3.55 |           3.91 |           3.21 |           3.49 |         **2.91** |
| French                      |                   5.36 |           7.13 |             7.80 |           7.97 |           5.70 |           6.04 |         **5.36** |
| Italian                     |               **2.28** |           3.38 |             3.68 |           3.80 |           2.65 |           3.54 |             2.48 |
| Dutch                       |               **2.72** |           4.63 |             4.54 |           5.00 |           4.69 |           5.64 |             3.91 |
| Portuguese                  |               **4.32** |           6.09 |             5.24 |           5.43 |           5.75 |           5.49 |             4.91 |
| **Low-resource languages**  |                        |                |                  |                |                |                |                  |
| Kazakh                      |               **3.95** |          14.20 |            35.82 |         107.72 |         101.27 |         105.95 |           125.01 |
| Kyrgyz                      |               **4.32** |          17.51 |            87.32 |         107.04 |         103.25 |         107.44 |           120.53 |
| Georgian                    |               **4.52** |          10.34 |            68.64 |         136.58 |         113.51 |         127.93 |           170.19 |
| Armenian                    |               **3.06** |           7.97 |            47.86 |         122.29 |         105.97 |         114.44 |           138.37 |
| Romanian                    |               **4.90** |          17.20 |            16.73 |          12.25 |          14.76 |          45.62 |            21.30 |
| Belarusian                  |               **2.50** |           7.25 |            74.02 |          15.58 |          81.70 |          85.09 |            84.82 |
| Polish                      |               **3.40** |           6.04 |             5.51 |           6.56 |           3.69 |          12.92 |             5.28 |
| Uzbek                       |               **3.70** |          21.82 |           104.79 |         106.88 |          97.57 |         107.61 |           109.00 |
| Tatar                       |               **4.31** |          50.72 |            96.66 |         107.15 |         103.49 |         106.94 |           120.17 |
| Tajik                       |               **4.98** |          29.82 |           111.82 |         106.69 |         102.65 |         106.55 |           114.76 |
| Azerbaijani                 |               **5.31** |          24.85 |            26.58 |          17.64 |         100.51 |          80.15 |            83.29 |
| Bashkir                     |               **2.74** |          12.83 |            96.49 |         108.88 |         101.80 |         107.49 |           121.11 |
| Turkmen                     |               **6.85** |          30.97 |           108.65 |         106.96 |         101.98 |         108.65 |           109.46 |
| **Average**                 |               **3.96** |      **14.10** |        **44.09** |      **52.77** |      **55.64** |      **59.80** |        **64.57** |

### Streaming

| Model                  | Mode               | Chunk (ms) | Average WER (%) ↓ |
| :--------------------- | :----------------- | ---------: | ----------------: |
| GigaAM Multilingual-v2 | Auto language      |        160 |              7.00 |
|                        | Auto language      |        320 |              6.21 |
|                        | Auto language      |        640 |              5.58 |
| GigaAM Multilingual-v2 | Language specified |        160 |              5.47 |
|                        | Language specified |        320 |              4.98 |
|                        | Language specified |        640 |              4.70 |
| Nemotron ASR 0.6B      | Auto language      |         80 |             61.90 |
|                        | Auto language      |        320 |             61.06 |
|                        | Auto language      |        560 |             60.53 |
|                        | Auto language      |       1120 |             59.72 |

## End-to-end ASR

### Punctuation recognition

Punctuation micro-F1 (%) for E2E outputs in auto mode against Common Voice, FLEURS, and internal scripted speech references, pooled across four punctuation marks and all test sets per language. Higher is better.

| Language    | GigaAM Multilingual-v2 | Whisper large-v3 | Qwen3-ASR 1.7B | Nemotron 30B-A4B | Rumble 2 Flash | Parakeet Ultra |
| :---------- | ---------------------: | ---------------: | -------------: | ---------------: | -------------: | -------------: |
| Russian     |              **90.93** |            87.79 |          87.32 |            88.38 |          87.02 |          89.45 |
| English     |              **89.55** |            79.07 |          87.62 |            87.77 |          88.06 |          89.17 |
| German      |              **94.82** |            92.07 |          93.81 |            94.07 |          92.96 |          94.29 |
| Spanish     |                  90.45 |            87.43 |          90.42 |            89.34 |          90.02 |      **91.29** |
| French      |                  88.33 |            83.80 |          86.82 |            86.59 |          85.71 |      **89.09** |
| Italian     |                  90.01 |            85.78 |          88.06 |            86.52 |          87.59 |      **90.03** |
| Dutch       |              **94.61** |            91.77 |          92.33 |            92.57 |          92.73 |          93.14 |
| Polish      |                  85.15 |            76.90 |          76.36 |            79.23 |          82.63 |      **91.90** |
| Portuguese  |                  69.62 |            77.29 |          79.08 |            73.61 |      **81.21** |          78.72 |
| Romanian    |              **92.19** |            76.33 |          63.66 |            76.34 |          81.43 |          82.26 |
| Azerbaijani |              **88.55** |            67.21 |          46.32 |            48.10 |          72.13 |          21.41 |
| Bashkir     |              **76.62** |            35.37 |          14.87 |            19.95 |          22.23 |          17.87 |
| Belarusian  |              **93.01** |            71.21 |          65.53 |            68.90 |          83.75 |          71.97 |
| Armenian    |              **93.24** |            74.97 |           5.56 |             9.12 |          10.36 |          10.76 |
| Georgian    |              **91.90** |            55.43 |          11.17 |            10.10 |          10.38 |          19.16 |
| Kyrgyz      |              **92.41** |            46.20 |          18.75 |            20.46 |          27.47 |          21.12 |
| Kazakh      |              **89.57** |            61.16 |          17.57 |            20.64 |          27.16 |          20.77 |
| Tajik       |              **89.83** |            15.31 |          14.37 |            15.88 |          17.86 |          17.21 |
| Turkmen     |              **84.89** |            30.79 |          15.44 |            26.41 |          30.19 |          20.99 |
| Tatar       |              **87.06** |            44.94 |          19.25 |            21.43 |          26.92 |          20.96 |
| Uzbek       |              **83.20** |            32.93 |          14.22 |            30.95 |          25.53 |          31.86 |
| **Average** |              **88.38** |        **65.42** |      **51.83** |        **54.59** |      **58.25** |      **55.40** |

| Scored punctuation | Comma | Full stop | Question mark | Exclamation mark |
| :----------------- | :---- | :-------- | :------------ | :--------------- |
| Armenian           | `,`   | `։`       | `՞`           | `՜`              |
| Other 20 languages | `,`   | `.`       | `?`           | `!`              |

## Voice Activity Detection

### ROC-AUC

Five public corpora and an internal manually annotated speech dataset, totaling 283.67 hours. Higher ROC-AUC is better. The best value in each row before rounding is in **bold**. FireRedVAD was evaluated in offline mode.

| Test set / split     | Hours | GigaAM VAD | pyannote segmentation-3.0 | Silero VAD 6.3 | FireRedVAD | TEN VAD |
| :------------------- | ----: | ---------: | ------------------------: | -------------: | ---------: | ------: |
| Parameters (M) ↓     |       |       1.33 |                      1.47 |           0.24 |       0.59 |    0.08 |
| AISHELL-4 test       |  12.7 |       0.97 |                      0.94 |           0.91 |   **0.97** |    0.92 |
| AliMeeting test far  |  10.8 |   **0.97** |                      0.94 |           0.95 |   **0.97** |    0.94 |
| AliMeeting test near |  32.3 |   **0.95** |                      0.88 |           0.94 |       0.94 |    0.93 |
| Earnings-21          |  39.3 |   **0.98** |                      0.97 |           0.96 |       0.97 |    0.95 |
| MSDWild few.val      |   9.9 |   **0.95** |                      0.88 |           0.92 |       0.92 |    0.89 |
| MSDWild many.val     |   4.1 |   **0.93** |                      0.87 |           0.89 |       0.89 |    0.86 |
| VoxConverse test     |  43.5 |   **0.96** |                      0.92 |           0.95 |       0.92 |    0.93 |
| GigaAM VAD test      | 131.1 |   **0.98** |                      0.97 |           0.97 |       0.97 |    0.96 |

### Precision and background rejection at 95% recall

Precision was measured on the six speech datasets. ESC-50 accuracy is the percentage of 2,000 background clips with no false speech detection. At least 100 ms of consecutive detected speech counts as a false positive for the entire clip. Higher precision and accuracy are better.

| Model                     | Recall (%) | Precision (%) ↑ | ESC-50 accuracy (%) ↑ |
| :------------------------ | ---------: | --------------: | --------------------: |
| GigaAM VAD                |      95.00 |           90.62 |             **84.90** |
| pyannote segmentation-3.0 |      95.00 |       **91.64** |                 75.80 |
| Silero VAD 6.3            |      95.00 |           86.46 |                 53.65 |
| FireRedVAD                |      95.00 |           84.91 |                 45.15 |
| TEN VAD                   |      95.00 |           84.24 |                  9.60 |

### ASR on long recordings

The same GigaAM-v4 RNN-T RAW model was evaluated on 2,544 recordings totaling 89.41 hours with two segmentation pipelines. WER (%) is computed from pooled word and error counts. Lower is better. Both the VAD model and the segmentation rules differed between the two pipelines.

| ASR model       | pyannote (previous segmentation) | GigaAM VAD (new segmentation) |
| :-------------- | -------------------------------: | ----------------------------: |
| GigaAM-v4 RNN-T |                             10.8 |                      **10.1** |

## Emotion recognition

GigaAM-Emo was trained on the [Dusha](https://arxiv.org/pdf/2212.12266.pdf) dataset

|                                                                                                                                                                         |                     | Crowd             |                |                     | Podcast           |                |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ----------------- | -------------- | ------------------- | ----------------- | -------------- |
|                                                                                                                                                                         | Unweighted Accuracy | Weighted Accuracy | Macro F1-score | Unweighted Accuracy | Weighted Accuracy | Macro F1-score |
| [DUSHA](https://arxiv.org/pdf/2212.12266.pdf) baseline <br/> ([MobileNetV2](https://arxiv.org/abs/1801.04381) + [Self-Attention](https://arxiv.org/pdf/1805.08318.pdf)) | 0.83                | 0.76              | 0.77           | 0.89                | 0.53              | 0.54           |
| [АБК](https://aij.ru/archive?albumId=2&videoId=337) ([TIM-Net](https://arxiv.org/pdf/2211.08233.pdf))                                                                   | 0.84                | 0.77              | 0.78           | **0.90**            | 0.50              | 0.55           |
| GigaAM-Emo                                                                                                                                                              | **0.90**            | **0.87**          | **0.84**       | **0.90**            | **0.76**          | **0.67**       |
