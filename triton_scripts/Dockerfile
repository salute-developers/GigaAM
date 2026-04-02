FROM nvcr.io/nvidia/tritonserver:24.10-py3

RUN pip install --no-cache-dir \
    "torch>=2.6,<2.11" \
    "torchaudio>=2.6,<2.11" \
    sentencepiece \
    omegaconf \
    onnxruntime-gpu \
    tqdm \
    hydra-core
