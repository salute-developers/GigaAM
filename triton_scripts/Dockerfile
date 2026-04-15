FROM nvcr.io/nvidia/tritonserver:24.10-py3

RUN pip install --no-cache-dir \
    "torch>=2.6,<2.11" \
    "torchaudio>=2.6,<2.11" \
    hydra-core==1.3.* \
    soundfile==0.13.* \
    omegaconf==2.3.* \
    sentencepiece \
    onnxruntime-gpu \
    tqdm
