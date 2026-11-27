FROM nvcr.io/nvidia/tritonserver:24.10-py3

RUN pip install --no-cache-dir \
    "torch>=2.6" \
    "torchaudio>=2.6" \
    "numpy==1.26.*" \
    hydra-core==1.3.* \
    soundfile==0.13.* \
    omegaconf==2.3.* \
    sentencepiece \
    onnx \
    onnxruntime-gpu \
    tensorrt-cu12-bindings==10.5.0 \
    tqdm
