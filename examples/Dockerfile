FROM nvcr.io/nvidia/nemo:22.08

RUN mkdir -p /workspace/data
WORKDIR /workspace/data

RUN wget https://n-ws-q0bez.s3pd12.sbercloud.ru/b-ws-q0bez-jpv/GigaAM/{ssl_model_weights.ckpt,emo_model_weights.ckpt,ctc_model_weights.ckpt,ctc_model_config.yaml,emo_model_config.yaml,encoder_config.yaml,example.mp3}

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install Cython
RUN pip install git+https://github.com/NVIDIA/NeMo.git@r1.21.0#egg=nemo_toolkit[all]
RUN pip install -U soundfile