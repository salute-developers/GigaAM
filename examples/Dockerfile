FROM nvcr.io/nvidia/nemo:23.10

RUN mkdir -p /workspace/data
WORKDIR /workspace/data

RUN wget https://n-ws-q0bez.s3pd12.sbercloud.ru/b-ws-q0bez-jpv/GigaAM/{ssl_model_weights.ckpt,emo_model_weights.ckpt,ctc_model_weights.ckpt,rnnt_model_weights.ckpt,ctc_model_config.yaml,emo_model_config.yaml,encoder_config.yaml,rnnt_model_config.yaml,tokenizer_all_sets.tar,example.wav,long_example.wav}
RUN tar -xf tokenizer_all_sets.tar && rm tokenizer_all_sets.tar

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install Cython
RUN pip install git+https://github.com/NVIDIA/NeMo.git@1fa961ba03ab5f8c91b278640e29807079373372#egg=nemo_toolkit[all]
RUN pip install -U soundfile
RUN pip install pyannote.audio==3.2.0