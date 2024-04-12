```
apt install python3-dev
apt install python3-venv
apt install ffmpeg libavcodec-extra
```

## virtual environment
```bash
python3.10 -m venv venv && . venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install Cython
pip install git+https://github.com/NVIDIA/NeMo.git@r1.21.0#egg=nemo_toolkit[all]
mkdir ./data
wget https://n-ws-q0bez.s3pd12.sbercloud.ru/b-ws-q0bez-jpv/GigaAM/{ssl_model_weights.ckpt,emo_model_weights.ckpt,ctc_model_weights.ckpt,ctc_model_config.yaml,emo_model_config.yaml,encoder_config.yaml,example.mp3} -P ./data

# GigaAM
python ssl_inference.py --encoder_config ./data/encoder_config.yaml \
    --model_weights ./data/ssl_model_weights.ckpt --device cuda --audio_path ./data/example.mp3

# encoded signal shape: torch.Size([1, 768, 159])

# GigaAM-CTC
python ctc_inference.py --model_config ./data/ctc_model_config.yaml \
    --model_weights ./data/ctc_model_weights.ckpt --device cuda --audio_path ./data/example.mp3

# transcription: а и правда никакой

# GigaAM-Emo
python emo_inference.py --model_config ./data/emo_model_config.yaml \
    --model_weights ./data/emo_model_weights.ckpt --device cuda --audio_path ./data/example.mp3

# angry: 0.019, sad: 0.137, neutral: 0.735, positive: 0.108
```

## Docker

```bash
docker build -t gigaam_image .

# GigaAM
docker run -v $PWD:/workspace/gigaam --gpus all gigaam_image \
    python /workspace/gigaam/ssl_inference.py --encoder_config /workspace/data/encoder_config.yaml \
    --model_weights /workspace/data/ssl_model_weights.ckpt \
    --device cuda --audio_path /workspace/data/example.mp3

# encoded signal shape: torch.Size([1, 768, 159])

# GigaAM-CTC
docker run -v $PWD:/workspace/gigaam --gpus all gigaam_image \
    python /workspace/gigaam/ctc_inference.py --model_config /workspace/data/ctc_model_config.yaml \
    --model_weights /workspace/data/ctc_model_weights.ckpt \
    --device cuda --audio_path /workspace/data/example.mp3

# transcription: а и правда никакой

# GigaAM-Emo
docker run -v $PWD:/workspace/gigaam --gpus all gigaam_image \
    python /workspace/gigaam/emo_inference.py --model_config /workspace/data/emo_model_config.yaml \
    --model_weights /workspace/data/emo_model_weights.ckpt \
    --device cuda --audio_path /workspace/data/example.mp3

# angry: 0.019, sad: 0.137, neutral: 0.735, positive: 0.108
```
