* [Virtual environment](#virtual-environment)
* [Docker](#docker)
* For long-form inference:
  * generate [Hugging Face API token](https://huggingface.co/docs/hub/security-tokens)
  * accept the conditions to access [pyannote/voice-activity-detection](https://huggingface.co/pyannote/voice-activity-detection) files and content
  * accept the conditions to access [pyannote/segmentation](https://huggingface.co/pyannote/segmentation) files and content


## Virtual environment
```bash 
apt install python3-dev
apt install python3-venv
apt install ffmpeg libavcodec-extra
```

```bash
python3.10 -m venv venv && . venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install Cython
pip install -U wheel
pip install git+https://github.com/NVIDIA/NeMo.git@1fa961ba03ab5f8c91b278640e29807079373372#egg=nemo_toolkit[all]
pip install pyannote.audio==3.2.0
mkdir ./data
wget https://n-ws-q0bez.s3pd12.sbercloud.ru/b-ws-q0bez-jpv/GigaAM/{ssl_model_weights.ckpt,emo_model_weights.ckpt,ctc_model_weights.ckpt,rnnt_model_weights.ckpt,ctc_model_config.yaml,emo_model_config.yaml,encoder_config.yaml,rnnt_model_config.yaml,tokenizer_all_sets.tar,example.wav,long_example.wav} -P ./data && tar -xf ./data/tokenizer_all_sets.tar --directory ./data/ && rm ./data/tokenizer_all_sets.tar

# GigaAM
python ssl_inference.py --encoder_config ./data/encoder_config.yaml \
    --model_weights ./data/ssl_model_weights.ckpt --device cuda --audio_path ./data/example.wav

# encoded signal shape: torch.Size([1, 768, 283])

# GigaAM-CTC
python ctc_inference.py --model_config ./data/ctc_model_config.yaml \
    --model_weights ./data/ctc_model_weights.ckpt --device cuda --audio_path ./data/example.wav

# transcription: ничьих не требуя похвал счастлив уж я надеждой сладкой что дева с трепетом любви посмотрит может быть украдкой на песни грешные мои у лукоморья дуб зеленый

# GigaAM-CTC long-form
python ctc_longform_inference.py --model_config ./data/ctc_model_config.yaml \
    --model_weights ./data/ctc_model_weights.ckpt --device cuda \
    --audio_path ./data/long_example.wav --hf_token <YOUR_HF_TOKEN>

# [00:00:00 - 00:16:83]: вечерня отошла давно но в кельях тихо и темно уже и сам эгумин строгий свои молитвы прекратил и кости ветхие склонил перекрестясь на одр убогий кругом и сон и тишина но церкви дверь отворена
# [00:17:10 - 00:32:61]: трепещет луч лампады и тускло озаряет он и темную живопись икон и возлощенные оклады и раздается в тишине то тяжкий вздох то шепот важный и мрачно дремлет в вашине старинный свод
# ...

# GigaAM-RNNT
python rnnt_inference.py --model_config ./data/rnnt_model_config.yaml \
    --model_weights ./data/rnnt_model_weights.ckpt --tokenizer_path ./data/tokenizer_all_sets \
    --device cuda --audio_path ./data/example.wav

# transcription: ничьих не требуя похвал счастлив уж я надеждой сладкой что дева с трепетом любви посмотрит может быть украдкой на песни грешные мои у лукоморья дуб зеленый

# GigaAM-RNNT long-form
python rnnt_longform_inference.py --model_config ./data/rnnt_model_config.yaml \
    --model_weights ./data/rnnt_model_weights.ckpt --tokenizer_path ./data/tokenizer_all_sets \
    --device cuda --audio_path ./data/long_example.wav --hf_token <YOUR_HF_TOKEN>

# [00:00:00 - 00:16:83]: вечерня отошла давно но в кельях тихо и темно уже и сам игумин строгий свои молитвы прекратил и кости ветхие склонил перекрестясь на одр убогий кругом и сон и тишина но церкви дверь отворена
# [00:17:10 - 00:32:61]: трепещет луч лампады и тускло озаряет он и темну живопись икон и возлащенные оклады и раздается в тишине то тяжкий вздох то шепот важный и мрачно дремлет в вышине старинный свод
# ...

# GigaAM-Emo
python emo_inference.py --model_config ./data/emo_model_config.yaml \
    --model_weights ./data/emo_model_weights.ckpt --device cuda --audio_path ./data/example.wav

# angry: 0.000, sad: 0.002, neutral: 0.923, positive: 0.074
```

## Docker

```bash
docker build -t gigaam_image .

# GigaAM
docker run -v $PWD:/workspace/gigaam --gpus all gigaam_image \
    python /workspace/gigaam/ssl_inference.py --encoder_config /workspace/data/encoder_config.yaml \
    --model_weights /workspace/data/ssl_model_weights.ckpt \
    --device cuda --audio_path /workspace/data/example.wav

# encoded signal shape: torch.Size([1, 768, 283])

# GigaAM-CTC
docker run -v $PWD:/workspace/gigaam --gpus all gigaam_image \
    python /workspace/gigaam/ctc_inference.py --model_config /workspace/data/ctc_model_config.yaml \
    --model_weights /workspace/data/ctc_model_weights.ckpt \
    --device cuda --audio_path /workspace/data/example.wav

# transcription: ничьих не требуя похвал счастлив уж я надеждой сладкой что дева с трепетом любви посмотрит может быть украдкой на песни грешные мои у лукоморья дуб зеленый

# GigaAM-CTC longform
docker run -v $PWD:/workspace/gigaam --gpus all gigaam_image \
    python /workspace/gigaam/ctc_longform_inference.py --model_config /workspace/data/ctc_model_config.yaml \
    --model_weights /workspace/data/ctc_model_weights.ckpt --device cuda \
    --audio_path /workspace/data/long_example.wav --hf_token <YOUR_HF_TOKEN>

# [00:00:00 - 00:16:83]: вечерня отошла давно но в кельях тихо и темно уже и сам эгумин строгий свои молитвы прекратил и кости ветхие склонил перекрестясь на одр убогий кругом и сон и тишина но церкви дверь отворена
# [00:17:10 - 00:32:61]: трепещет луч лампады и тускло озаряет он и темную живопись икон и возлощенные оклады и раздается в тишине то тяжкий вздох то шепот важный и мрачно дремлет в вашине старинный свод
# ...

# GigaAM-RNNT
docker run -v $PWD:/workspace/gigaam --gpus all gigaam_image \
    python /workspace/gigaam/rnnt_inference.py --model_config /workspace/data/rnnt_model_config.yaml \
    --model_weights /workspace/data/rnnt_model_weights.ckpt --tokenizer_path /workspace/data/tokenizer_all_sets \
    --device cuda --audio_path /workspace/data/example.wav

# transcription: ничьих не требуя похвал счастлив уж я надеждой сладкой что дева с трепетом любви посмотрит может быть украдкой на песни грешные мои у лукоморья дуб зеленый

# GigaAM-RNNT longform
docker run -v $PWD:/workspace/gigaam --gpus all gigaam_image \
    python /workspace/gigaam/rnnt_longform_inference.py --model_config /workspace/data/rnnt_model_config.yaml \
    --model_weights /workspace/data/rnnt_model_weights.ckpt --tokenizer_path /workspace/data/tokenizer_all_sets \
    --device cuda --audio_path /workspace/data/long_example.wav --hf_token <YOUR_HF_TOKEN>

# [00:00:00 - 00:16:83]: вечерня отошла давно но в кельях тихо и темно уже и сам игумин строгий свои молитвы прекратил и кости ветхие склонил перекрестясь на одр убогий кругом и сон и тишина но церкви дверь отворена
# [00:17:10 - 00:32:61]: трепещет луч лампады и тускло озаряет он и темну живопись икон и возлащенные оклады и раздается в тишине то тяжкий вздох то шепот важный и мрачно дремлет в вышине старинный свод
# ...

# GigaAM-Emo
docker run -v $PWD:/workspace/gigaam --gpus all gigaam_image \
    python /workspace/gigaam/emo_inference.py --model_config /workspace/data/emo_model_config.yaml \
    --model_weights /workspace/data/emo_model_weights.ckpt \
    --device cuda --audio_path /workspace/data/example.wav

# angry: 0.000, sad: 0.002, neutral: 0.923, positive: 0.074
```
