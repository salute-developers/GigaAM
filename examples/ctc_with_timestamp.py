import argparse

import torch
import torchaudio
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor,
)
from nemo.collections.asr.parts.preprocessing.features import (
    FilterbankFeaturesTA as NeMoFilterbankFeaturesTA,
)


class FilterbankFeaturesTA(NeMoFilterbankFeaturesTA):
    def __init__(self, mel_scale: str = "htk", wkwargs=None, **kwargs):
        if "window_size" in kwargs:
            del kwargs["window_size"]
        if "window_stride" in kwargs:
            del kwargs["window_stride"]

        super().__init__(**kwargs)

        self._mel_spec_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=self._sample_rate,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=kwargs["nfilt"],
            window_fn=self.torch_windows[kwargs["window"]],
            mel_scale=mel_scale,
            norm=kwargs["mel_norm"],
            n_fft=kwargs["n_fft"],
            f_max=kwargs.get("highfreq", None),
            f_min=kwargs.get("lowfreq", 0),
            wkwargs=wkwargs,
        )


class AudioToMelSpectrogramPreprocessor(NeMoAudioToMelSpectrogramPreprocessor):
    def __init__(self, mel_scale: str = "htk", **kwargs):
        super().__init__(**kwargs)
        kwargs["nfilt"] = kwargs["features"]
        del kwargs["features"]
        self.featurizer = (
            FilterbankFeaturesTA(  # Deprecated arguments; kept for config compatibility
                mel_scale=mel_scale,
                **kwargs,
            )
        )

def get_timestamps(logprobs, blank_id, stride, sample_rate):
    hypotheses, word_timestamps = [], []
    timestamp_dict = {}
    last_char = None
    current_word = ''
    word_start_frame = 0

    # Алфавит из конфигурации
    alphabet = [' ', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']

    for frame, logprob in enumerate(logprobs[0]):
        char = logprob.argmax().item()
        
        if char != blank_id:
            if char != last_char:
                if current_word and char == 0:  # Пробел
                    end_time = frame * stride / sample_rate
                    timestamp_dict[current_word] = (word_start_frame * stride / sample_rate, end_time)
                    word_timestamps.append((current_word, word_start_frame * stride / sample_rate, end_time))
                    hypotheses.append(current_word)
                    current_word = ''
                    word_start_frame = frame
                else:
                    current_word += alphabet[char]
            last_char = char

    if current_word:
        end_time = len(logprobs[0]) * stride / sample_rate
        timestamp_dict[current_word] = (word_start_frame * stride / sample_rate, end_time)
        word_timestamps.append((current_word, word_start_frame * stride / sample_rate, end_time))
        hypotheses.append(current_word)

    return ' '.join(hypotheses), word_timestamps

def parse_with_timestamp(audio_path: str, model: EncDecCTCModel):
    device = model.device
    
    # Загрузка аудио
    audio, sample_rate = torchaudio.load(audio_path)
    audio = audio.to(device)

    # Убедимся, что аудио имеет правильную форму (batch, time)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.dim() > 2:
        audio = audio.squeeze()
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Получаем длину аудио
    audio_length = torch.tensor([audio.shape[1]], device=device)

    # Получаем логарифмические вероятности
    with torch.no_grad():
        log_probs, encoded_len, greedy_predictions = model(
            input_signal=audio, input_signal_length=audio_length
        )

    # Получаем stride
    stride = model.cfg.preprocessor['n_window_stride']
    
    blank_id = len(model.decoder.vocabulary)
    transcription, timestamps = get_timestamps(log_probs.cpu().numpy(), blank_id, stride, sample_rate)

    # print(f"Transcription: {transcription}")
    # print("Word timestamps:")
    # for word, start, end in timestamps:
    #     print(f"  {word}: {start:.2f}s - {end:.2f}s")
    
    return transcription, timestamps

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = EncDecCTCModel.from_config_file("./ctc_model_config.yaml")
ckpt = torch.load("./ctc_model_weights.ckpt", map_location="cpu")
model.load_state_dict(ckpt, strict=False)
model.eval()
model = model.to(device)

transcription, timestamps = parse_with_timestamp("example.wav", model)
print(f"transcription: {transcription}")
print(f"timestamps: {timestamps}")
