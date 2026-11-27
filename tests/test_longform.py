import logging
import os
import tempfile
from typing import List, Tuple

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

import gigaam
from gigaam.utils import download_long_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


_predictions = {
    "v3_ctc": [
        {
            "transcription": "вечерня отошла давно но в кельях тихо и темно уже и сам игумен строгий свои молитвы прекратил и кости ветхие склонил перекрестясь на одр убогий кругом и сон и тишина но церкви дверь отворена трепещет луч лампады и тускло озаряет он и темну живопись икон и позлащенные оклады и раздается в тишине то тяжкий вздох то шепот важный",  # noqa: E501
            "boundaries": (0.0, 29.700000000000003),
        },
        {
            "transcription": "и мрачно дремлет в вышине старинный свод глухой и влажный стоят за клиросом чернец и грешник неподвижны оба",  # noqa: E501
            "boundaries": (29.700000000000003, 39.400000000000006),
        },
        {
            "transcription": "и шепот их как глаз из гроба и грешник бледен как мертвец монах",  # noqa: E501
            "boundaries": (39.72, 46.2),
        },
        {
            "transcription": "несчастный полно перестань ужасна исповедь злодея заплачена тобою дань тому кто в злобе пламенея лукаво грешника блюдет и к вечной гибели ведет смирись опомнись время время раскаянье покров я разрешу тебя грехов сложи мучительное бремя",  # noqa: E501
            "boundaries": (46.76, 71.16),
        },
    ],
    "v3_e2e_rnnt": [
        {
            "transcription": "Вечерня отошла давно, Но в кельях тихо и темно; Уже и сам игумен строгий Свои молитвы прекратил И кости ветхие склонил, Перекрестясь на одр убогий. Кругом и сон, и тишина; Но церкви дверь отворена; Трепещет луч лампады; И тускло озаряет он И тёмну живопись икон, И возлащённые оклады. И раздаётся в тишине — То тяжкий вздох, то шёпот важный.",  # noqa: E501
            "boundaries": (0.0, 29.700000000000003),
        },
        {
            "transcription": "И мрачно дремлет в вышине старинный свод, глухой и влажный. Стоят за клиросом чернец и грешник, Неподвижны оба.",  # noqa: E501
            "boundaries": (29.700000000000003, 39.400000000000006),
        },
        {
            "transcription": "И шёпот их — как глаз из гроба, и грешник бледен, как мертвец — монах.",  # noqa: E501
            "boundaries": (39.72, 46.2),
        },
        {
            "transcription": "Несчастный! Полно, перестань. Ужасна исповедь злодея, Заплачена тобою дань — Тому, Кто в злобе пламенея Лукавого грешника блюдёт И к вечной гибели ведёт. Смирись, опомнись, Время, время, Раскаянье, покров, Я разрешу тебя, Грехов сложи Мучительное бремя.",  # noqa: E501
            "boundaries": (46.76, 71.16),
        },
    ],
}


def generate_long_audio(duration=60.0, sr=16000, include_silence=True):
    """Generate long test audio with speech-like segments and silence"""
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.zeros_like(t, dtype=np.float32)
    segment_durations = list(np.random.uniform(0.2, 5, size=100))
    current_time = 0.0

    for i, seg_duration in enumerate(segment_durations):
        if current_time + seg_duration > duration:
            break
        seg_t = np.linspace(0, seg_duration, int(sr * seg_duration))
        freq1, freq2, freq3 = 100 + i * 20, 200 + i * 30, 300 + i * 40
        segment = (
            0.4 * np.sin(2 * np.pi * freq1 * seg_t)
            + 0.3 * np.sin(2 * np.pi * freq2 * seg_t)
            + 0.2 * np.sin(2 * np.pi * freq3 * seg_t)
            + 0.1 * np.random.normal(0, 0.2, len(seg_t))
        )
        envelope = signal.windows.tukey(len(segment), alpha=0.1)
        segment = segment * envelope
        start_idx, end_idx = (
            int(current_time * sr),
            int(current_time * sr) + len(segment),
        )
        audio[start_idx:end_idx] = segment
        if include_silence and i < len(segment_durations) - 1:
            current_time += seg_duration + np.random.uniform(0.1, 0.5)
        else:
            current_time += seg_duration
    return audio


def validate_segmentation_boundaries(
    boundaries: List[Tuple[float, float]], audio_duration: float
):
    """Validate segmentation boundaries meet requirements"""
    issues = []
    total_duration = 0.0

    for i, (start, end) in enumerate(boundaries):
        duration = end - start
        if duration < 0.2:
            issues.append(f"Segment {i} too short: {duration:.2f}s")
        if duration > 30.0:
            issues.append(f"Segment {i} too long: {duration:.2f}s")
        if start >= end:
            issues.append(f"Segment {i} invalid boundaries: {start:.2f}-{end:.2f}")
        total_duration += duration

    if boundaries and boundaries[-1][1] > audio_duration:
        issues.append(
            f"Last segment exceeds audio: {boundaries[-1][1]:.2f} > {audio_duration:.2f}"
        )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "total_segments": len(boundaries),
    }


@pytest.mark.parametrize("duration", [30.0, 60.0, 120.0])
def test_segmentation_functionality(duration):
    """Test audio segmentation with different durations"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        try:
            audio = generate_long_audio(duration=duration)
            sf.write(f.name, audio, 16000)

            from gigaam.vad_utils import segment_audio_file

            segments, boundaries = segment_audio_file(f.name, sr=16000)

            validation = validate_segmentation_boundaries(boundaries, duration)
            assert validation[
                "valid"
            ], f"Boundary validation failed: {validation['issues']}"
            assert len(segments) == len(
                boundaries
            ), "Segments and boundaries count mismatch"

            logger.info(f"Segmentation: {len(segments)} segments for {duration}s audio")

        finally:
            if os.path.exists(f.name):
                os.remove(f.name)


@pytest.mark.parametrize("revision", ["v3_ctc", "v3_e2e_rnnt"])
def test_transcribe_longform(revision):
    """Test longform transcription for different models"""
    from gigaam.types import LongformTranscriptionResult, Segment

    model = gigaam.load_model(revision)
    result = model.transcribe_longform(download_long_audio(), word_timestamps=True)
    assert isinstance(result, LongformTranscriptionResult)
    assert len(result.segments) == len(_predictions[revision])
    for segment, expected in zip(result.segments, _predictions[revision]):
        assert segment.text == expected["transcription"]
        assert (segment.start, segment.end) == pytest.approx(
            expected["boundaries"], abs=0.1
        )
        assert isinstance(segment, Segment)
        assert segment.text and 0 <= segment.start < segment.end
        assert segment.end - segment.start <= 30.0
        assert segment.words
        assert all(
            segment.start <= word.start <= word.end <= segment.end + 0.001
            for word in segment.words
        )


@pytest.mark.parametrize("revision", ["v3_ctc"])
def test_longform_consistency(revision):
    """Test that multiple runs produce consistent results"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        try:
            audio = generate_long_audio(duration=30.0)
            sf.write(f.name, audio, 16000)

            model = gigaam.load_model(revision)
            result1 = model.transcribe_longform(f.name)
            result2 = model.transcribe_longform(f.name)

            assert len(result1.segments) == len(
                result2.segments
            ), "Inconsistent segment count"
            for seg1, seg2 in zip(result1.segments, result2.segments):
                assert (seg1.start, seg1.end) == (
                    seg2.start,
                    seg2.end,
                ), "Inconsistent boundaries"

        finally:
            if os.path.exists(f.name):
                os.remove(f.name)


def test_segmentation_edge_cases():
    """Test segmentation with edge cases"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        try:
            # Very short audio
            audio = generate_long_audio(duration=0.5)
            sf.write(f.name, audio, 16000)

            from gigaam.vad_utils import segment_audio_file

            segments, boundaries = segment_audio_file(f.name, sr=16000)

            # Should handle short audio gracefully
            assert isinstance(segments, list), "Should return list even for short audio"

        finally:
            if os.path.exists(f.name):
                os.remove(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_conformer_vad_speech_defaults():
    from gigaam.vad_utils import speech_timestamps

    assert speech_timestamps(np.zeros(25), 0.04, 1.0) == []
    assert speech_timestamps(np.full(25, 0.5), 0.04, 1.0) == [(0.0, 1.0)]
    assert speech_timestamps(np.array([0.5, 0.5, 0, 0, 0]), 0.04, 0.2) == []


def test_vad_onnx(tmp_path):
    import torch

    from gigaam.onnx_utils import infer_onnx, load_onnx
    from gigaam.vad_utils import speech_timestamps

    model = gigaam.load_model("vad", device="cpu")
    audio = torch.cat(
        (torch.zeros(16000), gigaam.load_audio(download_long_audio())[:160001])
    )
    model.to_onnx(str(tmp_path))
    sessions, cfg = load_onnx(str(tmp_path), "vad", provider="CPUExecutionProvider")
    expected = model.speech_probs(audio, batch_size=2).numpy()
    actual = infer_onnx([audio], cfg, sessions, batch_size=2)[0]
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
    assert speech_timestamps(actual, 0.04, len(audio) / 16000) == speech_timestamps(
        expected, 0.04, len(audio) / 16000
    )
