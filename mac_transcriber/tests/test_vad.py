from pathlib import Path

import torch

from mac_transcriber import asr


def test_vad_backend_defaults_to_rms(monkeypatch):
    monkeypatch.delenv("MAC_TRANSCRIBER_VAD", raising=False)
    assert asr.vad_backend() == "rms"


def test_vad_backend_reads_silero_case_insensitive(monkeypatch):
    monkeypatch.setenv("MAC_TRANSCRIBER_VAD", "SILERO")
    assert asr.vad_backend() == "silero"


def test_vad_backend_unknown_value_falls_back_to_rms(monkeypatch):
    monkeypatch.setenv("MAC_TRANSCRIBER_VAD", "whisper")
    assert asr.vad_backend() == "rms"


def test_detect_speech_intervals_silero_uses_loaded_model(monkeypatch):
    wav = torch.zeros(asr.SR * 5)
    sentinel = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(asr, "load_silero_model", lambda: sentinel)

    import silero_vad

    def fake_timestamps(audio, model, *, sampling_rate):
        captured["model"] = model
        captured["sampling_rate"] = sampling_rate
        return [{"start": asr.SR, "end": asr.SR * 2}]

    monkeypatch.setattr(silero_vad, "get_speech_timestamps", fake_timestamps)

    intervals = asr.detect_speech_intervals(wav, backend="silero")

    assert captured["model"] is sentinel
    assert captured["sampling_rate"] == asr.SR
    assert intervals
    # Время абсолютное и в пределах исходного аудио.
    for start, end in intervals:
        assert 0 <= start < end <= wav.numel()


def test_detect_speech_intervals_falls_back_to_rms_when_silero_unavailable(monkeypatch):
    wav = torch.zeros(asr.SR * 3)

    def raise_unavailable():
        raise asr.SileroUnavailable("silero-vad is not installed")

    monkeypatch.setattr(asr, "load_silero_model", raise_unavailable)

    silero_intervals = asr.detect_speech_intervals(wav, backend="silero")
    rms_intervals = asr.detect_speech_intervals(wav, backend="rms")

    assert silero_intervals == rms_intervals


def test_build_segments_silero_keeps_absolute_sorted_times(monkeypatch):
    monkeypatch.setenv("MAC_TRANSCRIBER_VAD", "silero")
    waveform = torch.zeros(asr.SR * 10)
    monkeypatch.setattr(asr.gigaam, "load_audio", lambda _path: waveform)
    monkeypatch.setattr(asr, "load_silero_model", lambda: object())

    import silero_vad

    # Намеренно отдаём куски в обратном порядке: финальная сортировка по времени
    # должна выстроить их хронологически, а не в порядке детекции.
    monkeypatch.setattr(
        silero_vad,
        "get_speech_timestamps",
        lambda audio, model, *, sampling_rate: [
            {"start": asr.SR * 5, "end": asr.SR * 6},
            {"start": asr.SR * 1, "end": asr.SR * 2},
        ],
    )

    tracks = [asr.TrackSpec(path=Path("01.m4a"), speaker="Alice")]
    segments = asr.build_segments(tracks)

    assert [segment.speaker for segment in segments] == ["Alice", "Alice"]
    assert all(segment.track == "01.m4a" for segment in segments)
    starts = [segment.start for segment in segments]
    assert starts == sorted(starts)
    assert segments[0].start < segments[1].start
