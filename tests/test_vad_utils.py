import torch


class _FakeSegment:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _FakeTimeline:
    def support(self):
        return [_FakeSegment(1.0, 3.0), _FakeSegment(4.0, 6.0)]


class _FakeSadSegments:
    def get_timeline(self):
        return _FakeTimeline()


class _FakePyannote4Pipeline:
    def __init__(self):
        self.used_apply = False

    def prepare_one(self, audio_file, preload=False):
        assert set(audio_file) == {"waveform", "sample_rate"}
        assert audio_file["waveform"].shape == (1, 80)
        assert audio_file["sample_rate"] == 10
        assert preload is False
        return audio_file

    def apply(self, prepared):
        assert prepared["sample_rate"] == 10
        self.used_apply = True
        return _FakeSadSegments()

    def __call__(self, audio_file):
        raise AssertionError("segment_audio_file should use preloaded waveform input")


def test_segmentation_uses_preloaded_waveform_for_pyannote_pipeline(monkeypatch):
    """VAD should use the same decoded waveform later sliced for ASR chunks."""
    from gigaam import vad_utils

    pipeline = _FakePyannote4Pipeline()
    monkeypatch.setattr(vad_utils, "get_pipeline", lambda device: pipeline)
    monkeypatch.setattr(
        vad_utils,
        "load_audio",
        lambda wav_file, sample_rate: torch.arange(80, dtype=torch.float32),
    )

    segments, boundaries = vad_utils.segment_audio_file(
        "fake.wav",
        sr=10,
        max_duration=10.0,
        min_duration=8.0,
        strict_limit_duration=30.0,
    )

    assert pipeline.used_apply
    assert boundaries == [(1.0, 6.0)]
    assert torch.equal(segments[0], torch.arange(10, 60, dtype=torch.float32))
