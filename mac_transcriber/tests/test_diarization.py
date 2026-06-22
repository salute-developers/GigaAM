from pathlib import Path

import pytest
import torch

from mac_transcriber import asr


class FakeTurn:
    def __init__(self, start, end):
        self.start = start
        self.end = end


def test_build_input_segments_uses_diarization_for_single_file(monkeypatch, tmp_path):
    input_dir = tmp_path
    audio_path = input_dir / "audio.m4a"
    audio_path.write_bytes(b"placeholder")
    waveform = torch.linspace(-0.1, 0.1, steps=asr.SR * 4)

    monkeypatch.setattr(asr, "DIARIZATION_PAD_S", 0.0, raising=False)
    monkeypatch.setattr(asr.gigaam, "load_audio", lambda _path: waveform)
    monkeypatch.setattr(
        asr,
        "diarize_audio",
        lambda _path, *, device, metadata: [
            asr.DiarizedTurn(speaker="SPEAKER_01", start=0.0, end=1.0),
            asr.DiarizedTurn(speaker="SPEAKER_00", start=1.25, end=2.0),
            asr.DiarizedTurn(speaker="SPEAKER_01", start=2.25, end=2.75),
        ],
        raising=False,
    )

    segments, tracks = asr.build_input_segments(
        input_dir=input_dir,
        metadata={},
        device="cpu",
    )

    assert [(segment.speaker, segment.start, segment.end) for segment in segments] == [
        ("Speaker 1", 0.0, 1.0),
        ("Speaker 2", 1.25, 2.0),
        ("Speaker 1", 2.25, 2.75),
    ]
    assert all(segment.track == "audio.m4a" for segment in segments)
    assert [(track.path, track.speaker) for track in tracks] == [
        (audio_path, "Speaker 1"),
        (audio_path, "Speaker 2"),
    ]


def test_build_input_segments_falls_back_when_diarization_is_unavailable(monkeypatch, tmp_path):
    input_dir = tmp_path
    audio_path = input_dir / "audio.m4a"
    audio_path.write_bytes(b"placeholder")
    fallback_segments = [
        asr.Segment(
            speaker="Speaker",
            track="audio.m4a",
            start=0.0,
            end=1.0,
            wav=torch.zeros(asr.SR),
        )
    ]

    def raise_unavailable(_path: Path, *, device: str, metadata: dict):
        raise asr.DiarizationUnavailable("not configured")

    def fake_build_segments(tracks, progress_callback=None):
        assert tracks == [asr.TrackSpec(path=audio_path, speaker="Speaker")]
        return fallback_segments

    monkeypatch.setattr(asr, "diarize_audio", raise_unavailable, raising=False)
    monkeypatch.setattr(asr, "build_segments", fake_build_segments)

    segments, tracks = asr.build_input_segments(
        input_dir=input_dir,
        metadata={},
        device="cpu",
    )

    assert segments == fallback_segments
    assert tracks == [asr.TrackSpec(path=audio_path, speaker="Speaker")]


def test_build_input_segments_keeps_zoom_participant_tracks(monkeypatch, tmp_path):
    input_dir = tmp_path
    participants_dir = input_dir / "participants"
    participants_dir.mkdir()
    (input_dir / "audio.m4a").write_bytes(b"placeholder")
    (participants_dir / "01.m4a").write_bytes(b"participant 1")
    (participants_dir / "02.m4a").write_bytes(b"participant 2")
    fallback_segments = [
        asr.Segment(
            speaker="Alice",
            track="01.m4a",
            start=0.0,
            end=1.0,
            wav=torch.zeros(asr.SR),
        )
    ]

    def fail_if_called(_path: Path, *, device: str, metadata: dict):
        raise AssertionError("diarization should not run for split Zoom tracks")

    def fake_build_segments(tracks, progress_callback=None):
        assert tracks == [
            asr.TrackSpec(path=participants_dir / "01.m4a", speaker="Alice"),
            asr.TrackSpec(path=participants_dir / "02.m4a", speaker="Bob"),
        ]
        return fallback_segments

    monkeypatch.setattr(asr, "diarize_audio", fail_if_called, raising=False)
    monkeypatch.setattr(asr, "build_segments", fake_build_segments)

    segments, tracks = asr.build_input_segments(
        input_dir=input_dir,
        metadata={
            "zoom_participant_tracks": [
                {"speaker_name": "Alice"},
                {"speaker_name": "Bob"},
            ]
        },
        device="cpu",
    )

    assert segments == fallback_segments
    assert tracks == [
        asr.TrackSpec(path=participants_dir / "01.m4a", speaker="Alice"),
        asr.TrackSpec(path=participants_dir / "02.m4a", speaker="Bob"),
    ]


def test_build_input_segments_rejects_zoom_backfill_without_participant_files(monkeypatch, tmp_path):
    input_dir = tmp_path
    (input_dir / "audio.m4a").write_bytes(b"placeholder")

    def fail_if_called(_path: Path, *, device: str, metadata: dict):
        raise AssertionError("diarization should not run for Zoom participant-track mode")

    monkeypatch.setattr(asr, "diarize_audio", fail_if_called, raising=False)

    with pytest.raises(ValueError, match="participant tracks"):
        asr.build_input_segments(
            input_dir=input_dir,
            metadata={"source": "zoom_cloud_backfill", "processing_mode": "zoom_participant_tracks"},
            device="cpu",
        )


def test_build_input_segments_rejects_zoom_backfill_track_count_mismatch(tmp_path):
    input_dir = tmp_path
    participants_dir = input_dir / "participants"
    participants_dir.mkdir()
    (input_dir / "audio.m4a").write_bytes(b"placeholder")
    (participants_dir / "01.m4a").write_bytes(b"participant 1")

    with pytest.raises(ValueError, match="does not match"):
        asr.build_input_segments(
            input_dir=input_dir,
            metadata={
                "processing_mode": "zoom_participant_tracks",
                "zoom_participant_tracks": [
                    {"speaker_name": "Alice"},
                    {"speaker_name": "Bob"},
                ],
            },
            device="cpu",
        )


def test_diarize_audio_converts_source_before_pipeline(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.m4a"
    prepared_path = tmp_path / "audio.diarization.wav"
    audio_path.write_bytes(b"placeholder")
    prepared_path.write_bytes(b"wav")
    seen = {}

    def fake_pipeline(path, **options):
        seen["path"] = path
        seen["options"] = options
        return [(FakeTurn(0.0, 1.2), "SPEAKER_00")]

    monkeypatch.setattr(asr, "diarization_enabled", lambda: True)
    monkeypatch.setattr(asr, "load_diarization_pipeline", lambda *, device: fake_pipeline)
    monkeypatch.setattr(asr, "prepare_diarization_audio", lambda path: prepared_path, raising=False)

    turns = asr.diarize_audio(audio_path, device="cpu", metadata={"participants": ["a", "b"]})

    assert seen == {
        "path": str(prepared_path),
        "options": {"num_speakers": 2},
    }
    assert turns == [asr.DiarizedTurn(speaker="SPEAKER_00", start=0.0, end=1.2)]
