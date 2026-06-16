import json

from mac_transcriber import zoom_import
from mac_transcriber.scripts import import_zoom_vtt


def test_parse_zoom_vtt_extracts_segments_and_speakers():
    vtt = """WEBVTT

1
00:00:01.000 --> 00:00:03.500
Ilya Savin: Привет.
Продолжаем план.

2
00:00:04.000 --> 00:00:05.250
Без спикера
"""

    segments = zoom_import.parse_zoom_vtt(vtt)

    assert [(segment.start, segment.end, segment.speaker, segment.track, segment.text) for segment in segments] == [
        (1.0, 3.5, "Ilya Savin", "zoom_transcript", "Привет. Продолжаем план."),
        (4.0, 5.25, "Zoom", "zoom_transcript", "Без спикера"),
    ]


def test_segments_from_zoom_timeline_extracts_connector_segments():
    timeline = [
        {
            "display_name": "Ilya",
            "ts": "00:00:03.226",
            "end_ts": "00:00:05.410",
            "text": "Нужно все записывать",
            "avatar": "https://example.invalid/avatar.png",
        },
        {
            "display_name": "",
            "ts": "00:00:06.000",
            "end_ts": "00:00:07.000",
            "text": "Без имени",
        },
    ]

    segments = zoom_import.segments_from_zoom_timeline(timeline)

    assert [(segment.start, segment.end, segment.speaker, segment.track, segment.text) for segment in segments] == [
        (3.226, 5.41, "Ilya", "zoom_connector", "Нужно все записывать"),
        (6.0, 7.0, "Zoom", "zoom_connector", "Без имени"),
    ]


def test_sanitize_zoom_metadata_removes_secrets():
    metadata = {
        "uuid": "abc",
        "recording_play_passcode": "secret",
        "share_url": "https://zoom.example/share",
        "recording_files": [
            {
                "id": "file-1",
                "file_type": "TRANSCRIPT",
                "download_url": "https://zoom.example/download",
                "play_url": "https://zoom.example/play",
            }
        ],
        "participant_audio_files": [
            {
                "id": "participant-1",
                "participant_name": "Alice",
                "download_url": "https://zoom.example/participant",
            }
        ],
    }

    sanitized = zoom_import.sanitize_zoom_recording_metadata(metadata)

    assert "recording_play_passcode" not in sanitized
    assert sanitized["share_url"] == "https://zoom.example/share"
    assert sanitized["recording_files"] == [{"id": "file-1", "file_type": "TRANSCRIPT"}]
    assert sanitized["participant_audio_files"] == [{"id": "participant-1", "participant_name": "Alice"}]


def test_import_zoom_vtt_writes_inputs_and_artifacts(tmp_path, monkeypatch):
    calls = []

    def fake_write_artifacts(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("mac_transcriber.asr.write_artifacts", fake_write_artifacts)
    ticks = iter([10.0, 10.0])
    monkeypatch.setattr("mac_transcriber.zoom_import.monotonic", lambda: next(ticks))
    meeting_dir = tmp_path / "zoom-meeting"
    metadata = {
        "uuid": "m7y0",
        "topic": "Быстрый созвон",
        "start_time": "2026-06-15T13:30:00Z",
        "recording_play_passcode": "secret",
    }

    result = zoom_import.import_zoom_vtt(
        meeting_dir=meeting_dir,
        metadata=metadata,
        vtt_text="WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nSpeaker: Hello\n",
    )

    saved_metadata = json.loads((meeting_dir / "input" / "metadata.json").read_text(encoding="utf-8"))
    assert saved_metadata["meeting_id"] == "m7y0"
    assert saved_metadata["title"] == "Быстрый созвон"
    assert "recording_play_passcode" not in saved_metadata
    assert (meeting_dir / "input" / "zoom_transcript.vtt").read_text(encoding="utf-8").startswith("WEBVTT")
    assert result == {"segments": 1, "tracks": 1, "elapsed_seconds": 0.0}

    assert len(calls) == 1
    assert calls[0]["meeting_id"] == "m7y0"
    assert calls[0]["title"] == "Быстрый созвон"
    assert calls[0]["model_name"] == "zoom_transcript"
    assert calls[0]["segments"][0].text == "Hello"


def test_import_zoom_timeline_writes_inputs_and_artifacts(tmp_path, monkeypatch):
    calls = []

    def fake_write_artifacts(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("mac_transcriber.asr.write_artifacts", fake_write_artifacts)
    ticks = iter([10.0, 10.0])
    monkeypatch.setattr("mac_transcriber.zoom_import.monotonic", lambda: next(ticks))
    meeting_dir = tmp_path / "zoom-meeting"
    metadata = {"meeting_id": "zoom-1", "title": "Zoom topic"}
    timeline = [{"display_name": "Ilya", "ts": "00:00:01.000", "end_ts": "00:00:02.000", "text": "Hello"}]

    result = zoom_import.import_zoom_timeline(
        meeting_dir=meeting_dir,
        metadata=metadata,
        timeline=timeline,
    )

    assert json.loads((meeting_dir / "input" / "zoom_timeline.json").read_text(encoding="utf-8")) == timeline
    assert result == {"segments": 1, "tracks": 1, "elapsed_seconds": 0.0}
    assert calls[0]["source_filename"] == "zoom_timeline.json"
    assert calls[0]["model_name"] == "zoom_connector"
    assert calls[0]["segments"][0].speaker == "Ilya"


def test_import_zoom_vtt_cli_imports_metadata_and_vtt(tmp_path, monkeypatch, capsys):
    calls = []
    vtt_path = tmp_path / "source.vtt"
    metadata_path = tmp_path / "metadata.json"
    vtt_path.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n", encoding="utf-8")
    metadata_path.write_text(json.dumps({"uuid": "zoom-1", "topic": "Zoom topic"}), encoding="utf-8")

    def fake_import_zoom_vtt(**kwargs):
        calls.append(kwargs)
        return {"segments": 1, "tracks": 1, "elapsed_seconds": 0.0}

    monkeypatch.setattr("mac_transcriber.zoom_import.import_zoom_vtt", fake_import_zoom_vtt)
    exit_code = import_zoom_vtt.main(
        [
            str(vtt_path),
            "--metadata",
            str(metadata_path),
            "--work-root",
            str(tmp_path / "meetings"),
        ]
    )

    assert exit_code == 0
    assert calls[0]["meeting_dir"] == tmp_path / "meetings" / "zoom-1"
    assert calls[0]["metadata"]["topic"] == "Zoom topic"
    assert calls[0]["vtt_text"].startswith("WEBVTT")
    payload = json.loads(capsys.readouterr().out)
    assert payload["artifact_dir"] == str(tmp_path / "meetings" / "zoom-1" / "artifacts")
    assert payload["segments"] == 1


def test_import_zoom_vtt_cli_sanitizes_meeting_directory_name(tmp_path, monkeypatch):
    calls = []
    vtt_path = tmp_path / "source.vtt"
    metadata_path = tmp_path / "metadata.json"
    vtt_path.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n", encoding="utf-8")
    metadata_path.write_text(json.dumps({"uuid": "/ZIZUd6UQ3ORsbu2VWJ9+Q=="}), encoding="utf-8")

    def fake_import_zoom_vtt(**kwargs):
        calls.append(kwargs)
        return {"segments": 1, "tracks": 1, "elapsed_seconds": 0.0}

    monkeypatch.setattr("mac_transcriber.zoom_import.import_zoom_vtt", fake_import_zoom_vtt)
    exit_code = import_zoom_vtt.main(
        [
            str(vtt_path),
            "--metadata",
            str(metadata_path),
            "--work-root",
            str(tmp_path / "meetings"),
        ]
    )

    assert exit_code == 0
    assert calls[0]["meeting_dir"] == tmp_path / "meetings" / "_ZIZUd6UQ3ORsbu2VWJ9_Q__"
    assert calls[0]["metadata"]["meeting_id"] == "/ZIZUd6UQ3ORsbu2VWJ9+Q=="
