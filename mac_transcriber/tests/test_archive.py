import json
from pathlib import Path
from types import SimpleNamespace

from mac_transcriber import asr
from mac_transcriber.archive import (
    build_file_inventory,
    sha256_file,
    write_meeting_manifest,
)


def test_sha256_file_hashes_file_contents(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_bytes(b"archive me\n")

    assert sha256_file(file_path) == (
        "f8c6b8802a0763060206861d47cd273e89f44e27e49e1614"
        "d4689c889fb739bd"
    )


def test_build_file_inventory_captures_inputs_artifacts_and_status(tmp_path):
    meeting_dir = tmp_path / "meeting-123"
    input_dir = meeting_dir / "input"
    artifacts_dir = meeting_dir / "artifacts"
    input_dir.mkdir(parents=True)
    artifacts_dir.mkdir()

    files = {
        "input/audio.m4a": b"source audio",
        "input/metadata.json": b'{"title":"Demo"}\n',
        "input/participants/alice.m4a": b"alice track",
        "artifacts/transcript.md": "# Transcript\n".encode(),
        "artifacts/coverage.json": b"[]\n",
        "status.json": b'{"status":"done"}\n',
    }
    for relative_path, contents in files.items():
        path = meeting_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)

    (artifacts_dir / "manifest.json").write_text("old manifest\n", encoding="utf-8")
    (meeting_dir / "debug.log").write_text("not archived\n", encoding="utf-8")

    inventory = build_file_inventory(meeting_dir)

    by_path = {entry["relative_path"]: entry for entry in inventory}
    assert list(by_path) == sorted(files)
    assert by_path["input/audio.m4a"]["kind"] == "source_audio"
    assert by_path["input/metadata.json"]["kind"] == "metadata"
    assert by_path["input/participants/alice.m4a"]["kind"] == "participant_audio"
    assert by_path["artifacts/transcript.md"]["kind"] == "transcript"
    assert by_path["artifacts/coverage.json"]["kind"] == "coverage"
    assert by_path["status.json"]["kind"] == "status"
    assert "artifacts/manifest.json" not in by_path
    assert "debug.log" not in by_path
    assert by_path["input/audio.m4a"]["size_bytes"] == len(b"source audio")
    assert by_path["input/audio.m4a"]["sha256"] == sha256_file(
        meeting_dir / "input/audio.m4a"
    )
    assert isinstance(by_path["input/audio.m4a"]["modified_at"], str)


def test_write_meeting_manifest_writes_auditable_manifest(tmp_path):
    meeting_dir = tmp_path / "meeting-456"
    input_dir = meeting_dir / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "audio.m4a").write_bytes(b"audio")
    metadata = {
        "meeting_id": "meeting-456",
        "title": "Синк проекта",
        "source_filename": "созвон.m4a",
        "extra": {"language": "ru"},
    }
    result = {"status": "done", "segments": 2}

    manifest_path = write_meeting_manifest(meeting_dir, metadata, result)

    assert manifest_path == meeting_dir / "artifacts" / "manifest.json"
    assert manifest_path.read_bytes().endswith(b"\n")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["meeting_id"] == "meeting-456"
    assert manifest["title"] == "Синк проекта"
    assert manifest["source_filename"] == "созвон.m4a"
    assert isinstance(manifest["generated_at"], str)
    assert manifest["metadata"] == metadata
    assert manifest["result"] == result
    assert [entry["relative_path"] for entry in manifest["files"]] == [
        "input/audio.m4a"
    ]


def test_write_artifacts_writes_manifest_and_segment_ids(tmp_path, monkeypatch):
    meeting_dir = tmp_path / "meeting-789"
    output_dir = meeting_dir / "artifacts"
    (meeting_dir / "input").mkdir(parents=True)
    (meeting_dir / "input" / "audio.m4a").write_bytes(b"audio")
    (meeting_dir / "input" / "metadata.json").write_text(
        json.dumps(
            {
                "meeting_id": "stale-id",
                "title": "Stale title",
                "source_filename": "stale.m4a",
                "language_code": "ru",
                "participants": ["Alice", "Bob"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "mac_transcriber.reporting.write_report_artifacts",
        lambda **kwargs: _fake_report_artifacts(output_dir, kwargs),
    )
    context_pack = {
        "facts": [{"meeting_id": "old-call", "text": "Старое решение."}],
        "segments": [],
    }
    monkeypatch.setattr(
        "mac_transcriber.memory_db.load_report_context_pack",
        lambda **_kwargs: (context_pack, None),
    )
    monkeypatch.setattr(
        "mac_transcriber.memory_db.sync_meeting_memory",
        lambda *_args: None,
    )

    asr.write_artifacts(
        output_dir=output_dir,
        meeting_id="meeting-789",
        title="Demo meeting",
        source_filename="audio.m4a",
        model_name="v3_e2e_rnnt",
        tracks=[asr.TrackSpec(path=Path("audio.m4a"), speaker="Speaker 1")],
        segments=[
            asr.Segment(
                speaker="Speaker 2",
                track="audio.m4a",
                start=2.0,
                end=3.0,
                text="second",
            ),
            asr.Segment(
                speaker="Speaker 1",
                track="audio.m4a",
                start=0.5,
                end=1.0,
                text="first",
            ),
        ],
        elapsed_seconds=12.3456,
    )

    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists()
    assert json.loads((output_dir / "context_pack.json").read_text(encoding="utf-8")) == context_pack
    report_kwargs = json.loads((output_dir / "report_kwargs.json").read_text(encoding="utf-8"))
    assert report_kwargs["context_pack"] == context_pack
    transcript = json.loads((output_dir / "transcript.json").read_text(encoding="utf-8"))
    assert [row["segment_id"] for row in transcript] == ["S0001", "S0002"]
    assert [row["text"] for row in transcript] == ["first", "second"]
    segments_tsv = (output_dir / "segments.tsv").read_text(encoding="utf-8")
    assert segments_tsv.splitlines() == [
        "start\tend\tspeaker\ttrack\ttext",
        "0.500\t1.000\tSpeaker 1\taudio.m4a\tfirst",
        "2.000\t3.000\tSpeaker 2\taudio.m4a\tsecond",
    ]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["metadata"]["meeting_id"] == "meeting-789"
    assert manifest["metadata"]["title"] == "Demo meeting"
    assert manifest["metadata"]["source_filename"] == "audio.m4a"
    assert manifest["metadata"]["model_name"] == "v3_e2e_rnnt"
    assert manifest["metadata"]["segment_count"] == 2
    assert manifest["metadata"]["track_count"] == 1
    assert manifest["metadata"]["language_code"] == "ru"
    assert manifest["metadata"]["participants"] == ["Alice", "Bob"]
    assert manifest["result"]["meeting_id"] == "meeting-789"
    assert manifest["result"]["title"] == "Demo meeting"
    assert manifest["result"]["source_filename"] == "audio.m4a"
    assert manifest["result"]["model_name"] == "v3_e2e_rnnt"
    assert manifest["result"]["segment_count"] == 2
    assert manifest["result"]["track_count"] == 1
    assert manifest["result"]["segments"] == 2
    assert manifest["result"]["tracks"] == 1
    inventory_paths = {entry["relative_path"] for entry in manifest["files"]}
    assert {
        "artifacts/transcript.md",
        "artifacts/transcript.json",
        "artifacts/context_pack.json",
        "artifacts/summary.json",
    } <= inventory_paths


def test_write_artifacts_records_memory_sync_error_and_rewrites_manifest(tmp_path, monkeypatch):
    meeting_dir = tmp_path / "meeting-memory"
    output_dir = meeting_dir / "artifacts"
    (meeting_dir / "input").mkdir(parents=True)
    (meeting_dir / "input" / "audio.m4a").write_bytes(b"audio")

    monkeypatch.setattr(
        "mac_transcriber.reporting.write_report_artifacts",
        lambda **_kwargs: _fake_report_artifacts(output_dir),
    )
    monkeypatch.setattr(
        "mac_transcriber.memory_db.load_report_context_pack",
        lambda **_kwargs: (None, "context db down"),
    )
    monkeypatch.setattr(
        "mac_transcriber.memory_db.sync_meeting_memory",
        lambda *_args: "db down",
    )

    asr.write_artifacts(
        output_dir=output_dir,
        meeting_id="meeting-memory",
        title="Memory meeting",
        source_filename="audio.m4a",
        model_name="v3_e2e_rnnt",
        tracks=[asr.TrackSpec(path=Path("audio.m4a"), speaker="Speaker 1")],
        segments=[
            asr.Segment(
                speaker="Speaker 1",
                track="audio.m4a",
                start=0.0,
                end=1.0,
                text="hello",
            )
        ],
        elapsed_seconds=1.2,
    )

    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["context_pack_error"] == "context db down"
    assert summary["memory_sync_error"] == "db down"

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["result"]["memory_sync_error"] == "db down"
    summary_entry = next(
        entry
        for entry in manifest["files"]
        if entry["relative_path"] == "artifacts/summary.json"
    )
    assert summary_entry["sha256"] == sha256_file(summary_path)


def _fake_report_artifacts(output_dir, kwargs=None):
    if kwargs is not None:
        safe_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in {"meeting_id", "title", "source_filename", "model_name", "context_pack"}
        }
        (output_dir / "report_kwargs.json").write_text(
            json.dumps(safe_kwargs, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    for filename in (
        "report.json",
        "report.md",
        "report.html",
        "report.typ",
        "report_health.json",
        "coverage.json",
        "slack_summary.md",
    ):
        (output_dir / filename).write_text("{}\n", encoding="utf-8")
    return SimpleNamespace(
        json_path=output_dir / "report.json",
        markdown_path=output_dir / "report.md",
        html_path=output_dir / "report.html",
        typst_path=output_dir / "report.typ",
        health_path=output_dir / "report_health.json",
        coverage_path=output_dir / "coverage.json",
        slack_summary_path=output_dir / "slack_summary.md",
        slack_text="Report ready",
        slack_files=[],
        pdf_path=None,
        pdf_error=None,
        generated_by="fake",
        status="ok",
        alerts=[],
    )
