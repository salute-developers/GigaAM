import json

from fastapi.testclient import TestClient

from mac_transcriber.archive import sha256_file, write_meeting_manifest
from mac_transcriber import service


def test_create_meeting_persists_form_title(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "ROOT", tmp_path)
    monkeypatch.setattr(service, "API_KEY", "")
    monkeypatch.setattr(service, "_process_meeting", lambda _meeting_id: None)

    client = TestClient(service.app)
    response = client.post(
        "/meetings",
        data={"title": "Проектный синк"},
        files={"file": ("recording.m4a", b"fake audio", "audio/mp4")},
    )

    assert response.status_code == 200
    meeting_id = response.json()["id"]
    metadata = json.loads(
        (tmp_path / "meetings" / meeting_id / "input" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["title"] == "Проектный синк"


def test_create_meeting_derives_title_from_zoom_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "ROOT", tmp_path)
    monkeypatch.setattr(service, "API_KEY", "")
    monkeypatch.setattr(service, "_process_meeting", lambda _meeting_id: None)

    client = TestClient(service.app)
    response = client.post(
        "/meetings",
        files={
            "file": (
                "Быстрый созвон - -tech-leads - 2026-06-15 10-02.m4a",
                b"fake audio",
                "audio/mp4",
            )
        },
    )

    assert response.status_code == 200
    meeting_id = response.json()["id"]
    metadata = json.loads(
        (tmp_path / "meetings" / meeting_id / "input" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["title"] == "Быстрый созвон - tech-leads"


def test_create_meeting_persists_zoom_participant_tracks(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "ROOT", tmp_path)
    monkeypatch.setattr(service, "API_KEY", "")
    monkeypatch.setattr(service, "_process_meeting", lambda _meeting_id: None)

    client = TestClient(service.app)
    response = client.post(
        "/meetings",
        data={
            "processing_mode": "zoom_participant_tracks",
            "participants": "Alice,Bob",
            "zoom_participant_tracks_json": json.dumps(
                [{"speaker_name": "Alice"}, {"speaker_name": "Bob"}]
            ),
        },
        files=[
            ("file", ("audio.m4a", b"mixed archive", "audio/mp4")),
            ("zoom_participant_files", ("alice.m4a", b"alice audio", "audio/mp4")),
            ("zoom_participant_files", ("bob.m4a", b"bob audio", "audio/mp4")),
        ],
    )

    assert response.status_code == 200
    meeting_id = response.json()["id"]
    input_dir = tmp_path / "meetings" / meeting_id / "input"
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))

    assert (input_dir / "audio.m4a").read_bytes() == b"mixed archive"
    assert (input_dir / "participants" / "01.m4a").read_bytes() == b"alice audio"
    assert (input_dir / "participants" / "02.m4a").read_bytes() == b"bob audio"
    assert metadata["processing_mode"] == "zoom_participant_tracks"
    assert metadata["participants"] == ["Alice", "Bob"]
    assert metadata["zoom_participant_tracks"] == [
        {"speaker_name": "Alice"},
        {"speaker_name": "Bob"},
    ]


def test_manifest_json_artifact_is_available(tmp_path):
    meeting_dir = tmp_path / "meeting-with-manifest"
    artifacts_dir = meeting_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    (artifacts_dir / "context_pack.json").write_text("{}\n", encoding="utf-8")

    assert service.ARTIFACTS["manifest_json"] == ("manifest.json", "application/json")
    assert service.ARTIFACTS["context_pack_json"] == ("context_pack.json", "application/json")
    assert "manifest_json" in service._available_artifacts(meeting_dir)
    assert "context_pack_json" in service._available_artifacts(meeting_dir)


def test_terminal_status_refreshes_existing_manifest_and_syncs_memory(
    tmp_path,
    monkeypatch,
):
    meeting_dir = tmp_path / "meeting-final-status"
    input_dir = meeting_dir / "input"
    artifacts_dir = meeting_dir / "artifacts"
    input_dir.mkdir(parents=True)
    artifacts_dir.mkdir()
    metadata = {
        "meeting_id": "meeting-final-status",
        "title": "Final status",
        "source_filename": "audio.m4a",
    }
    result = {"meeting_id": "meeting-final-status", "segments": 1}
    (input_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (artifacts_dir / "summary.json").write_text("{}\n", encoding="utf-8")
    (meeting_dir / "status.json").write_text(
        json.dumps({"status": "processing"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_meeting_manifest(meeting_dir, metadata, result)
    sync_calls = []

    def fake_sync(sync_meeting_dir, manifest_path):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status_entry = next(
            entry
            for entry in manifest["files"]
            if entry["relative_path"] == "status.json"
        )
        assert status_entry["sha256"] == sha256_file(sync_meeting_dir / "status.json")
        sync_calls.append(
            {
                "meeting_dir": sync_meeting_dir,
                "manifest_path": manifest_path,
                "status_hash": status_entry["sha256"],
            }
        )
        return "db down"

    monkeypatch.setattr(service, "sync_meeting_memory", fake_sync, raising=False)

    service._write_status(
        meeting_dir,
        "completed",
        phase="completed",
        progress=1.0,
        result={"segments": 1},
    )

    assert len(sync_calls) == 1
    assert sync_calls[0]["meeting_dir"] == meeting_dir
    assert sync_calls[0]["manifest_path"] == artifacts_dir / "manifest.json"
    status = json.loads((meeting_dir / "status.json").read_text(encoding="utf-8"))
    assert status["memory_sync_error"] == "db down"
    manifest = json.loads(
        (artifacts_dir / "manifest.json").read_text(encoding="utf-8")
    )
    status_entry = next(
        entry for entry in manifest["files"] if entry["relative_path"] == "status.json"
    )
    assert sync_calls[0]["status_hash"] != status_entry["sha256"]
    assert status_entry["sha256"] == sha256_file(meeting_dir / "status.json")
    assert manifest["metadata"] == metadata
    assert manifest["result"] == result
