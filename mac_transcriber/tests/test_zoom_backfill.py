import json

import pytest

from mac_transcriber.scripts import backfill_zoom_recordings as backfill


def test_select_primary_audio_file_prefers_m4a():
    recording = {
        "recording_files": [
            {"id": "video", "file_type": "MP4", "file_size": 100},
            {"id": "audio", "file_type": "M4A", "recording_type": "audio_only", "file_size": 10},
        ]
    }

    assert backfill.select_primary_audio_file(recording)["id"] == "audio"


def test_prepare_meeting_dir_downloads_audio_and_participants_without_secret_metadata(tmp_path):
    downloads = {}

    class FakeClient:
        def download_recording_file(self, url):
            downloads[url] = downloads.get(url, 0) + 1
            return f"bytes:{url}".encode()

    recording = {
        "uuid": "/abc+==",
        "id": 123,
        "topic": "Team sync",
        "share_url": "https://zoom/share",
        "recording_play_passcode": "secret",
        "recording_files": [
            {
                "id": "audio",
                "file_type": "M4A",
                "recording_type": "audio_only",
                "download_url": "https://zoom/audio",
                "play_url": "https://zoom/play",
                "recording_start": "2026-06-15T10:00:00Z",
            }
        ],
        "participant_audio_files": [
            {
                "id": "p1",
                "file_type": "M4A",
                "participant_name": "Alice",
                "download_url": "https://zoom/p1",
            },
            {
                "id": "p2",
                "file_name": "Audio only - Bob.m4a",
                "download_url": "https://zoom/p2",
            },
        ],
    }

    prepared = backfill.prepare_meeting_dir(
        recording=recording,
        client=FakeClient(),
        work_root=tmp_path,
        force=False,
    )

    metadata_text = (prepared.meeting_dir / "input" / "metadata.json").read_text(encoding="utf-8")
    metadata = json.loads(metadata_text)
    assert prepared.meeting_dir == tmp_path / "zoom__abc___"
    assert (prepared.meeting_dir / "input" / "audio.m4a").read_bytes() == b"bytes:https://zoom/audio"
    assert (prepared.meeting_dir / "input" / "participants" / "01.m4a").exists()
    assert (prepared.meeting_dir / "input" / "participants" / "02.m4a").exists()
    assert metadata["meeting_id"] == "zoom__abc___"
    assert metadata["zoom_uuid"] == "/abc+=="
    assert metadata["title"] == "Team sync"
    assert metadata["participants"] == ["Alice", "Bob"]
    assert [track["speaker_name"] for track in metadata["zoom_participant_tracks"]] == ["Alice", "Bob"]
    assert "download_url" not in metadata_text
    assert "play_url" not in metadata_text
    assert "recording_play_passcode" not in metadata_text


def test_prepare_meeting_dir_reuses_existing_audio_without_download(tmp_path):
    class FailClient:
        def download_recording_file(self, url):
            raise AssertionError("should not download existing meeting")

    meeting_dir = tmp_path / "zoom_existing"
    input_dir = meeting_dir / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "audio.m4a").write_bytes(b"existing")
    participants_dir = input_dir / "participants"
    participants_dir.mkdir()
    (participants_dir / "01.m4a").write_bytes(b"alice")
    (input_dir / "metadata.json").write_text(
        json.dumps(
            {
                "meeting_id": "zoom_existing",
                "title": "Existing",
                "zoom_participant_tracks": [{"speaker_name": "Alice"}],
            }
        ),
        encoding="utf-8",
    )

    prepared = backfill.prepare_meeting_dir(
        recording={"uuid": "existing", "topic": "Existing"},
        client=FailClient(),
        work_root=tmp_path,
        force=False,
    )

    assert prepared.meeting_dir == meeting_dir
    assert prepared.participant_tracks == 1


def test_prepare_meeting_dir_rebuilds_existing_mismatched_participant_tracks(tmp_path):
    downloads = []

    class FakeClient:
        def download_recording_file(self, url):
            downloads.append(url)
            return f"bytes:{url}".encode()

    meeting_dir = tmp_path / "zoom_existing"
    input_dir = meeting_dir / "input"
    participants_dir = input_dir / "participants"
    participants_dir.mkdir(parents=True)
    (input_dir / "audio.m4a").write_bytes(b"stale")
    (participants_dir / "01.m4a").write_bytes(b"alice")
    (input_dir / "metadata.json").write_text(
        json.dumps(
            {
                "meeting_id": "zoom_existing",
                "title": "Existing",
                "zoom_participant_tracks": [
                    {"speaker_name": "Alice"},
                    {"speaker_name": "Bob"},
                ],
            }
        ),
        encoding="utf-8",
    )

    prepared = backfill.prepare_meeting_dir(
        recording={
            "uuid": "existing",
            "topic": "Existing",
            "recording_files": [{"id": "audio", "file_type": "M4A", "download_url": "https://zoom/audio"}],
            "participant_audio_files": [
                {"id": "p1", "participant_name": "Alice", "download_url": "https://zoom/p1"},
                {"id": "p2", "participant_name": "Bob", "download_url": "https://zoom/p2"},
            ],
        },
        client=FakeClient(),
        work_root=tmp_path,
        force=False,
    )

    assert prepared.meeting_dir == meeting_dir
    assert prepared.participant_tracks == 2
    assert downloads == ["https://zoom/audio", "https://zoom/p1", "https://zoom/p2"]
    assert sorted(path.name for path in participants_dir.glob("*.m4a")) == ["01.m4a", "02.m4a"]


def test_prepare_meeting_dir_retries_partial_existing_directory(tmp_path):
    class FakeClient:
        def download_recording_file(self, url):
            return b"audio"

    meeting_dir = tmp_path / "zoom_retry"
    (meeting_dir / "input").mkdir(parents=True)
    (meeting_dir / "input" / "metadata.json").write_text("{}", encoding="utf-8")

    prepared = backfill.prepare_meeting_dir(
        recording={
            "uuid": "retry",
            "topic": "Retry",
            "recording_files": [{"id": "audio", "file_type": "M4A", "download_url": "https://zoom/audio"}],
            "participant_audio_files": [
                {"id": "p1", "participant_name": "Alice", "download_url": "https://zoom/p1"}
            ],
        },
        client=FakeClient(),
        work_root=tmp_path,
        force=False,
    )

    assert prepared.meeting_dir == meeting_dir
    assert (meeting_dir / "input" / "audio.m4a").read_bytes() == b"audio"


def test_prepare_meeting_dir_requires_participant_audio_files(tmp_path):
    class FakeClient:
        def download_recording_file(self, url):
            return b"audio"

    try:
        backfill.prepare_meeting_dir(
            recording={
                "uuid": "mixed",
                "topic": "Mixed only",
                "recording_files": [{"id": "audio", "file_type": "M4A", "download_url": "https://zoom/audio"}],
            },
            client=FakeClient(),
            work_root=tmp_path,
            force=False,
        )
    except ValueError as exc:
        assert "participant_audio_files" in str(exc)
    else:
        raise AssertionError("participant tracks should be required")


def test_prepare_meeting_dir_requires_downloadable_participant_audio_files(tmp_path):
    downloads = []

    class FakeClient:
        def download_recording_file(self, url):
            downloads.append(url)
            return b"audio"

    with pytest.raises(ValueError, match="download_url"):
        backfill.prepare_meeting_dir(
            recording={
                "uuid": "missing-url",
                "topic": "Missing URL",
                "recording_files": [{"id": "audio", "file_type": "M4A", "download_url": "https://zoom/audio"}],
                "participant_audio_files": [{"id": "p1", "participant_name": "Alice"}],
            },
            client=FakeClient(),
            work_root=tmp_path,
            force=False,
        )

    assert downloads == []
    assert not (tmp_path / "zoom_missing-url" / "input" / "audio.m4a").exists()


def test_backfill_fetches_recording_detail_before_download(tmp_path, monkeypatch):
    class FakeClient:
        def __init__(self):
            self.detail_calls = []

        def list_recordings(self, *, user_id, start_date, end_date, page_size):
            return [
                {
                    "uuid": "one",
                    "id": 1,
                    "topic": "One",
                    "recording_files": [{"id": "audio", "file_type": "M4A"}],
                }
            ]

        def get_recordings(self, *, meeting_id):
            self.detail_calls.append(meeting_id)
            return {
                "uuid": meeting_id,
                "id": 1,
                "topic": "One detail",
                "recording_files": [{"id": "audio", "file_type": "M4A", "download_url": "https://zoom/audio"}],
                "participant_audio_files": [
                    {"id": "p1", "participant_name": "Alice", "download_url": "https://zoom/p1"}
                ],
            }

        def download_recording_file(self, url):
            return f"bytes:{url}".encode()

    fake_client = FakeClient()
    monkeypatch.setattr(backfill, "ZoomClient", lambda **_kwargs: fake_client)

    result = backfill.backfill_zoom_recordings(
        env={
            "ZOOM_ACCOUNT_ID": "account",
            "ZOOM_CLIENT_ID": "client",
            "ZOOM_CLIENT_SECRET": "secret",
            "ZOOM_HOST_USER_ID": "me",
        },
        start_date="2026-06-01",
        end_date="2026-06-16",
        work_root=tmp_path,
        dry_run=False,
        limit=10,
        force=False,
        download_only=True,
        page_size=50,
    )

    assert fake_client.detail_calls == ["one"]
    assert result[0]["status"] == "downloaded"
    assert result[0]["participant_tracks"] == 1
    assert (tmp_path / "zoom_one" / "input" / "participants" / "01.m4a").exists()


def test_backfill_does_not_transcribe_missing_downloadable_participant_tracks(tmp_path, monkeypatch):
    class FakeClient:
        def list_recordings(self, *, user_id, start_date, end_date, page_size):
            return [{"uuid": "one", "id": 1, "topic": "One"}]

        def get_recordings(self, *, meeting_id):
            return {
                "uuid": meeting_id,
                "id": 1,
                "topic": "One detail",
                "recording_files": [{"id": "audio", "file_type": "M4A", "download_url": "https://zoom/audio"}],
                "participant_audio_files": [{"id": "p1", "participant_name": "Alice"}],
            }

    def fail_if_transcribed(**_kwargs):
        raise AssertionError("missing participant tracks must not fall back to mixed audio")

    monkeypatch.setattr(backfill, "ZoomClient", lambda **_kwargs: FakeClient())
    monkeypatch.setattr(backfill, "transcribe_meeting", fail_if_transcribed)

    result = backfill.backfill_zoom_recordings(
        env={
            "ZOOM_ACCOUNT_ID": "account",
            "ZOOM_CLIENT_ID": "client",
            "ZOOM_CLIENT_SECRET": "secret",
            "ZOOM_HOST_USER_ID": "me",
        },
        start_date="2026-06-01",
        end_date="2026-06-16",
        work_root=tmp_path,
        dry_run=False,
        limit=10,
        force=False,
        download_only=False,
        page_size=50,
    )

    assert result[0]["status"] == "failed"
    assert "download_url" in result[0]["error"]


def test_backfill_dry_run_lists_recordings_without_download(tmp_path, monkeypatch):
    class FakeClient:
        def list_recordings(self, *, user_id, start_date, end_date, page_size):
            return [{"uuid": "one", "id": 1, "topic": "One"}]

    monkeypatch.setattr(backfill, "ZoomClient", lambda **_kwargs: FakeClient())

    result = backfill.backfill_zoom_recordings(
        env={
            "ZOOM_ACCOUNT_ID": "account",
            "ZOOM_CLIENT_ID": "client",
            "ZOOM_CLIENT_SECRET": "secret",
            "ZOOM_HOST_USER_ID": "me",
        },
        start_date="2026-06-01",
        end_date="2026-06-16",
        work_root=tmp_path,
        dry_run=True,
        limit=10,
        force=False,
        download_only=False,
        page_size=50,
    )

    assert result == [{"meeting_id": "zoom_one", "status": "dry_run", "title": "One"}]
    assert not (tmp_path / "zoom_one").exists()


def test_apply_runtime_env_sets_transcriber_environment(monkeypatch):
    monkeypatch.delenv("MAC_TRANSCRIBER_DATABASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    backfill.apply_runtime_env(
        {
            "MAC_TRANSCRIBER_DATABASE_URL": "postgresql://example/db",
            "OPENAI_API_KEY": "openai-key",
            "ZOOM_CLIENT_SECRET": "zoom-secret",
        }
    )

    assert backfill.os.environ["MAC_TRANSCRIBER_DATABASE_URL"] == "postgresql://example/db"
    assert backfill.os.environ["OPENAI_API_KEY"] == "openai-key"
    assert "ZOOM_CLIENT_SECRET" not in backfill.RUNTIME_ENV_KEYS
