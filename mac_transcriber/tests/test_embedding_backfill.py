from mac_transcriber.scripts import backfill_embeddings


def test_backfill_embeddings_cli_loads_env_and_calls_upsert(monkeypatch, tmp_path, capsys):
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "\n".join(
            [
                "MAC_TRANSCRIBER_DATABASE_URL=postgresql://example/db",
                "OPENAI_API_KEY=test-key",
            ]
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_upsert(*args, **kwargs):
        calls.append((args, kwargs))
        return {"meetings": 1, "chunks": 3}

    monkeypatch.setattr(backfill_embeddings.memory_db, "upsert_meeting_embeddings", fake_upsert)

    result = backfill_embeddings.main(
        [
            "--env-file",
            str(env_file),
            "--meeting-id",
            "meeting-123",
            "--model",
            "text-embedding-3-small",
            "--batch-size",
            "7",
        ]
    )

    assert result == 0
    assert calls == [
        (
            ("postgresql://example/db",),
            {
                "api_key": "test-key",
                "meeting_id": "meeting-123",
                "model": "text-embedding-3-small",
                "batch_size": 7,
            },
        )
    ]
    assert '"chunks": 3' in capsys.readouterr().out
