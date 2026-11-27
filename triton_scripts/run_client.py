"""HTTP helpers for generated v4 full-context and packet endpoints."""

import uuid

import numpy as np
from tritonclient.http import InferenceServerClient, InferInput


def infer_dual_packet(
    audio,
    model_name,
    triton_url="localhost:8000",
    language=None,
    stream_id=None,
    start=False,
    end=False,
    streaming=None,
    future_context_ms=None,
    client=None,
    audio_lengths=None,
    final=None,
):
    """Send one clip or a flat batch of clips or packets with their lengths."""
    import json

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("Audio packets must be mono one-dimensional arrays")
    lengths = np.asarray(
        [len(audio)] if audio_lengths is None else audio_lengths, dtype=np.int64
    )
    if final is not None:
        end = final
    values = {
        "AUDIO": (audio if audio.size else np.zeros(1, dtype=np.float32), "FP32"),
        "AUDIO_LENGTH": (lengths, "INT64"),
    }
    if language is not None:
        tag = json.dumps(language) if isinstance(language, (dict, list)) else language
        values["LANGUAGE"] = (np.array([tag.encode("utf-8")], dtype=object), "BYTES")
    if stream_id is not None:
        ids = [stream_id] if isinstance(stream_id, str) else stream_id
        if len(ids) != len(lengths):
            raise ValueError("Supply one stream_id per packet")
        values.update(
            STREAM_ID=(
                np.array([key.encode("utf-8") for key in ids], dtype=object),
                "BYTES",
            ),
            START=(np.atleast_1d(np.asarray(start, dtype=np.bool_)), "BOOL"),
            END=(np.atleast_1d(np.asarray(end, dtype=np.bool_)), "BOOL"),
        )
        if streaming is not None:
            values["STREAMING"] = (np.array([streaming], dtype=np.bool_), "BOOL")
        if future_context_ms is not None:
            values["FUTURE_CONTEXT_MS"] = (
                np.array([future_context_ms], dtype=np.int64),
                "INT64",
            )
    elif start or end or streaming is not None or future_context_ms is not None:
        raise ValueError("Stream controls require a stream_id")
    inputs = []
    for name, (value, dtype) in values.items():
        item = InferInput(name, value.shape, dtype)
        item.set_data_from_numpy(value)
        inputs.append(item)
    if client is None:
        with InferenceServerClient(url=triton_url.removeprefix("http://")) as client:
            result = client.infer(model_name, inputs)
    else:
        result = client.infer(model_name, inputs)
    texts = {
        head: [
            value.decode("utf-8") for value in result.as_numpy(head.upper() + "_TEXT")
        ]
        for head in ("raw", "e2e")
    }
    return (
        texts
        if audio_lengths is not None
        else {head: rows[0] for head, rows in texts.items()}
    )


def infer_dual_packets(packets, model_name, *, start=False, final=False, **kwargs):
    """Send a mapping of stream IDs to mono packets and return deltas by ID."""
    if not 1 <= len(packets) <= 4:
        raise ValueError("Supply one to four stream packets")
    waves = [np.asarray(wave, dtype=np.float32) for wave in packets.values()]
    if any(wave.ndim != 1 for wave in waves):
        raise ValueError("Audio packets must be mono one-dimensional arrays")
    texts = infer_dual_packet(
        np.concatenate(waves),
        model_name,
        stream_id=list(packets),
        audio_lengths=[len(wave) for wave in waves],
        start=start,
        final=final,
        **kwargs,
    )
    return {
        key: {head: rows[index] for head, rows in texts.items()}
        for index, key in enumerate(packets)
    }


def infer_dual_stream(
    audio_chunks,
    model_name,
    triton_url="localhost:8000",
    language=None,
    stream_id=None,
    future_context_ms=None,
):
    """Yield raw/e2e text deltas for packets, then flush with an empty END packet."""
    stream_id = stream_id or uuid.uuid4().hex
    started = False
    dynamic_context = "_dual_dynamic_" in model_name
    with InferenceServerClient(url=triton_url.removeprefix("http://")) as client:
        for audio in audio_chunks:
            yield infer_dual_packet(
                audio,
                model_name,
                language=language,
                stream_id=stream_id,
                start=not started,
                streaming=True if dynamic_context else None,
                future_context_ms=(
                    future_context_ms if dynamic_context and not started else None
                ),
                client=client,
            )
            started = True
        yield infer_dual_packet(
            np.zeros(0, dtype=np.float32),
            model_name,
            language=language,
            stream_id=stream_id,
            start=not started,
            end=True,
            streaming=True if dynamic_context else None,
            future_context_ms=(
                future_context_ms if dynamic_context and not started else None
            ),
            client=client,
        )
