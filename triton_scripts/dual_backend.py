"""Triton backend for offline clips and persistent dual-head streams."""

import json
from pathlib import Path

import hydra
import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from gigaam.onnx_utils import OnnxGigaAMStreaming, infer_onnx, load_onnx


def _input(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    return None if tensor is None else tensor.as_numpy()


def _scalar(request, name, index):
    value = _input(request, name)
    if value is None:
        return None
    return value[0 if value.size == 1 else index]


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


class TritonPythonModel:
    def initialize(self, args):
        config = json.loads(args["model_config"])
        parameters = {k: v["string_value"] for k, v in config["parameters"].items()}
        bundle = Path(__file__).resolve().parent / "graphs"
        self.provider, self.mode = parameters["provider"], parameters["mode"]
        create = packet_graph = None
        manifest_path = bundle / f'{parameters["model_version"]}.streaming.json'
        manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        )
        if self.provider in ("TensorRT", "TensorRT-fp16"):
            from trt_utils import session_factory

            packet_graph = manifest.get("packet_decode")
            create = session_factory(
                self.provider,
                int(args["model_instance_device_id"]),
                direct_streaming=bool(packet_graph or manifest.get("ctc_labels")),
            )
        self.sessions, self.cfg = load_onnx(
            bundle, parameters["model_version"], self.provider, session_factory=create
        )
        if self.mode == "offline" or "whole" in self.sessions:
            self.preprocessor = hydra.utils.instantiate(self.cfg.preprocessor)
            encoder = self.sessions.get("whole", self.sessions["encoder"])
            if device := getattr(encoder, "cuda_device", None):
                self.preprocessor.to(device)
            if self.provider == "CUDAExecutionProvider":
                options = encoder.get_provider_options().get(self.provider)
                if options:
                    self.preprocessor.cuda(int(options["device_id"]))
        self.packet_decoder = self.sessions.get("packet_decode")
        self.packet_heads = manifest.get("packet_heads", ())
        self.default_language = self.cfg.get("onnx_default_language")
        self.max_streams = int(parameters.get("max_streams", "32"))
        self.streams = {}
        self.stream_models = {}
        self.batched = self.sessions.get("manifest", {}).get("batched", False)

    def _language(self, request, index):
        raw = _scalar(request, "LANGUAGE", index)
        language = self.default_language if raw is None else _text(raw)
        if isinstance(language, str) and language.startswith(("{", "[")):
            language = json.loads(language)
            if isinstance(language, list):
                language = "-".join(language)
        return language

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                waves = self._audio(request)
                if self.mode == "offline":
                    texts = self._whole(waves, request)
                else:
                    ids = list(map(_text, _input(request, "STREAM_ID")))
                    if len(ids) != len(waves):
                        raise ValueError("Supply one STREAM_ID per packet")
                    if len(set(ids)) != len(waves):
                        raise ValueError(
                            "Use distinct STREAM_ID values in a packet batch"
                        )
                    if self._one_pass(request, ids, waves):
                        texts = self._whole(waves, request, **self._context(request, 0))
                        responses.append(self._response(texts))
                        continue
                    streams = list(self._streams(request, ids))
                    texts = {"raw": [], "e2e": []}
                    calls = [
                        (key, stream, wave, final)
                        for key, wave, (stream, final) in zip(ids, waves, streams)
                    ]
                    if self.batched and (
                        len(calls) > 1 or self.packet_decoder is not None
                    ):
                        results = self._batch_streams(calls)
                    else:
                        results = (self._stream(*call) for call in calls)
                    try:
                        for result in results:
                            for head, delta in result.items():
                                texts[head].append(delta.text)
                    except Exception:
                        for key in ids:
                            self.streams.pop(key, None)
                        raise
                responses.append(self._response(texts))
            except Exception as error:
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(error)))
                )
        return responses

    @torch.inference_mode()
    def _whole(self, waves, request, **context):
        encoder = self.sessions.get("whole", self.sessions["encoder"])
        return infer_onnx(
            waves,
            self.cfg,
            self.sessions,
            preprocessor=self.preprocessor,
            batch_size=min(len(waves), getattr(encoder, "max_batch", len(waves))),
            progress=False,
            language=[self._language(request, i) for i in range(len(waves))],
            **context,
        )

    def _one_pass(self, request, ids, waves):
        if (
            "whole" not in self.sessions
            or len(self.streams) >= self.max_streams
            or not _input(request, "START").all()
            or not _input(request, "END").all()
        ):
            return False
        context = self._context(request, 0)
        return all(
            wave.numel() >= self.cfg.preprocessor.n_fft
            and key not in self.streams
            and self._context(request, i) == context
            for i, (key, wave) in enumerate(zip(ids, waves, strict=True))
        )

    @staticmethod
    def _response(texts):
        return pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor(
                    head.upper() + "_TEXT",
                    np.array([text.encode("utf-8") for text in rows], dtype=object),
                )
                for head, rows in texts.items()
            ]
        )

    def _audio(self, request):
        audio, lengths = _input(request, "AUDIO"), _input(request, "AUDIO_LENGTH")
        if not 1 <= len(lengths) <= 4 or (lengths < 0).any():
            raise ValueError("Supply one to four non-negative audio lengths")
        for name in ("LANGUAGE", "START", "END", "STREAMING", "FUTURE_CONTEXT_MS"):
            value = _input(request, name)
            if value is not None and value.size not in (1, len(lengths)):
                raise ValueError(f"{name} must have one value or one per packet")
        if sum(lengths.tolist()) > len(audio):
            raise ValueError("AUDIO_LENGTH exceeds the audio payload")
        offsets = np.cumsum(lengths)
        return [
            torch.from_numpy(wave)
            for wave in np.split(audio[: offsets[-1]], offsets[:-1])
        ]

    def _streams(self, request, ids):
        live = len(self.streams)
        for index, stream_id in enumerate(ids):
            start = bool(_scalar(request, "START", index))
            final = bool(_scalar(request, "END", index))
            stream = self.streams.get(stream_id)
            if start == (stream is not None):
                raise ValueError(f"Invalid START for stream {stream_id!r}")
            if start:
                if live >= self.max_streams:
                    raise ValueError("Maximum live stream count reached")
                context = self._context(request, index)
                key = (context.get("streaming", True), context.get("future_context_ms"))
                stream = OnnxGigaAMStreaming(
                    self.cfg,
                    self.sessions,
                    language=self._language(request, index),
                    _model=self.stream_models.get(key),
                    **context,
                )
                self.stream_models.setdefault(key, stream.model)
            live += start - final
            yield stream, final

    def _batch_streams(self, calls):
        groups = {}
        for index, (_, stream, _, _) in enumerate(calls):
            encoder = stream.model.encoder
            groups.setdefault((encoder.streaming, encoder.chunk_size), []).append(index)
        max_batch = getattr(self.sessions["encoder"], "max_batch", len(calls))
        batches = (
            rows[i : i + max_batch]
            for rows in groups.values()
            for i in range(0, len(rows), max_batch)
        )
        with torch.inference_mode():
            for indices in batches:
                streams = [calls[i][1] for i in indices]
                packets = [
                    (
                        stream._packet_features(calls[i][2]),
                        stream.encoder_state,
                        calls[i][3],
                        stream._lang_embedding,
                    )
                    for i, stream in zip(indices, streams, strict=True)
                ]
                encoded, lengths = streams[0].model.encoder.forward_many(packets)
                if not encoded.size(-1):
                    continue
                if self.packet_decoder is not None:
                    self._decode_packet(streams, encoded, lengths)
                else:
                    lengths = torch.from_numpy(lengths)
                    for name, (head, decoder) in streams[0]._decoders.items():
                        decoder.decode(
                            head,
                            encoded,
                            lengths,
                            [stream.decoder_states[name] for stream in streams],
                        )
            results = [stream._publish_packet(final) for _, stream, _, final in calls]
        for key, stream, _, final in calls:
            self._retain_stream(key, stream, final)
        yield from results

    def _decode_packet(self, streams, encoded, lengths):
        groups = [
            [stream.decoder_states[name] for stream in streams]
            for name in self.packet_heads
        ]
        blanks = [streams[0]._decoders[name][1].blank_id for name in self.packet_heads]
        previous = np.array(
            [
                [blank if state.previous is None else state.previous for state in group]
                for blank, group in zip(blanks, groups, strict=True)
            ],
            dtype=np.int64,
        )
        width = self.cfg.heads[self.packet_heads[0]].decoder.pred_hidden
        recurrent = np.zeros((2, len(groups), len(streams), width), dtype=np.float32)
        for head, group in enumerate(groups):
            for row, state in enumerate(group):
                if state.hidden is not None:
                    recurrent[:, head, row] = state.hidden
        tokens, previous, hidden, cell = self.packet_decoder.run(
            None,
            dict(
                encoded=encoded,
                lengths=lengths,
                previous=previous,
                hidden=recurrent[0],
                cell=recurrent[1],
            ),
        )
        for head, group in enumerate(groups):
            for row, state in enumerate(group):
                state.previous = int(previous[head, row])
                state.hidden = np.stack((hidden[head, row], cell[head, row]))
                labels = tokens[:, head, row]
                state.tokens.extend(labels[labels >= 0].tolist())

    def _stream(self, stream_id, stream, wave, final):
        result = stream.transcribe_partial(wave, final=final)
        self._retain_stream(stream_id, stream, final)
        return result

    def _retain_stream(self, stream_id, stream, final):
        if final:
            self.streams.pop(stream_id, None)
        else:
            self.streams[stream_id] = stream

    def _context(self, request, index):
        return {
            name: value.item()
            for name, value in (
                ("streaming", _scalar(request, "STREAMING", index)),
                ("future_context_ms", _scalar(request, "FUTURE_CONTEXT_MS", index)),
            )
            if value is not None
        }

    def finalize(self):
        self.streams.clear()
