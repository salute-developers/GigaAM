import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import hydra
import numpy as np
import omegaconf
import onnxruntime as rt
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from .decoder import CTCHead, RNNTHead
from .decoding import Tokenizer
from .language import LanguageEmbedding
from .preprocess import FeatureExtractor
from .streaming import EncoderStreamState, GigaAMStreaming
from .utils import AudioDataset

warnings.simplefilter("ignore", category=UserWarning)

MAX_LETTERS_PER_FRAME = 3


def _session_float_dtype(session: rt.InferenceSession, name=None) -> np.dtype:
    """Infer numpy float dtype from a named or first float input of an ONNX session."""
    _type_map: Dict[str, np.dtype] = {
        "tensor(float16)": np.dtype(np.float16),
        "tensor(float)": np.dtype(np.float32),
        "tensor(double)": np.dtype(np.float64),
    }
    for inp in [*session.get_inputs(), *session.get_outputs()]:
        if name is not None and inp.name != name:
            continue
        if inp.type in _type_map:
            return _type_map[inp.type]
    return np.dtype(np.float32)


def _build_inputs(session: rt.InferenceSession, values: List[np.ndarray]) -> dict:
    return {node.name: data for node, data in zip(session.get_inputs(), values)}


def infer_onnx(
    data: Union[str, Sequence[Union[str, np.ndarray, torch.Tensor]]],
    model_cfg: omegaconf.DictConfig,
    sessions: Any,
    preprocessor: Optional[FeatureExtractor] = None,
    tokenizer: Optional[Tokenizer] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    progress: bool = True,
    language=None,
    *,
    streaming=None,
    future_context_ms=None,
    vad=None,
) -> Union[List[str], np.ndarray, List[np.ndarray], Dict[str, List[str]]]:
    """
    Perform inference of GigaAM model with ONNX Runtime.

    Parameters
    ----------
    data : Path to a manifest file or an iterable of audio paths / waveforms.
    model_cfg : Model configuration.
    sessions : List of ONNX Runtime inference sessions.
    preprocessor : Optional[FeatureExtractor].
    tokenizer : Optional[Tokenizer].
    batch_size : Inference batch size.
    num_workers : Number of workers for data loading (use for large datasets).
    progress : Whether to show progress bar.
    vad : Optional (VAD sessions, VAD config) from load_onnx. Segments long-form
        audio paths before ASR; supports a path or a sequence of paths.

    Returns
    -------
    Union[List[str], np.ndarray, List[np.ndarray]]
        List of texts (ASR) / probs (Emo) / arrays (SSL) per sample.
    """
    if model_cfg.model_name == "vad":
        from .vad_utils import speech_probabilities

        preprocessor = preprocessor or hydra.utils.instantiate(model_cfg.preprocessor)
        samples = AudioDataset(
            [data] if isinstance(data, str) and Path(data).suffix != ".tsv" else data
        )
        session = sessions[0]
        dtype = _session_float_dtype(session)
        feature_dtype = torch.from_numpy(np.empty((), dtype)).dtype

        def forward(features, lengths):
            inputs = [features.to(feature_dtype), lengths.cpu().numpy()]
            return torch.from_numpy(_run(session, inputs)[0])

        return [
            speech_probabilities(
                samples[i], preprocessor, forward, batch_size, model_cfg
            ).numpy()
            for i in range(len(samples))
        ]
    if vad is not None:
        from .vad_utils import segment_audio_file

        paths = [data] if isinstance(data, str) else data
        longform_texts: Dict[str, List[str]] = {
            name: [] for name in (model_cfg.heads if model_cfg.get("heads") else [""])
        }
        for index, path in enumerate(paths):
            if not isinstance(path, str) or Path(path).suffix == ".tsv":
                raise ValueError("Long-form ONNX inference requires audio paths")
            segments, _ = segment_audio_file(
                path,
                16000,
                vad=lambda audio: cast(
                    List[np.ndarray],
                    infer_onnx([audio], vad[1], vad[0], batch_size=batch_size),
                )[0],
            )
            longform_decoded = (
                cast(
                    Union[List[str], Dict[str, List[str]]],
                    infer_onnx(
                        segments,
                        model_cfg,
                        sessions,
                        preprocessor,
                        tokenizer,
                        batch_size,
                        num_workers,
                        progress,
                        (
                            language[index]
                            if isinstance(language, (list, tuple))
                            else language
                        ),
                        streaming=streaming,
                        future_context_ms=future_context_ms,
                    ),
                )
                if segments
                else {name: [] for name in longform_texts}
            )
            batches = (
                longform_decoded
                if isinstance(longform_decoded, dict)
                else {"": longform_decoded}
            )
            for name, chunk_texts in batches.items():
                longform_texts[name].append(" ".join(chunk_texts))
        return longform_texts if model_cfg.get("heads") else longform_texts[""]
    manifest = sessions.get("manifest") if isinstance(sessions, dict) else None
    if manifest and "whole" not in sessions:
        raise ValueError(
            "Re-export streaming artifacts with the batched whole-audio graph"
        )
    model_name = model_cfg.model_name

    if any(s in model_name for s in ["v1", "v2", "emo"]) and batch_size > 32:
        logging.warning(
            f"Batch size {batch_size} can be too large for v1/v2-family models. "
            "This value can cause CUDA/cuDNN errors in Conv2d subsampling. "
            "Forcing batch size to 32."
        )
        batch_size = 32

    if preprocessor is None:
        preprocessor = hydra.utils.instantiate(model_cfg.preprocessor)
        encoder = (
            sessions.get("whole", sessions["encoder"])
            if isinstance(sessions, dict)
            else sessions[0]
        )
        device = getattr(encoder, "cuda_device", None)
        if device is None and hasattr(encoder, "get_provider_options"):
            options = encoder.get_provider_options().get("CUDAExecutionProvider")
            if options:
                device = torch.device("cuda", int(options["device_id"]))
        if device is not None:
            preprocessor.to(device)
    dual = bool(model_cfg.get("heads"))
    if isinstance(data, str) and Path(data).suffix != ".tsv":
        data = [data]

    dataset = AudioDataset(data)

    loader = (
        [AudioDataset.collate(list(dataset))]
        if not num_workers and 0 < len(dataset) <= batch_size
        else torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=AudioDataset.collate,
            num_workers=num_workers,
        )
    )
    enc_sess = (
        sessions.get("whole", sessions["encoder"])
        if isinstance(sessions, dict)
        else sessions[0]
    )
    dtype = _session_float_dtype(enc_sess)
    device = (
        next(preprocessor.buffers(), torch.empty(0)).device
        if isinstance(preprocessor, nn.Module)
        else torch.device("cpu")
    )
    feature_dtype = torch.from_numpy(np.empty((), dtype)).dtype
    representations = "emo" in model_name or "ssl" in model_name
    label_decodings = sessions.get("fc", {}) if isinstance(sessions, dict) else {}
    advance = sessions.get("advance") if isinstance(sessions, dict) else None
    packet_decoder = sessions.get("packet_decode") if manifest else None
    outputs = []
    model = None
    if not (representations or label_decodings):
        if isinstance(sessions, dict) and tokenizer is None:
            model = sessions.get("whole_model")
        if model is None:
            model = _Model(model_cfg, sessions, tokenizer=tokenizer)
    texts: Dict[str, List[str]] = {
        name: [] for name in (model.heads if model else label_decodings)
    }
    conditioning = model.language if model else None
    if label_decodings and any(
        node.name == "language_vector" for node in enc_sess.get_inputs()
    ):
        conditioning = LanguageEmbedding(model_cfg.language_conditioning)
    language = model_cfg.get("onnx_default_language") if language is None else language
    offset = 0
    for wavs, wav_lens in tqdm(loader, desc="ASR inference", disable=not progress):
        features, lengths = preprocessor(
            wavs.to(device=device, dtype=torch.float32), wav_lens
        )
        features = features.to(feature_dtype)
        values = [
            features,
            lengths.cpu().numpy().astype(np.int64, copy=False),
        ]
        if manifest:
            context_ms = (
                manifest["future_context_ms"]
                if future_context_ms is None
                else future_context_ms
            )
            if streaming is False or not context_ms:
                chunk = -1
            else:
                allowed = {
                    int(c.future_context_ms) for c in model_cfg.streaming.contexts
                }
                if context_ms not in allowed:
                    raise ValueError(f"Unsupported future context: {context_ms}")
                chunk = int(context_ms / manifest["frame_shift_ms"])
            values.append(np.array([chunk], dtype=np.int64))
        if conditioning is not None:
            tags = (
                language[offset : offset + len(wavs)]
                if isinstance(language, (list, tuple))
                else language
            )
            values.append(
                conditioning.encode_tags(tags, len(wavs))
                .numpy()
                .astype(_session_float_dtype(enc_sess, "language_vector"), copy=False)
            )
        encoded_outputs = _run(enc_sess, values)
        offset += len(wavs)
        if label_decodings:
            lengths, labels = encoded_outputs
            if advance is not None:
                (tokens,) = _run(advance, [labels, lengths])
                for name, rows in zip(model_cfg.onnx_fc, tokens.transpose(1, 2, 0)):
                    texts[name].extend(
                        label_decodings[name].tokenizer.decode(row[row >= 0].tolist())
                        for row in rows
                    )
            else:
                lengths = torch.from_numpy(lengths)
                for name, values in zip(model_cfg.onnx_fc, labels):
                    decoded = label_decodings[name].decode_labels(
                        torch.from_numpy(values), lengths
                    )
                    texts[name].extend(row[0] for row in decoded)
            continue
        if model is None:
            outputs.extend(list(encoded_outputs[0]))
            continue
        encoded, lengths = encoded_outputs
        if packet_decoder is not None:
            names = manifest["packet_heads"]
            previous = np.repeat(
                [[model.decodings[name].blank_id] for name in names], len(wavs), axis=1
            )
            width = model_cfg.heads[names[0]].decoder.pred_hidden
            state = np.zeros((*previous.shape, width), dtype=np.float32)
            tokens, *_ = _run(
                packet_decoder, [encoded, lengths, previous, state, state]
            )
            for name, rows in zip(names, tokens.transpose(1, 2, 0)):
                texts[name].extend(
                    model.decodings[name].tokenizer.decode(row[row >= 0].tolist())
                    for row in rows
                )
            continue
        for name, decoding in model.decodings.items():
            decoded = decoding.decode(
                model.heads[name], torch.from_numpy(encoded), torch.from_numpy(lengths)
            )
            texts[name].extend(value[0] for value in decoded)
    return outputs if representations else (texts if dual else texts["head"])


def _providers_list(
    provider: Optional[str], *, use_tf32=True
) -> List[Union[str, Tuple[str, dict]]]:
    if provider == "CPUExecutionProvider":
        return ["CPUExecutionProvider"]
    if provider is not None and provider != "CUDAExecutionProvider":
        if provider not in rt.get_available_providers():
            raise RuntimeError(
                f"Required ONNX Runtime provider unavailable: {provider}"
            )
        return [provider]
    cuda_opts = {"cudnn_conv_algo_search": "HEURISTIC"}
    if not use_tf32:
        cuda_opts["use_tf32"] = "0"
    if "CUDAExecutionProvider" in rt.get_available_providers():
        return [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_onnx(
    onnx_dir,
    model_version=None,
    provider=None,
    *,
    streaming=None,
    session_factory=None,
    num_threads=None,
):
    """
    Load a GigaAM model from ONNX Runtime given a model version.
    Supports any family of models (ASR, Emo, SSL).
    CPU sessions use eight threads unless ``num_threads`` is set.
    """
    directory = Path(onnx_dir)
    manifest_path = directory / f"{model_version}.streaming.json"
    if model_version is None:
        manifest_path = directory / "streaming.json"
    manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    )
    if manifest is not None and model_version is None:
        model_version = manifest.get("model_name", model_version)
    cfg = omegaconf.OmegaConf.load(directory / f"{model_version}.yaml")
    if manifest is not None and provider is None:
        provider = "CPUExecutionProvider"
    streaming_ctc = manifest is not None and cfg.model_name.startswith("v4_ctc_")
    providers = (
        []
        if session_factory is not None
        else _providers_list(provider, use_tf32=not streaming_ctc)
    )
    options = rt.SessionOptions()
    options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = (
        num_threads
        if num_threads is not None
        else (8 if providers == ["CPUExecutionProvider"] else (2 if manifest else 1))
    )
    options.inter_op_num_threads = 1
    options.log_severity_level = 3

    def create(filename, *, whole=False):
        path = directory / filename
        if session_factory is not None:
            return session_factory(path)
        session = rt.InferenceSession(
            str(path),
            providers=(
                _providers_list(provider) if whole and streaming_ctc else providers
            ),
            sess_options=options,
        )
        if provider is not None:
            session.disable_fallback()
        return session

    if cfg.get("heads") or manifest:
        heads, decodings = cfg.heads, cfg.decodings
        for config in decodings.values():
            if config.get("model_path"):
                config.model_path = str(directory / Path(config.model_path).name)
        if manifest is None and cfg.get("onnx_fc"):
            sessions = {
                "encoder": create(f"{model_version}_encoder.onnx"),
                "fc": hydra.utils.instantiate(decodings, _convert_="all"),
            }
            if heads[cfg.onnx_fc[0]].get("joint"):
                sessions["advance"] = create(f"{model_version}_advance.onnx")
            return sessions, cfg
        sessions = {
            "heads": {
                name: [
                    create(f"{model_version}{'_' + name if name else ''}_{part}.onnx")
                    for part in (
                        ("decoder", "joint") if head.get("joint") else ("ctc",)
                    )
                ]
                for name, head in heads.items()
            }
        }
        sessions["encoder"] = create(
            manifest["step"]["file"] if manifest else f"{model_version}_encoder.onnx"
        )
        if manifest:
            front = manifest["frontend"]
            sessions["frontend"] = create(front["file"])
            sessions["manifest"] = manifest
            if manifest.get("whole"):
                sessions["whole"] = create(
                    manifest["whole"]["file"], whole=bool(cfg.get("onnx_fc"))
                )
                if cfg.get("onnx_fc"):
                    sessions["fc"] = hydra.utils.instantiate(decodings, _convert_="all")
                    if heads[cfg.onnx_fc[0]].get("joint"):
                        filename = manifest.get("advance", {}).get(
                            "file", f"{model_version}_advance.onnx"
                        )
                        sessions["advance"] = create(filename)
                else:
                    sessions["whole_model"] = _Model(cfg, sessions)
            if "packet_decode" in manifest:
                sessions["packet_decode"] = create(manifest["packet_decode"])
        return sessions, cfg
    parts = (
        ("_encoder", "_decoder", "_joint")
        if "rnnt" in model_version
        else (("_encoder",) if "ssl" in model_version else ("",))
    )
    return [create(f"{model_version}{part}.onnx") for part in parts], cfg


def _run(session, values):
    if isinstance(values[0], torch.Tensor) and values[0].is_cuda:
        if getattr(session, "cuda_device", None) is not None:
            return session.run(None, _build_inputs(session, values))
        if (
            hasattr(session, "io_binding")
            and "CUDAExecutionProvider" in session.get_providers()
        ):
            return _run_cuda(session, values)
        values = [values[0].cpu(), *values[1:]]
    return session.run(
        None, _build_inputs(session, [np.ascontiguousarray(value) for value in values])
    )


def _run_cuda(session, values):
    features, *cpu_values = values
    features = features.contiguous()
    first, *inputs = session.get_inputs()
    binding = session.io_binding()
    binding.bind_input(
        first.name,
        "cuda",
        features.get_device(),
        _session_float_dtype(session),
        features.shape,
        features.data_ptr(),
    )
    for node, value in zip(inputs, cpu_values, strict=True):
        binding.bind_cpu_input(node.name, np.ascontiguousarray(value))
    for node in session.get_outputs():
        binding.bind_output(node.name)
    torch.cuda.current_stream(features.device).synchronize()
    session.run_with_iobinding(binding)
    return binding.copy_outputs_to_cpu()


def _zeros(spec):
    return [np.zeros(item["shape"], dtype=item["dtype"]) for item in spec]


def _run_cached_cuda(session, values, caches, device):
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    binding = session.io_binding()
    for node, value in zip(inputs, values):
        binding.bind_cpu_input(node.name, np.ascontiguousarray(value))
    packed = [
        torch.cat([torch.as_tensor(value, device=device) for value in parts])
        for parts in zip(*caches)
    ]
    retained = [
        torch.empty_like(
            cache, dtype=getattr(torch, _session_float_dtype(session, node.name).name)
        )
        for node, cache in zip(outputs[4:], packed, strict=True)
    ]

    def bind_caches(bind, nodes, tensors):
        for node, cache in zip(nodes, tensors, strict=True):
            bind(
                node.name,
                "cuda",
                device.index,
                _session_float_dtype(session, node.name),
                tuple(cache.shape),
                cache.data_ptr(),
            )

    bind_caches(binding.bind_input, inputs[len(values) :], packed)
    for node in outputs[:4]:
        binding.bind_output(node.name)
    bind_caches(binding.bind_output, outputs[4:], retained)
    torch.cuda.current_stream(device).synchronize()
    session.run_with_iobinding(binding)
    return [*(value.numpy() for value in binding.get_outputs()[:4]), *retained]


@dataclass
class _State(EncoderStreamState):
    valid: int = 0
    caches: Union[list, Tuple[list, int]] = field(default_factory=list)
    frontend: list = field(default_factory=list)


class _Encoder(nn.Module):
    def __init__(self, manifest, sessions, streaming=True):
        super().__init__()
        dtype = torch.from_numpy(
            np.empty((), _session_float_dtype(sessions["encoder"]))
        ).dtype
        self.anchor = nn.Parameter(torch.empty(0, dtype=dtype), requires_grad=False)
        self.manifest, self.sessions = manifest, sessions
        self.subsampling_factor = manifest["subsampling_factor"]
        self.frame_shift_ms = manifest["frame_shift_ms"]
        self.chunk_size = int(
            manifest["future_context_ms"] / manifest["frame_shift_ms"]
        )
        self.dynamic_context = any(
            node.name == "context_frames" for node in sessions["encoder"].get_inputs()
        )
        self.streaming = streaming
        self.conditioned = any(
            node.name == "language_vector" for node in sessions["encoder"].get_inputs()
        )
        session = sessions["encoder"]
        self.cache_device = getattr(session, "cuda_device", None)
        if isinstance(session, rt.InferenceSession):
            options = session.get_provider_options().get("CUDAExecutionProvider")
            if options:
                self.cache_device = torch.device("cuda", int(options["device_id"]))

    def create_stream_state(self, chunk, left):
        return _State()

    def forward(self, signal, lengths, state, final=False, lang_embedding=None):
        encoded, valid = self.forward_many([(signal, state, final, lang_embedding)])
        return encoded[:, :, : valid[0]], lengths.new_tensor(valid), state

    def forward_many(self, packets):
        manifest = self.manifest
        chunk = self.chunk_size
        size = self.subsampling_factor * chunk if self.streaming else None
        ready = [
            state.take_ready(signal, size, final, axis=2)
            for signal, state, final, _ in packets
        ]
        rows = [i for i, (_, count) in enumerate(ready) if count]
        lengths = np.zeros(len(packets), dtype=np.int64)
        if not rows:
            return torch.empty(len(packets), manifest["encoder_dim"], 0), lengths
        states = [packets[i][1] for i in rows]
        counts = np.array([ready[i][1] for i in rows], dtype=np.int64)
        multiple = size or self.subsampling_factor
        physical = (int(counts.max()) + multiple - 1) // multiple * multiple
        mel = torch.cat([F.pad(ready[i][0], (0, physical - ready[i][1])) for i in rows])
        frontend_spec = manifest["frontend"]["states"]
        cache_spec = manifest["states"]
        if manifest.get("ctc_labels"):
            values, next_front = [mel, counts], []
            cache_spec = frontend_spec + cache_spec
        else:
            front = [state.frontend or _zeros(frontend_spec) for state in states]
            features, valid, *next_front = _run(
                self.sessions["frontend"],
                [mel, counts, *(np.concatenate(parts) for parts in zip(*front))],
            )
            values = [features, valid]
        if self.dynamic_context:
            values.append(np.array([chunk if self.streaming else -1], dtype=np.int64))
        values.extend(
            np.array([getattr(s, key) for s in states], dtype=np.int64)
            for key in ("position", "valid")
        )
        if self.conditioned:
            values.append(torch.cat([packets[i][3] for i in rows]))
        caches = [state.caches or _zeros(cache_spec) for state in states]
        session = self.sessions["encoder"]
        if hasattr(session, "run_cached"):
            result = session.run_cached(values, caches)
        elif self.cache_device is not None:
            result = _run_cached_cuda(session, values, caches, self.cache_device)
        else:
            result = _run(
                session, [*values, *(np.concatenate(parts) for parts in zip(*caches))]
            )
        encoded, valid, position, cache_valid, *next_caches = result
        output = np.zeros((len(packets), *encoded.shape[1:]), dtype=encoded.dtype)
        output[rows], lengths[rows] = encoded, valid
        for i, state in enumerate(states):
            state.frontend = [cache[i : i + 1].copy() for cache in next_front]
            if manifest.get("ctc_labels") and hasattr(session, "run_cached"):
                state.caches = (next_caches, i)
            else:
                state.caches = [
                    (
                        cache[i : i + 1].to(dtype=getattr(torch, spec["dtype"])).clone()
                        if self.cache_device is not None
                        else cache[i : i + 1].astype(spec["dtype"], copy=True)
                    )
                    for cache, spec in zip(next_caches, cache_spec, strict=True)
                ]
            state.position, state.valid = int(position[i]), int(cache_valid[i])
        return torch.from_numpy(output), lengths


class _CTC(CTCHead):
    def __init__(self, session, label_index=None):
        nn.Module.__init__(self)
        self.session = session
        self.label_index = label_index

    def forward(self, encoder_output):
        if self.label_index is not None:
            return encoder_output[:, self.label_index]
        return (
            encoder_output
            if self.session is None
            else torch.from_numpy(_run(self.session, [encoder_output])[0])
        )


class _Predictor(nn.Module):
    def __init__(self, session, cfg, blank):
        super().__init__()
        self.session, self.cfg, self.blank_id = session, cfg, blank

    def predict(self, label, state=None, batch_size=1):
        if label is None:
            label = torch.full((batch_size, 1), self.blank_id, dtype=torch.long)
        if state is None:
            shape = (
                self.cfg.decoder.pred_rnn_layers,
                batch_size,
                self.cfg.decoder.pred_hidden,
            )
            state = [
                np.zeros(shape, dtype=_session_float_dtype(self.session))
                for _ in range(2)
            ]
        decoded, h, c = _run(self.session, [label, *state])
        return torch.from_numpy(decoded), (torch.from_numpy(h), torch.from_numpy(c))


class _Joint(nn.Module):
    def __init__(self, session):
        super().__init__()
        self.session = session
        shape = getattr(session.get_inputs()[0], "shape", None)
        self.supports_frame_windows = shape is not None and not isinstance(
            shape[2], int
        )

    def joint(self, encoded, prediction):
        return torch.from_numpy(
            _run(self.session, [encoded.transpose(1, 2), prediction.transpose(1, 2)])[0]
        )


class _RNNT(RNNTHead):
    def __init__(self, sessions, cfg, blank):
        nn.Module.__init__(self)
        self.decoder = _Predictor(sessions[0], cfg, blank)
        self.joint = _Joint(sessions[1])


class _Model(nn.Module):
    def __init__(self, cfg, sessions, streaming=True, tokenizer=None):
        super().__init__()
        self.cfg = cfg
        dual = bool(cfg.get("heads"))
        encoder = sessions["encoder"] if isinstance(sessions, dict) else sessions[0]
        if isinstance(sessions, dict) and "manifest" in sessions:
            self.encoder = _Encoder(sessions["manifest"], sessions, streaming)
            self.preprocessor = hydra.utils.instantiate(cfg.preprocessor)
        self.language_fusion = None
        self.language = (
            LanguageEmbedding(cfg.language_conditioning)
            if any(node.name == "language_vector" for node in encoder.get_inputs())
            else None
        )
        self._device = torch.device("cpu")
        heads = cfg.heads if dual else {"head": cfg.get("head")}
        decodings = cfg.decodings if dual else {"head": cfg.decoding}
        self.decodings = {
            name: (
                hydra.utils.instantiate(config, model_path=None)
                if not dual and tokenizer is not None
                else hydra.utils.instantiate(config)
            )
            for name, config in decodings.items()
        }
        self.heads = nn.ModuleDict()
        for name, config in heads.items():
            decoding = self.decodings[name]
            if not dual and tokenizer is not None:
                decoding.tokenizer = tokenizer
                decoding.blank_id = len(tokenizer)
            parts = sessions["heads"][name] if dual else sessions[1:]
            if config and config.get("joint"):
                if not dual:
                    decoding.max_symbols = MAX_LETTERS_PER_FRAME
                self.heads[name] = _RNNT(parts, config, decoding.blank_id)
            else:
                label_heads = (
                    sessions.get("manifest", {}).get("ctc_labels", ()) if dual else ()
                )
                index = label_heads.index(name) if label_heads else None
                self.heads[name] = _CTC(parts[0] if dual else None, index)

    def encode_language(self, language, batch_size):
        return (
            self.language.encode_tags(language, batch_size)
            if self.language is not None
            else None
        )


class OnnxGigaAMStreaming(GigaAMStreaming):
    def __init__(
        self,
        cfg,
        sessions,
        language=None,
        future_context_ms=None,
        *,
        streaming=True,
        word_timestamps=False,
        _model=None,
    ):
        nn.Module.__init__(self)
        inputs = {node.name for node in sessions["encoder"].get_inputs()}
        context = sessions["manifest"]["future_context_ms"]
        if "context_frames" not in inputs and (
            not streaming or future_context_ms not in (None, context)
        ):
            raise ValueError(
                "Fixed-context ONNX streaming supports only its exported context."
            )
        self._streaming = streaming
        super().__init__(
            _model if _model is not None else _Model(cfg, sessions, streaming),
            (
                sessions["manifest"]["future_context_ms"]
                if future_context_ms is None
                else future_context_ms
            ),
            cfg.get("onnx_default_language") if language is None else language,
            word_timestamps,
        )

    @property
    def streaming(self):
        return self._streaming

    def _configure_context(self, value):
        encoder = self.model.encoder
        if (
            not encoder.dynamic_context
            and value != encoder.manifest["future_context_ms"]
        ):
            raise ValueError(
                "Fixed-context ONNX streaming supports only its exported context."
            )
        super()._configure_context(value)

    @property
    def future_context_ms(self):
        return super().future_context_ms if self.streaming else None
