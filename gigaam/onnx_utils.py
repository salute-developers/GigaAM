import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import hydra
import numpy as np
import omegaconf
import onnxruntime as rt
import torch
from tqdm.auto import tqdm

from .decoding import Tokenizer
from .preprocess import FeatureExtractor
from .utils import AudioDataset

warnings.simplefilter("ignore", category=UserWarning)

MAX_LETTERS_PER_FRAME = 3


def _session_float_dtype(session: rt.InferenceSession) -> np.dtype:
    """Infer numpy float dtype from the first float input of an ONNX session."""
    _type_map: Dict[str, np.dtype] = {
        "tensor(float16)": np.dtype(np.float16),
        "tensor(float)": np.dtype(np.float32),
        "tensor(double)": np.dtype(np.float64),
    }
    for inp in session.get_inputs():
        if inp.type in _type_map:
            return _type_map[inp.type]
    return np.dtype(np.float32)


def _build_inputs(session: rt.InferenceSession, values: List[np.ndarray]) -> dict:
    return {node.name: data for node, data in zip(session.get_inputs(), values)}


def _decode_ctc_batch(
    labels: np.ndarray,
    lengths: np.ndarray,
    tokenizer: Tokenizer,
) -> List[str]:
    blank_id = len(tokenizer)
    b, t = labels.shape
    lengths = np.clip(np.asarray(lengths, dtype=np.int64).reshape(-1), 0, t)

    skip_mask = labels != blank_id
    skip_mask[:, 1:] &= labels[:, 1:] != labels[:, :-1]

    time = np.arange(t, dtype=np.int64)[None, :]
    skip_mask &= time < lengths[:, None]

    return [tokenizer.decode(labels[i][skip_mask[i]].tolist()) for i in range(b)]


def _cat_states(
    states: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    hs = [s[0] for s in states]
    cs = [s[1] for s in states]
    return np.concatenate(hs, axis=1), np.concatenate(cs, axis=1)


def _split_state(
    state: Tuple[np.ndarray, np.ndarray],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    h, c = state
    b = h.shape[1]
    return [(h[:, i : i + 1], c[:, i : i + 1]) for i in range(b)]


def _decode_rnnt_batch(
    enc_features: np.ndarray,
    enc_len: np.ndarray,
    model_cfg: omegaconf.DictConfig,
    sessions: List[Optional[rt.InferenceSession]],
    tokenizer: Tokenizer,
) -> List[str]:
    pred_sess, joint_sess = sessions[1:]
    dtype = _session_float_dtype(pred_sess)

    enc_features = np.asarray(enc_features, dtype=dtype, order="C")
    blank_idx = len(tokenizer)
    pred_hidden = model_cfg.head.decoder.pred_hidden
    pred_rnn_layers = model_cfg.head.decoder.pred_rnn_layers
    B, _, T = enc_features.shape

    hyps: List[List[int]] = [[] for _ in range(B)]
    last_label: List[Optional[np.ndarray]] = [None] * B
    dec_state: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None] * B

    def emit_batch(batch_idx: List[int], t: int, fresh: bool) -> List[int]:
        idx = np.asarray(batch_idx, dtype=np.int64)
        f = enc_features[idx, :, t : t + 1]

        if fresh:
            labels = np.full((len(batch_idx), 1), blank_idx, dtype=np.int64)
            h = np.zeros((pred_rnn_layers, len(batch_idx), pred_hidden), dtype=dtype)
            c = np.zeros((pred_rnn_layers, len(batch_idx), pred_hidden), dtype=dtype)
        else:
            labels = np.concatenate([last_label[i] for i in batch_idx], axis=0)
            h, c = _cat_states([dec_state[i] for i in batch_idx])

        pred_outputs = pred_sess.run(
            [node.name for node in pred_sess.get_outputs()],
            _build_inputs(pred_sess, [labels, h, c]),
        )

        joint_outputs = joint_sess.run(
            [node.name for node in joint_sess.get_outputs()],
            _build_inputs(
                joint_sess,
                [f, pred_outputs[0].swapaxes(1, 2)],
            ),
        )

        k = joint_outputs[0][:, 0, 0, :].argmax(axis=-1)
        emit_pos = np.nonzero(k != blank_idx)[0]
        if emit_pos.size == 0:
            return []

        hidden_parts = _split_state((pred_outputs[1], pred_outputs[2]))
        out = []

        for p in emit_pos.tolist():
            bi = batch_idx[p]
            tok = int(k[p])

            hyps[bi].append(tok)
            last_label[bi] = np.array([[tok]], dtype=np.int64)
            dec_state[bi] = hidden_parts[p]
            out.append(bi)

        return out

    enc_len = np.asarray(enc_len, dtype=np.int64).reshape(-1)
    for t in range(T):
        active = np.nonzero(t < enc_len)[0].tolist()
        if not active:
            break

        for _ in range(MAX_LETTERS_PER_FRAME):
            if not active:
                break

            fresh = [i for i in active if dec_state[i] is None]
            stateful = [i for i in active if dec_state[i] is not None]

            next_active = []
            if fresh:
                next_active.extend(emit_batch(fresh, t, fresh=True))
            if stateful:
                next_active.extend(emit_batch(stateful, t, fresh=False))

            if not next_active:
                break

            active = next_active

    return [tokenizer.decode(h) for h in hyps]


def infer_onnx(
    data: Union[str, Sequence[Union[str, np.ndarray, torch.Tensor]]],
    model_cfg: omegaconf.DictConfig,
    sessions: List[Optional[rt.InferenceSession]],
    preprocessor: Optional[FeatureExtractor] = None,
    tokenizer: Optional[Tokenizer] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    progress: bool = True,
) -> Union[List[str], np.ndarray, List[np.ndarray]]:
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

    Returns
    -------
    Union[List[str], np.ndarray, List[np.ndarray]]
        List of texts (ASR) / probs (Emo) / arrays (SSL) per sample.
    """
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
    if tokenizer is None and ("ctc" in model_name or "rnnt" in model_name):
        tokenizer = hydra.utils.instantiate(model_cfg.decoding).tokenizer

    if isinstance(data, str) and Path(data).suffix != ".tsv":
        data = [data]

    dataset = AudioDataset(data)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=AudioDataset.collate,
        num_workers=num_workers,
    )
    loader_iter = (
        tqdm(loader, desc="Inference")
        if progress and ("emo" in model_name or "ssl" in model_name)
        else loader
    )

    enc_sess = sessions[0]
    dtype = _session_float_dtype(enc_sess)

    if "emo" in model_name or "ssl" in model_name:
        outputs = []
        for wavs, wav_lens in loader_iter:
            input_signal, input_lengths = preprocessor(wavs.float(), wav_lens)
            batch_outputs = enc_sess.run(
                [node.name for node in enc_sess.get_outputs()],
                _build_inputs(
                    enc_sess,
                    [
                        input_signal.contiguous().numpy().astype(dtype),
                        input_lengths.numpy().astype(np.int64),
                    ],
                ),
            )
            outputs.extend(list(batch_outputs[0]))

        return outputs

    texts = []
    asr_iter = tqdm(loader, desc="ASR inference") if progress else loader
    for wavs, wav_lens in asr_iter:
        input_signal, input_lengths = preprocessor(wavs.float(), wav_lens)
        batch_outputs = enc_sess.run(
            [node.name for node in enc_sess.get_outputs()],
            _build_inputs(
                enc_sess,
                [
                    input_signal.contiguous().numpy().astype(dtype),
                    input_lengths.numpy().astype(np.int64),
                ],
            ),
        )

        batch_features = batch_outputs[0]
        assert (
            len(batch_outputs) > 1
        ), "encoder must return enc_lengths for batched decoding"
        batch_lengths = np.asarray(batch_outputs[1], dtype=np.int64).reshape(-1)

        if "ctc" in model_name:
            texts.extend(
                _decode_ctc_batch(batch_features.argmax(-1), batch_lengths, tokenizer)
            )
        else:
            texts.extend(
                _decode_rnnt_batch(
                    batch_features, batch_lengths, model_cfg, sessions, tokenizer
                )
            )

    return texts


def _providers_list(provider: Optional[str]) -> List[Union[str, Tuple[str, dict]]]:
    if provider == "CPUExecutionProvider":
        return ["CPUExecutionProvider"]
    if provider is not None and provider != "CUDAExecutionProvider":
        return [provider]
    cuda_opts = {"cudnn_conv_algo_search": "HEURISTIC"}
    if "CUDAExecutionProvider" in rt.get_available_providers():
        return [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_onnx(
    onnx_dir: str,
    model_version: str,
    provider: Optional[str] = None,
) -> Tuple[
    List[rt.InferenceSession], Union[omegaconf.DictConfig, omegaconf.ListConfig]
]:
    """
    Load a GigaAM model from ONNX Runtime given a model version.
    Supports any family of models (ASR, Emo, SSL).
    """
    providers = _providers_list(provider)

    opts = rt.SessionOptions()
    opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 8 if providers == ["CPUExecutionProvider"] else 1
    opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    opts.log_severity_level = 3

    model_cfg = omegaconf.OmegaConf.load(f"{onnx_dir}/{model_version}.yaml")

    def _sess(path: str) -> rt.InferenceSession:
        return rt.InferenceSession(path, providers=providers, sess_options=opts)

    if "rnnt" not in model_version and "ssl" not in model_version:
        model_path = f"{onnx_dir}/{model_version}.onnx"
        sessions = [_sess(model_path)]
    elif "ssl" in model_version:
        pth = f"{onnx_dir}/{model_version}"
        sessions = [_sess(f"{pth}_encoder.onnx")]
    else:
        pth = f"{onnx_dir}/{model_version}"
        sessions = [
            _sess(f"{pth}_encoder.onnx"),
            _sess(f"{pth}_decoder.onnx"),
            _sess(f"{pth}_joint.onnx"),
        ]

    return sessions, model_cfg
