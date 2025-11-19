import warnings
from typing import List, Optional, Tuple, Union

import hydra
import numpy as np
import omegaconf
import onnxruntime as rt
import torch

from .decoding import Tokenizer
from .preprocess import FeatureExtractor, load_audio

warnings.simplefilter("ignore", category=UserWarning)


DTYPE = np.float32
MAX_LETTERS_PER_FRAME = 3


def infer_onnx(
    wav_file: str,
    model_cfg: omegaconf.DictConfig,
    sessions: List[rt.InferenceSession],
    preprocessor: Optional[FeatureExtractor] = None,
    tokenizer: Optional[Tokenizer] = None,
) -> Union[str, np.ndarray]:
    """Run ONNX sessions for the model, requires preprocessor instantiating"""
    model_name = model_cfg.model_name

    if preprocessor is None:
        preprocessor = hydra.utils.instantiate(model_cfg.preprocessor)
    if tokenizer is None and ("ctc" in model_name or "rnnt" in model_name):
        tokenizer = hydra.utils.instantiate(model_cfg.decoding).tokenizer

    input_signal = load_audio(wav_file)
    input_signal = preprocessor(
        input_signal.unsqueeze(0), torch.tensor([input_signal.shape[-1]])
    )[0].numpy()

    enc_sess = sessions[0]
    enc_inputs = {
        node.name: data
        for (node, data) in zip(
            enc_sess.get_inputs(),
            [input_signal.astype(DTYPE), [input_signal.shape[-1]]],
        )
    }
    enc_features = enc_sess.run(
        [node.name for node in enc_sess.get_outputs()], enc_inputs
    )[0]

    if "emo" in model_name or "ssl" in model_name:
        return enc_features

    blank_idx = len(tokenizer)
    token_ids = []
    prev_token = blank_idx
    if "ctc" in model_name:
        prev_tok = blank_idx
        for tok in enc_features.argmax(-1).squeeze().tolist():
            if (tok != prev_tok or prev_tok == blank_idx) and tok != blank_idx:
                token_ids.append(tok)
            prev_tok = tok
    else:
        pred_states = [
            np.zeros(shape=(1, 1, model_cfg.head.decoder.pred_hidden), dtype=DTYPE),
            np.zeros(shape=(1, 1, model_cfg.head.decoder.pred_hidden), dtype=DTYPE),
        ]
        pred_sess, joint_sess = sessions[1:]
        for j in range(enc_features.shape[-1]):
            emitted_letters = 0
            while emitted_letters < MAX_LETTERS_PER_FRAME:
                pred_inputs = {
                    node.name: data
                    for (node, data) in zip(
                        pred_sess.get_inputs(), [np.array([[prev_token]])] + pred_states
                    )
                }
                pred_outputs = pred_sess.run(
                    [node.name for node in pred_sess.get_outputs()], pred_inputs
                )

                joint_inputs = {
                    node.name: data
                    for node, data in zip(
                        joint_sess.get_inputs(),
                        [enc_features[:, :, [j]], pred_outputs[0].swapaxes(1, 2)],
                    )
                }
                log_probs = joint_sess.run(
                    [node.name for node in joint_sess.get_outputs()], joint_inputs
                )
                token = log_probs[0].argmax(-1)[0][0]

                if token != blank_idx:
                    prev_token = int(token)
                    pred_states = pred_outputs[1:]
                    token_ids.append(int(token))
                    emitted_letters += 1
                else:
                    break

    return tokenizer.decode(token_ids)


def load_onnx(
    onnx_dir: str,
    model_version: str,
    provider: Optional[str] = None,
) -> Tuple[
    List[rt.InferenceSession], Union[omegaconf.DictConfig, omegaconf.ListConfig]
]:
    """Load ONNX sessions for the given versions and cpu / cuda provider"""
    if provider is None and "CUDAExecutionProvider" in rt.get_available_providers():
        provider = "CUDAExecutionProvider"
    elif provider is None:
        provider = "CPUExecutionProvider"

    opts = rt.SessionOptions()
    opts.intra_op_num_threads = 16
    opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    opts.log_severity_level = 3

    model_cfg = omegaconf.OmegaConf.load(f"{onnx_dir}/{model_version}.yaml")

    if "rnnt" not in model_version and "ssl" not in model_version:
        model_path = f"{onnx_dir}/{model_version}.onnx"
        sessions = [
            rt.InferenceSession(model_path, providers=[provider], sess_options=opts)
        ]
    elif "ssl" in model_version:
        pth = f"{onnx_dir}/{model_version}"
        enc_sess = rt.InferenceSession(
            f"{pth}_encoder.onnx", providers=[provider], sess_options=opts
        )
        sessions = [enc_sess]
    else:
        pth = f"{onnx_dir}/{model_version}"
        enc_sess = rt.InferenceSession(
            f"{pth}_encoder.onnx", providers=[provider], sess_options=opts
        )
        pred_sess = rt.InferenceSession(
            f"{pth}_decoder.onnx", providers=[provider], sess_options=opts
        )
        joint_sess = rt.InferenceSession(
            f"{pth}_joint.onnx", providers=[provider], sess_options=opts
        )
        sessions = [enc_sess, pred_sess, joint_sess]

    return sessions, model_cfg
