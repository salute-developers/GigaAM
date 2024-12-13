import warnings
from typing import List, Optional

import numpy as np
import onnxruntime as rt
import torch

warnings.simplefilter("ignore", category=UserWarning)

import gigaam

D_MODEL = 768
DTYPE = np.float32
MAX_LETTERS_PER_FRAME = 3
SAMPLE_RATE = 16000
FEAT_IN = 64
PRED_HIDDEN = 320
BLANK_IDX = 33
VOCAB = [
    " ",
    "а",
    "б",
    "в",
    "г",
    "д",
    "е",
    "ж",
    "з",
    "и",
    "й",
    "к",
    "л",
    "м",
    "н",
    "о",
    "п",
    "р",
    "с",
    "т",
    "у",
    "ф",
    "х",
    "ц",
    "ч",
    "ш",
    "щ",
    "ъ",
    "ы",
    "ь",
    "э",
    "ю",
    "я",
]


def transcribe_sample(
    wav_file: str,
    model_type: str,
    sessions: List[rt.InferenceSession],
    preprocessor: Optional[gigaam.preprocess.FeatureExtractor] = None,
) -> str:
    if preprocessor is None:
        preprocessor = gigaam.preprocess.FeatureExtractor(SAMPLE_RATE, FEAT_IN)

    assert model_type in ["ctc", "rnnt"], "Only `ctc` and `rnnt` inference supported"

    input_signal = gigaam.load_audio(wav_file)
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

    token_ids = []
    prev_token = BLANK_IDX
    if model_type == "ctc":
        prev_tok = BLANK_IDX
        for tok in enc_features.argmax(-1).squeeze().tolist():
            if (tok != prev_tok or prev_tok == BLANK_IDX) and tok != BLANK_IDX:
                token_ids.append(tok)
            prev_tok = tok
    else:
        pred_states = [
            np.zeros(shape=(1, 1, PRED_HIDDEN), dtype=DTYPE),
            np.zeros(shape=(1, 1, PRED_HIDDEN), dtype=DTYPE),
        ]
        pred_sess, joint_sess = sessions[1:]
        for j in range(enc_features.shape[-1]):
            emitted_letters = 0
            while emitted_letters < MAX_LETTERS_PER_FRAME:
                pred_inputs = {
                    node.name: data
                    for (node, data) in zip(
                        pred_sess.get_inputs(), [[[prev_token]]] + pred_states
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

                if token != BLANK_IDX:
                    prev_token = int(token)
                    pred_states = pred_outputs[1:]
                    token_ids.append(int(token))
                    emitted_letters += 1
                else:
                    break

    return "".join(VOCAB[tok] for tok in token_ids)


def load_onnx_sessions(
    onnx_dir: str,
    model_type: str,
    model_version: Optional[str] = None,
) -> List[rt.InferenceSession]:
    if model_version is None:
        model_version = "v2"

    opts = rt.SessionOptions()
    opts.intra_op_num_threads = 16
    opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

    if model_type == "ctc":
        model_path = f"{onnx_dir}/{model_version}_{model_type}.onnx"
        sessions = [
            rt.InferenceSession(
                model_path, providers=["CPUExecutionProvider"], sess_options=opts
            )
        ]
    else:
        pth = f"{onnx_dir}/{model_version}_{model_type}"
        enc_sess = rt.InferenceSession(
            f"{pth}_encoder.onnx", providers=["CPUExecutionProvider"], sess_options=opts
        )
        pred_sess = rt.InferenceSession(
            f"{pth}_decoder.onnx", providers=["CPUExecutionProvider"], sess_options=opts
        )
        joint_sess = rt.InferenceSession(
            f"{pth}_joint.onnx", providers=["CPUExecutionProvider"], sess_options=opts
        )
        sessions = [enc_sess, pred_sess, joint_sess]

    return sessions
