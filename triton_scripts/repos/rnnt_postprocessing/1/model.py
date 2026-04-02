import os
from typing import Any, Dict, List

import numpy as np
import omegaconf
import onnxruntime as rt

from gigaam.decoding import Tokenizer
from gigaam.onnx_utils import infer_onnx


class TritonPythonModel:
    def initialize(self, args: Dict[str, Any]) -> None:
        model_version = args["model_version"]
        model_repository = args["model_repository"]

        config_path = os.path.join(model_repository, model_version, "config.yaml")

        if os.path.exists(config_path):
            cfg = omegaconf.OmegaConf.load(config_path)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

        model_name = cfg.model_name
        vocab = cfg.decoding.get("vocabulary")

        if cfg.decoding.get("model_path"):
            tokenizer_path = os.path.join(
                model_repository,
                model_version,
                f"{model_name}_tokenizer.model",
            )
        else:
            tokenizer_path = None

        self.cfg = cfg
        self.tokenizer = Tokenizer(vocab=vocab, model_path=tokenizer_path)

        # Load ONNX models
        decoder_path = os.path.join(
            model_repository, model_version, f"{model_name}_decoder.onnx"
        )
        joint_path = os.path.join(
            model_repository, model_version, f"{model_name}_joint.onnx"
        )

        if not os.path.exists(decoder_path):
            raise FileNotFoundError(f"Decoder ONNX model not found: {decoder_path}")
        if not os.path.exists(joint_path):
            raise FileNotFoundError(f"Joint ONNX model not found: {joint_path}")

        available_providers = rt.get_available_providers()
        provider = (
            "CUDAExecutionProvider"
            if "CUDAExecutionProvider" in available_providers
            else "CPUExecutionProvider"
        )
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = 16
        opts.log_severity_level = 3

        self.pred_sess = rt.InferenceSession(
            decoder_path,
            providers=[provider],
            sess_options=opts,
        )
        self.joint_sess = rt.InferenceSession(
            joint_path,
            providers=[provider],
            sess_options=opts,
        )

        self.pred_hidden = cfg.head.decoder.pred_hidden

    def execute(self, requests: Any) -> List[Any]:
        import triton_python_backend_utils as pb_utils  # type: ignore

        responses: List[Any] = []

        for request in requests:
            encoded = pb_utils.get_input_tensor_by_name(request, "encoded")
            encoded_lengths = pb_utils.get_input_tensor_by_name(
                request, "encoded_lengths"
            )

            encoded_np = encoded.as_numpy()
            encoded_lengths_np = encoded_lengths.as_numpy()

            texts = [
                infer_onnx(
                    wav_file=None,
                    model_cfg=self.cfg,
                    sessions=[None, self.pred_sess, self.joint_sess],
                    enc_features=encoded_np[i : i + 1, :, : encoded_lengths_np[i]],
                    tokenizer=self.tokenizer,
                )
                for i in range(encoded_np.shape[0])
            ]

            texts_bytes = [text.encode("utf-8") for text in texts]
            texts_array = np.array(texts_bytes, dtype=object)

            output_tensors = [
                pb_utils.Tensor("texts", texts_array),
            ]
            response = pb_utils.InferenceResponse(output_tensors=output_tensors)
            responses.append(response)

        return responses
