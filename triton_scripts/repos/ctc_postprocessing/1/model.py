import os
from typing import Any, Dict, List

import numpy as np
import omegaconf
import torch

from gigaam.decoding import CTCGreedyDecoding


class TritonPythonModel:
    def initialize(self, args: Dict[str, Any]) -> None:
        model_version = args["model_version"]
        model_repository = args["model_repository"]

        config_path = os.path.join(model_repository, model_version, "config.yaml")

        if os.path.exists(config_path):
            cfg = omegaconf.OmegaConf.load(config_path)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

        vocab = cfg.decoding.get("vocabulary")
        if cfg.decoding.get("model_path"):
            tokenizer_path = os.path.join(
                model_repository,
                model_version,
                f"{cfg.model_name}_tokenizer.model",
            )
        else:
            tokenizer_path = None

        self.decoding = CTCGreedyDecoding(vocabulary=vocab, model_path=tokenizer_path)

    def execute(self, requests: Any) -> List[Any]:
        import triton_python_backend_utils as pb_utils  # type: ignore

        responses: List[Any] = []

        for request in requests:
            token_ids = pb_utils.get_input_tensor_by_name(request, "token_ids")
            token_ids_lengths = pb_utils.get_input_tensor_by_name(
                request, "token_ids_lengths"
            )

            token_ids_np = token_ids.as_numpy()
            token_ids_lengths_np = token_ids_lengths.as_numpy()

            results = self.decoding.decode(
                head=None,
                encoded=None,
                lengths=torch.from_numpy(token_ids_lengths_np),
                labels=torch.from_numpy(token_ids_np),
            )
            texts = [self.decoding.tokenizer.decode(result[0]) for result in results]

            texts_bytes = [text.encode("utf-8") for text in texts]
            texts_array = np.array(texts_bytes, dtype=object)

            output_tensors = [
                pb_utils.Tensor("texts", texts_array),
            ]
            response = pb_utils.InferenceResponse(output_tensors=output_tensors)
            responses.append(response)

        return responses
