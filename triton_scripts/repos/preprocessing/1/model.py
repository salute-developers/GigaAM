import os
import sys
import warnings
from typing import Any, Dict, List

import numpy as np
import omegaconf
import torch
from torch import Tensor

from gigaam.preprocess import FeatureExtractor

warnings.simplefilter("ignore", category=UserWarning)


SAMPLE_RATE = 16000


class TritonPythonModel:
    def initialize(self, args: Dict[str, Any]) -> None:
        model_version = args["model_version"]
        model_repository = args["model_repository"]

        config_path = os.path.join(model_repository, model_version, "config.yaml")

        cfg = omegaconf.OmegaConf.load(config_path)

        # Check if model is v3 and warn if not
        if not cfg.model_name.startswith("v3"):
            sys.stderr.write(
                f"Model '{cfg.model_name}' does not belong to 'v3' family. "
                "Using old feature extraction version "
                "(incompatible with v3 models)."
            )
            sys.stderr.flush()

        preprocessor_dict = omegaconf.OmegaConf.to_container(
            cfg.preprocessor, resolve=True
        )
        preprocessor_dict.pop("_target_", None)
        self.preprocessor = FeatureExtractor(**preprocessor_dict)
        self.preprocessor.eval()

    def execute(self, requests: Any) -> List[Any]:
        import triton_python_backend_utils as pb_utils  # type: ignore

        responses: List[Any] = []

        for request in requests:
            audio_batch = pb_utils.get_input_tensor_by_name(request, "audio_batch")
            audio_lengths = pb_utils.get_input_tensor_by_name(request, "audio_lengths")

            audio_batch_np = audio_batch.as_numpy()
            audio_lengths_np = audio_lengths.as_numpy()

            audio_tensors: List[Tensor] = []
            start_idx = 0
            for length in audio_lengths_np:
                length_int = int(length)
                audio_tensors.append(
                    torch.from_numpy(
                        audio_batch_np[start_idx : start_idx + length_int]
                    ).float()
                )
                start_idx += length_int

            batch_audio = torch.nn.utils.rnn.pad_sequence(
                audio_tensors, batch_first=True
            )
            batch_lengths = torch.tensor(audio_lengths_np, dtype=torch.long)

            with torch.no_grad():
                features, feature_lengths = self.preprocessor(
                    batch_audio, batch_lengths
                )

            features_np = features.detach().cpu().numpy().astype(np.float16)
            feature_lengths_np = feature_lengths.detach().cpu().numpy().astype(np.int64)

            output_tensors = [
                pb_utils.Tensor("features", features_np),
                pb_utils.Tensor("feature_lengths", feature_lengths_np),
            ]

            response = pb_utils.InferenceResponse(output_tensors=output_tensors)
            responses.append(response)

        return responses
