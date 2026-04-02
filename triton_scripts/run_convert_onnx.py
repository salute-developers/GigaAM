import os
import shutil
import sys
import types
import warnings
from typing import Any, Tuple

import omegaconf
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gigaam  # noqa: E402
from gigaam.utils import onnx_converter  # noqa: E402


# We need to override the forward method to return the argmax of the logits
# to avoid variable output shapes in the ONNX / TRT and keep the unified config.
def forward_for_export_with_argmax(
    self: Any, features: Tensor, feature_lengths: Tensor
) -> Tuple[Tensor, Tensor]:
    encoded, encoded_len = self.encoder(features, feature_lengths)
    logits = self.head(encoded)
    token_ids = logits.argmax(dim=-1)
    return token_ids, encoded_len.long()


def _to_onnx_with_token_ids(self: Any, dir_path: str = ".") -> None:
    """Convert to ONNX with token ids instead of logits."""
    saved_forward = self.forward
    self.forward = self.forward_for_export
    try:
        onnx_converter(
            model_name="model",
            out_dir=dir_path,
            module=self,
            inputs=self.encoder.input_example(),
            input_names=["features", "feature_lengths"],
            output_names=["token_ids", "token_ids_lengths"],
            dynamic_axes={
                "features": {0: "batch_size", 2: "seq_len"},
                "feature_lengths": {0: "batch_size"},
                "token_ids": {0: "batch_size", 1: "seq_len"},
                "token_ids_lengths": {0: "batch_size"},
            },
        )
    finally:
        self.forward = saved_forward


def convert_ctc(model: Any) -> tuple[str, str]:
    save_path = "repos/ctc_encoder_onnx/1"
    postprocessing_dir = "repos/ctc_postprocessing/1"

    original_forward = model.forward_for_export
    original_to_onnx = model._to_onnx

    model.forward_for_export = types.MethodType(forward_for_export_with_argmax, model)
    model._to_onnx = types.MethodType(_to_onnx_with_token_ids, model)

    try:
        model.to_onnx(save_path)
    finally:
        model.forward_for_export = original_forward
        model._to_onnx = original_to_onnx

    return save_path, postprocessing_dir


def convert_rnnt(model: Any) -> tuple[str, str]:
    save_path = "repos/gigaam_encoder_onnx/1"
    postprocessing_dir = "repos/rnnt_postprocessing/1"

    # Save encoder, decoder and joint parts to onnx
    model.to_onnx(save_path)

    rename_onnx(
        f"{save_path}/{model.cfg.model_name}_encoder.onnx",
        f"{save_path}/model.onnx",
    )

    os.makedirs(postprocessing_dir, exist_ok=True)
    for part in ("decoder", "joint"):
        src = f"{save_path}/{model.cfg.model_name}_{part}.onnx"
        dst = f"{postprocessing_dir}/{model.cfg.model_name}_{part}.onnx"
        rename_onnx(src, dst)

    return save_path, postprocessing_dir


def rename_onnx(src: str, dst: str) -> None:
    if os.path.exists(src):
        os.rename(src, dst)
        print(f"Moved {src} -> {dst}")


def save_and_distribute_config(
    model: Any, save_path: str, postprocessing_dir: str
) -> None:
    config_path = f"{save_path}/config.yaml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    omegaconf.OmegaConf.save(model.cfg, config_path)
    print(f"Config saved to {config_path}")

    if not model.cfg.model_name.startswith("v3"):
        warnings.warn(
            f"Model '{model.cfg.model_name}' is not from 'v3' family. "
            "Triton preprocessing will use old feature extraction version.",
            UserWarning,
        )

    preprocessing_dir = "repos/preprocessing/1"
    for target_dir in (preprocessing_dir, postprocessing_dir):
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(config_path, f"{target_dir}/config.yaml")
        print(f"Config copied to {target_dir}/config.yaml")


def copy_tokenizer(model: Any, postprocessing_dir: str) -> None:
    tokenizer_path = model.cfg.decoding.get("model_path")
    if tokenizer_path and os.path.exists(tokenizer_path):
        filename = os.path.basename(tokenizer_path)
        shutil.copy(tokenizer_path, f"{postprocessing_dir}/{filename}")
        print(f"Tokenizer copied to {postprocessing_dir}/{filename}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_convert_onnx.py <model_version>")
        sys.exit(1)

    model_version = sys.argv[1]
    model = gigaam.load_model(model_version)

    if "ctc" in model_version:
        save_path, postprocessing_dir = convert_ctc(model)
    else:
        save_path, postprocessing_dir = convert_rnnt(model)

    # Save config and tokenizer for the postprocessing
    save_and_distribute_config(model, save_path, postprocessing_dir)
    copy_tokenizer(model, postprocessing_dir)


if __name__ == "__main__":
    main()
