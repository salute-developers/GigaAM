import logging
import os
import urllib.request
from typing import Optional, Tuple, Union

import torch
from tqdm import tqdm

from .model import GigaAM, GigaAMASR, GigaAMEmo
from .preprocess import load_audio
from .utils import format_time

# Default cache directory
_CACHE_DIR = os.path.expanduser("~/.cache/gigaam")
# Url with model checkpoints
_URL_DIR = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM"
_MODEL_NAMES = [
    "ctc",
    "rnnt",
    "ssl",
    "emo",
    "v1_ctc",
    "v1_rnnt",
    "v1_ssl",
    "v2_ctc",
    "v2_rnnt",
    "v2_ssl",
]


def _download_file(file_url: str, file_path: str) -> str:
    """Helper to download a file if not already cached."""
    if os.path.exists(file_path):
        return file_path

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with urllib.request.urlopen(file_url) as source, open(file_path, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length", 0)),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return file_path


def _download_model(model_name: str, download_root: str) -> Tuple[str, str]:
    """Download the model weights if not already cached."""
    if model_name not in _MODEL_NAMES:
        raise ValueError(
            f"Model '{model_name}' not found. Available model names: {_MODEL_NAMES}"
        )

    if model_name in ["ctc", "rnnt", "ssl"]:
        model_name = f"v2_{model_name}"
    if model_name == "emo":
        model_name = f"v1_{model_name}"
    model_url = f"{_URL_DIR}/{model_name}.ckpt"
    model_path = os.path.join(download_root, model_name + ".ckpt")
    return model_name, _download_file(model_url, model_path)


def _download_tokenizer(model_name: str, download_root: str) -> Optional[str]:
    """Download the tokenizer if required and return its path."""
    if model_name != "v1_rnnt":
        return None  # No tokenizer required for this model

    tokenizer_url = f"{_URL_DIR}/{model_name}_tokenizer.model"
    tokenizer_path = os.path.join(download_root, model_name + "_tokenizer.model")
    return _download_file(tokenizer_url, tokenizer_path)


def load_model(
    model_name: str,
    fp16_encoder: bool = True,
    use_flash: Optional[bool] = False,
    device: Optional[Union[str, torch.device]] = None,
    download_root: Optional[str] = None,
) -> Union[GigaAM, GigaAMEmo, GigaAMASR]:
    """
    Load the GigaAM model by name.

    Parameters
    ----------
    model_name : str
        The name of the model to load.
    fp16_encoder:
        Whether to convert encoder weights to FP16 precision.
    use_flash : Optional[bool]
        Whether to use flash_attn if the model allows it (requires the flash_attn library installed).
        Default to False.
    device : Optional[Union[str, torch.device]]
        The device to load the model onto. Defaults to "cuda" if available, otherwise "cpu".
    download_root : Optional[str]
        The directory to download the model to. Defaults to "~/.cache/gigaam".
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    if download_root is None:
        download_root = _CACHE_DIR

    model_name, model_path = _download_model(model_name, download_root)
    tokenizer_path = _download_tokenizer(model_name, download_root)

    checkpoint = torch.load(model_path, map_location="cpu")

    if use_flash is not None:
        checkpoint["cfg"].encoder.flash_attn = use_flash
    if checkpoint["cfg"].encoder.get("flash_attn", False) and device.type == "cpu":
        logging.warning("flash_attn is not supported on CPU. Disabling it...")
        checkpoint["cfg"].encoder.flash_attn = False

    if tokenizer_path is not None:
        checkpoint["cfg"].decoding.model_path = tokenizer_path

    if "ssl" in model_name:
        model = GigaAM(checkpoint["cfg"])
    elif "emo" in model_name:
        model = GigaAMEmo(checkpoint["cfg"])
    else:
        model = GigaAMASR(checkpoint["cfg"])

    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = model.eval()

    if fp16_encoder and device.type != "cpu":
        model.encoder = model.encoder.half()
    elif fp16_encoder:
        logging.warning("fp16 is not supported on CPU. Leaving fp32 weights...")

    checkpoint["cfg"].model_name = model_name
    return model.to(device)
