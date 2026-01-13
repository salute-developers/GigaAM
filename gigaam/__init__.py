import hashlib
import logging
import os
import urllib.request
import warnings
from typing import Optional, Tuple, Union

import torch
from tqdm import tqdm

from .model import GigaAM, GigaAMASR, GigaAMEmo
from .preprocess import load_audio, load_multichannel_audio
from .utils import format_time

__all__ = [
    "GigaAM",
    "GigaAMASR",
    "GigaAMEmo",
    "load_audio",
    "load_multichannel_audio",
    "format_time",
    "load_model",
]

# Default cache directory
_CACHE_DIR = os.path.expanduser("~/.cache/gigaam")
# Url with model checkpoints
_URL_DIR = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM"
_MODEL_HASHES = {
    "emo": "7ce76f9535cb254488985057c0d33006",
    "v1_ctc": "f027f199e590a391d015aeede2e66174",
    "v1_rnnt": "02c758999bcdc6afcb2087ef256d47ef",
    "v1_ssl": "dc7f7b231f7f91c4968dc21910e7b396",
    "v2_ctc": "e00f59cb5d39624fb30d1786044795bf",
    "v2_rnnt": "547460139acfebd842323f59ed54ab54",
    "v2_ssl": "cd4cf819c8191a07b9d7edcad111668e",
    "v3_ctc": "73413e7be9c6a5935827bfab5c0dd678",
    "v3_rnnt": "0fd2c9a1ff66abd8d32a3a07f7592815",
    "v3_e2e_ctc": "367074d6498f426d960b25f49531cf68",
    "v3_e2e_rnnt": "2730de7545ac43ad256485a462b0a27a",
    "v3_ssl": "70cbf5ed7303a0ed242ddb257e9dc6a6",
}


def _download_file(file_url: str, file_path: str):
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
    short_names = ["ctc", "rnnt", "e2e_ctc", "e2e_rnnt", "ssl"]
    possible_names = short_names + list(_MODEL_HASHES.keys())
    if model_name not in possible_names:
        raise ValueError(
            f"Model '{model_name}' not found. Available model names: {possible_names}"
        )

    if model_name in short_names:
        model_name = f"v3_{model_name}"
    model_url = f"{_URL_DIR}/{model_name}.ckpt"
    model_path = os.path.join(download_root, model_name + ".ckpt")
    return model_name, _download_file(model_url, model_path)


def _download_tokenizer(model_name: str, download_root: str) -> Optional[str]:
    """Download the tokenizer if required and return its path."""
    if model_name != "v1_rnnt" and "e2e" not in model_name:
        return None  # No tokenizer required for this model

    tokenizer_url = f"{_URL_DIR}/{model_name}_tokenizer.model"
    tokenizer_path = os.path.join(download_root, model_name + "_tokenizer.model")
    return _download_file(tokenizer_url, tokenizer_path)


def hash_path(ckpt_path: str) -> str:
    """Calculate binary file hash for checksum"""
    return hashlib.md5(open(ckpt_path, "rb").read()).hexdigest()


def _normalize_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    """Normalize device parameter to torch.device."""
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device_str)
    if isinstance(device, str):
        return torch.device(device)
    return device


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
    device_obj = _normalize_device(device)

    if download_root is None:
        download_root = _CACHE_DIR

    model_name, model_path = _download_model(model_name, download_root)
    tokenizer_path = _download_tokenizer(model_name, download_root)

    assert (
        hash_path(model_path) == _MODEL_HASHES[model_name]
    ), f"Model checksum failed. Please run `rm {model_path}` and reload the model"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=(FutureWarning))
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if use_flash is not None:
        checkpoint["cfg"].encoder.flash_attn = use_flash
    if checkpoint["cfg"].encoder.get("flash_attn", False) and device_obj.type == "cpu":
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

    model.load_state_dict(checkpoint["state_dict"])
    model = model.eval()

    if fp16_encoder and device_obj.type != "cpu":
        model.encoder = model.encoder.half()

    checkpoint["cfg"].model_name = model_name
    return model.to(device_obj)
