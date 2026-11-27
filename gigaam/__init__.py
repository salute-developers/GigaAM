import hashlib
import logging
import os
import time
import urllib.request
import warnings
from typing import Optional, Tuple, Union

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from .model import GigaAM, GigaAMASR, GigaAMDualASR, GigaAMEmo
from .preprocess import load_audio
from .utils import format_time, normalize_raw_text
from .vad_utils import GigaAMVAD

__all__ = [
    "GigaAM",
    "GigaAMASR",
    "GigaAMDualASR",
    "GigaAMEmo",
    "GigaAMVAD",
    "load_audio",
    "format_time",
    "load_model",
    "normalize_raw_text",
    "GigaAMStreaming",
]

# Default cache directory
_CACHE_DIR = os.path.expanduser("~/.cache/gigaam")
# Url with model checkpoints
_URL_DIR = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM"
_DOWNLOAD_RETRIES = 3
_FINAL_MODELS = (
    "multilingual_v2_rnnt",
    "multilingual_v2_ssl",
    "v4_ctc",
    "v4_rnnt",
    "v4_ssl",
    "vad",
)
_DUAL_TOKENIZER_MODELS = {"multilingual_v2_rnnt", "v4_rnnt"}
_MODEL_HASHES = {
    "vad": "de4f63d34555b07278c7e3022e2b4b7f",
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
    "multilingual_ctc": "5379d887c53ccd9cb95981e2a1832720",
    "multilingual_ssl": "af54fed7a0337eeae7c4a25b2f8779c8",
    "multilingual_large_ctc": "79a9adde50dd7f35bbf70927cb6557d0",
    "multilingual_large_ssl": "2ef65a2ca413f6e1f99a4df0e86c1cee",
    "v4_ctc": "7f81bb7c82588bf3f6956e9ca6ce9451",
    "v4_rnnt": "af1027b4fa8b306c7aa1974e4b740fe0",
    "v4_ssl": "22f509eba66cd723e06995e650ae99e8",
    "multilingual_v2_rnnt": "4c24acddcf2f81d512563067b40ca36d",
    "multilingual_v2_ssl": "ea8a54fb545b2e0f89a7f23dd9bcd807",
}


def _download_file(
    file_url: str, file_path: str, retries: int = _DOWNLOAD_RETRIES
) -> str:
    """Download a file if not already cached, retrying a few times on failure."""
    if os.path.exists(file_path):
        return file_path

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            with (
                urllib.request.urlopen(file_url) as source,
                open(file_path, "wb") as output,
            ):
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
        except BaseException as err:
            if os.path.exists(file_path):
                os.remove(file_path)
            if attempt == retries or not isinstance(err, Exception):
                raise
            logging.warning(
                "Download of %s failed (attempt %d/%d): %s. Retrying...",
                file_url,
                attempt,
                retries,
                err,
            )
            time.sleep(2)

    raise RuntimeError(f"Download of {file_url} failed after {retries} attempts.")


def _download_model(model_name: str, download_root: str) -> Tuple[str, str]:
    """Download the model weights if not already cached."""
    short_names = {
        "ctc": "v4_ctc",
        "rnnt": "v4_rnnt",
        "e2e_ctc": "v4_ctc",
        "e2e_rnnt": "v4_rnnt",
        "ssl": "v4_ssl",
        "multilingual_rnnt": "multilingual_v2_rnnt",
        "multilingual_ssl": "multilingual_v2_ssl",
    }
    possible_names = (
        list(short_names) + list(_MODEL_HASHES.keys()) + list(_FINAL_MODELS)
    )
    if model_name not in possible_names:
        raise ValueError(
            f"Model '{model_name}' not found. Available model names: {possible_names}"
        )

    if model_name in short_names:
        model_name = short_names[model_name]
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


def _apply_flash_policy(cfg, use_flash: Optional[bool], device_obj: torch.device):
    """Apply the flash_attn override and the CPU fallback to a model cfg."""
    if use_flash is not None:
        cfg.encoder.flash_attn = use_flash
    if cfg.encoder.get("flash_attn", False) and str(
        cfg.encoder.get("_target_", "")
    ).endswith(".ZapformerEncoder"):
        raise NotImplementedError(
            "flash_attn is not supported for Zapformer; use PyTorch SDPA instead."
        )
    if cfg.encoder.get("flash_attn", False) and device_obj.type == "cpu":
        logging.warning("flash_attn is not supported on CPU. Disabling it...")
        cfg.encoder.flash_attn = False


def _finalize_model(
    model: GigaAM,
    fp16_encoder: bool,
    device_obj: torch.device,
) -> GigaAM:
    """Shared load-time tail: eval mode, optional fp16 encoder, target device."""
    model = model.eval()
    if model.cfg.get("encoder_autocast_dtype") is not None:
        model.cfg.encoder_autocast_dtype = (
            "float16" if fp16_encoder and device_obj.type == "cuda" else "float32"
        )
    if fp16_encoder and device_obj.type == "cuda":
        model.encoder = model.encoder.half()
    return model.to(device_obj)


def _load_configured_model(
    model_path,
    config_path,
    fp16_encoder,
    use_flash,
    device_obj,
    language=None,
    checkpoint=None,
):
    if checkpoint is None:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    embedded = "cfg" in checkpoint and "state_dict" in checkpoint
    cfg = (
        OmegaConf.create(checkpoint["cfg"]) if embedded else OmegaConf.load(config_path)
    )
    weights = checkpoint["state_dict"] if embedded else checkpoint
    model_name = cfg.get("model_name", "")
    is_vad = model_name == "vad"
    if not is_vad:
        _apply_flash_policy(cfg, use_flash, device_obj)
        for decoding in cfg.get("decodings", {}).values():
            if decoding.get("model_path"):
                decoding.model_path = os.path.join(
                    os.path.dirname(config_path), os.path.basename(decoding.model_path)
                )
        decoding = cfg.get("decoding")
        if decoding and decoding.get("model_path"):
            decoding.model_path = os.path.join(
                os.path.dirname(config_path), os.path.basename(decoding.model_path)
            )
    if is_vad:
        model = GigaAMVAD(cfg)
    elif cfg.get("heads"):
        model = GigaAMDualASR(cfg)
        model.default_language = language
    elif "ssl" in model_name:
        model = GigaAM(cfg)
    elif "emo" in model_name:
        model = GigaAMEmo(cfg)
    else:
        model = GigaAMASR(cfg)
    model.load_state_dict(weights)
    return _finalize_model(model, False if is_vad else fp16_encoder, device_obj)


def _resolve_dual_tokenizers(model_name, cfg, download_root, checkpoint_dir=None):
    if model_name not in _DUAL_TOKENIZER_MODELS:
        return
    for head in ("raw", "e2e"):
        decoding = cfg.decodings[head]
        path = decoding.get("model_path")
        if checkpoint_dir and path:
            path = os.path.expanduser(path)
            if not os.path.isfile(path):
                local_path = os.path.join(checkpoint_dir, os.path.basename(path))
                if os.path.isfile(local_path):
                    path = local_path
            if os.path.isfile(path):
                decoding.model_path = path
                continue
            filename = f"{model_name}_{head}_tokenizer.model"
            if os.path.basename(path) != filename:
                raise FileNotFoundError(f"Tokenizer file not found: {path}")
        filename = f"{model_name}_{head}_tokenizer.model"
        decoding.model_path = _download_file(
            f"{_URL_DIR}/{filename}", os.path.join(download_root, filename)
        )


def load_model(
    model_name: str,
    fp16_encoder: bool = True,
    use_flash: Optional[bool] = False,
    device: Optional[Union[str, torch.device]] = None,
    download_root: Optional[str] = None,
    streaming: bool = False,
    future_context_ms: Optional[int] = None,
    language=None,
    word_timestamps: bool = False,
) -> Union[GigaAM, GigaAMVAD, "GigaAMStreaming"]:
    """
    Load the GigaAM model by name, or a local ``.ckpt`` from fine-tuning with ``train_utils/train.py``.

    Parameters
    ----------
    model_name : str
        Model name or a path to a ``.ckpt`` file.
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
    if streaming:
        model = load_model(model_name, fp16_encoder, use_flash, device, download_root)
        return GigaAMStreaming(model, future_context_ms, language, word_timestamps)
    # Normalize device parameter to torch.device.
    device_obj = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if download_root is None:
        download_root = _CACHE_DIR

    local_path = os.path.expanduser(model_name)
    config_path = os.path.splitext(local_path)[0] + ".yaml"
    if os.path.isfile(local_path):
        checkpoint = torch.load(local_path, map_location="cpu", weights_only=False)
        if "cfg" in checkpoint and "vad" in checkpoint["cfg"]:
            vad_model = GigaAMVAD(checkpoint["cfg"])
            vad_model.load_state_dict(checkpoint["state_dict"], strict=True)
            return vad_model.eval().to(device_obj)
        if "hyper_parameters" not in checkpoint and (
            ("cfg" in checkpoint and "state_dict" in checkpoint)
            or os.path.isfile(config_path)
        ):
            return _load_configured_model(
                local_path,
                config_path,
                fp16_encoder,
                use_flash,
                device_obj,
                language,
                checkpoint,
            )
        finetuned = checkpoint
        hparams = finetuned["hyper_parameters"]
        base_name = hparams["model_name"]
        sd = {
            k: v
            for k, v in finetuned["state_dict"].items()
            if k.startswith(
                (
                    "preprocessor.",
                    "encoder.",
                    "head.",
                    "heads.",
                    "language_embedding.",
                    "language_fusion.",
                )
            )
        }
        model_cfg = hparams.get("model_cfg")
        if model_cfg is not None:
            cfg = OmegaConf.create(model_cfg)
            if cfg.get("heads"):
                _resolve_dual_tokenizers(
                    base_name, cfg, download_root, os.path.dirname(local_path)
                )
                _apply_flash_policy(cfg, use_flash, device_obj)
                dual_model = GigaAMDualASR(cfg)
                dual_model.default_language = (
                    language if language is not None else hparams.get("language")
                )
                dual_model.load_state_dict(sd)
                return _finalize_model(dual_model, fp16_encoder, device_obj)
            if not (cfg.get("decoding") or {}).get("model_path"):
                _apply_flash_policy(cfg, use_flash, device_obj)
                asr_model = GigaAMASR(cfg)
                asr_model.load_state_dict(sd)
                return _finalize_model(asr_model, fp16_encoder, device_obj)

        model = load_model(
            base_name,
            fp16_encoder=fp16_encoder,
            use_flash=use_flash,
            device=device_obj,
            download_root=download_root,
        )
        model.load_state_dict(sd)
        return model

    model_name, model_path = _download_model(model_name, download_root)
    expected_hash = _MODEL_HASHES.get(model_name)
    if expected_hash is not None:
        assert (
            hash_path(model_path) == expected_hash
        ), f"Model checksum failed. Please remove {model_path} and reload the model"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=(FutureWarning))
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    checkpoint["cfg"] = OmegaConf.create(checkpoint["cfg"])
    _resolve_dual_tokenizers(model_name, checkpoint["cfg"], download_root)
    tokenizer_path = (
        _download_tokenizer(model_name, download_root)
        if model_name not in _FINAL_MODELS
        else None
    )

    if model_name == "vad":
        vad_model = GigaAMVAD(checkpoint["cfg"])
        vad_model.load_state_dict(checkpoint["state_dict"], strict=True)
        return vad_model.eval().to(device_obj)

    _apply_flash_policy(checkpoint["cfg"], use_flash, device_obj)

    if tokenizer_path is not None:
        checkpoint["cfg"].decoding.model_path = tokenizer_path

    if "ssl" in model_name:
        model = GigaAM(checkpoint["cfg"])
    elif "emo" in model_name:
        model = GigaAMEmo(checkpoint["cfg"])
    elif checkpoint["cfg"].get("heads"):
        model = GigaAMDualASR(checkpoint["cfg"])
        model.default_language = language
    else:
        model = GigaAMASR(checkpoint["cfg"])

    model.load_state_dict(checkpoint["state_dict"])
    checkpoint["cfg"].model_name = model_name
    return _finalize_model(model, fp16_encoder, device_obj)


from .streaming import GigaAMStreaming  # noqa: E402
