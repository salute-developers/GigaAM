"""SSL fine-tuning utilities."""

import argparse
import csv
import json
import os
import warnings
from typing import List, Tuple

from omegaconf import OmegaConf

from gigaam.model import GigaAM, GigaAMASR
from gigaam.utils import normalize_raw_text


def _finalize_vocab(chars) -> List[str]:
    """Deterministic, space-included ordering for a character set."""
    for ch in chars:
        if len(ch) != 1:
            raise ValueError(f"Vocabulary entries must be single characters: {ch!r}")
    return sorted(set(chars) | {" "})


def derive_vocab_from_manifest(manifest_path: str, raw_text: bool = True) -> List[str]:
    """
    Collect the set of characters that appear in a manifest's ``transcription``
    column — normalized when ``raw_text``, verbatim otherwise.
    """
    chars: set = set()
    with open(manifest_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "transcription" not in reader.fieldnames:
            raise ValueError(
                f"Manifest {manifest_path} has no 'transcription' column; "
                "cannot derive a vocabulary."
            )
        for row in reader:
            text = row.get("transcription") or ""
            chars.update(normalize_raw_text(text) if raw_text else text)
    if not chars:
        raise ValueError(f"No transcription characters found in {manifest_path}.")
    return _finalize_vocab(chars)


def resolve_vocab(args: argparse.Namespace) -> List[str]:
    """Resolve the target vocabulary."""
    if args.vocab:
        if not args.vocab.endswith(".json"):
            raise ValueError(f"Vocabulary file {args.vocab} is not a JSON file.")
        with open(args.vocab, encoding="utf-8") as f:
            tokens = json.load(f)
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            raise ValueError(f"{args.vocab} must contain a JSON list of strings.")
        vocab = _finalize_vocab(set(tokens))
    elif args.build_vocab_from_manifest:
        vocab = derive_vocab_from_manifest(args.train_manifest, raw_text=args.raw_text)
    else:
        raise ValueError(
            "Fine-tuning from an SSL backbone requires a target vocabulary: pass "
            "--vocab <file> or --build_vocab_from_manifest."
        )

    vocab_set = set(vocab)
    in_train = set(
        derive_vocab_from_manifest(args.train_manifest, raw_text=args.raw_text)
    )
    in_val = set(derive_vocab_from_manifest(args.val_manifest, raw_text=args.raw_text))
    for manifest, chars in (
        (args.train_manifest, in_train),
        (args.val_manifest, in_val),
    ):
        missing = chars - vocab_set
        if missing:
            warnings.warn(
                f"Characters {sorted(missing)} appear in {manifest} but are missing "
                "from the vocabulary; they will be silently dropped from the targets.",
                stacklevel=2,
            )
    if args.vocab:
        dead = vocab_set - in_train
        if dead:
            warnings.warn(
                f"Vocabulary entries {sorted(dead)} never occur in the "
                f"{'normalized ' if args.raw_text else ''}train transcriptions; "
                "they will train as dead classes.",
                stacklevel=2,
            )

    if args.save_vocab and os.environ.get("LOCAL_RANK", "0") == "0":
        with open(args.save_vocab, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)
        print(f"Saved vocabulary ({len(vocab)} chars) to {args.save_vocab}")
    return vocab


def _head_cfg(
    head_type: str,
    vocab: List[str],
    d_model: int,
    pred_hidden: int,
    pred_rnn_layers: int,
    joint_hidden: int,
) -> Tuple[dict, dict]:
    num_classes = len(vocab) + 1
    if head_type == "ctc":
        head = {
            "_target_": "gigaam.decoder.CTCHead",
            "feat_in": d_model,
            "num_classes": num_classes,
        }
    else:
        head = {
            "_target_": "gigaam.decoder.RNNTHead",
            "decoder": {
                "pred_hidden": pred_hidden,
                "pred_rnn_layers": pred_rnn_layers,
                "num_classes": num_classes,
            },
            "joint": {
                "enc_hidden": d_model,
                "pred_hidden": pred_hidden,
                "joint_hidden": joint_hidden,
                "num_classes": num_classes,
            },
        }
    decoding = {
        "_target_": (
            "gigaam.decoding.CTCGreedyDecoding"
            if head_type == "ctc"
            else "gigaam.decoding.RNNTGreedyDecoding"
        ),
        "vocabulary": list(vocab),
        "model_path": None,
    }
    return head, decoding


def build_asr_from_ssl(
    ssl: GigaAM,
    vocab: List[str],
    head_type: str = "ctc",
    raw_text: bool = True,
    rnnt_pred_hidden: int = 320,
    rnnt_pred_rnn_layers: int = 1,
    rnnt_joint_hidden: int = 320,
) -> GigaAMASR:
    """
    Build a ``GigaAMASR`` on top of a loaded SSL backbone with a randomly
    initialized, character-wise head sized to ``vocab``.
    """
    if type(ssl) is not GigaAM:
        raise TypeError(f"Expected an SSL GigaAM backbone, got {type(ssl).__name__}.")
    if head_type not in ("ctc", "rnnt"):
        raise ValueError(f"head_type must be 'ctc' or 'rnnt', got '{head_type}'.")

    head_cfg, decoding_cfg = _head_cfg(
        head_type,
        vocab,
        d_model=int(ssl.cfg.encoder.d_model),
        pred_hidden=rnnt_pred_hidden,
        pred_rnn_layers=rnnt_pred_rnn_layers,
        joint_hidden=rnnt_joint_hidden,
    )
    cfg = OmegaConf.create(
        {
            "model_name": ssl.cfg.model_name.replace("_ssl", "") + f"_{head_type}_ft",
            "raw_text": raw_text,
            "preprocessor": OmegaConf.to_container(ssl.cfg.preprocessor, resolve=True),
            "encoder": OmegaConf.to_container(ssl.cfg.encoder, resolve=True),
            "head": head_cfg,
            "decoding": decoding_cfg,
        }
    )

    model = GigaAMASR(cfg)
    model.encoder.load_state_dict(ssl.encoder.state_dict())
    model.preprocessor.load_state_dict(ssl.preprocessor.state_dict())
    return model.eval()
