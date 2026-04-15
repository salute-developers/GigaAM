"""Evaluate a GigaAM checkpoint (pretrained or fine-tuned)."""

import argparse
import json
import os
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import compute_wer

import gigaam
from gigaam.utils import AudioDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_manifest", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--model_name", default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_duration", type=float, default=None)
    p.add_argument("--min_duration", type=float, default=0.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--disable_tqdm", action="store_true", default=False)
    args = p.parse_args()

    src = args.checkpoint or args.model_name
    assert src, "Pass --checkpoint or --model_name"
    model = gigaam.load_model(src, device=args.device)

    ds = AudioDataset(
        args.eval_manifest,
        tokenizer=model.decoding.tokenizer,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        raw_text=False,
        return_tokens=False,
    )
    samples = ds.samples
    print(f"Loaded {len(samples)} samples")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=AudioDataset.collate,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu",
    )

    preds, idx = [], 0
    with torch.inference_mode():
        for wav_pad, wav_lens in tqdm(dl, desc="Inference", disable=args.disable_tqdm):
            enc, enc_len = model(wav_pad.to(args.device), wav_lens.to(args.device))
            for txt, _, _ in model.decoding.decode(model.head, enc, enc_len):
                s = samples[idx]
                preds.append(
                    {
                        "audio_filepath": s.item,
                        "text": s.text or "",
                        "pred_text": txt,
                        "duration": s.duration,
                    }
                )
                idx += 1

    manifest_path = Path(args.eval_manifest)
    if src and os.path.isfile(os.path.expanduser(src)):
        ckpt_path = Path(src)
        experiment = ckpt_path.parent.name
        step_match = re.search(r"step=(\d+)", ckpt_path.stem)
        step_tag = f"step_{step_match.group(1)}" if step_match else ckpt_path.stem
        ckpt_name = f"{experiment}/{step_tag}"
    else:
        ckpt_name = src
    out = manifest_path.parent / "predictions" / manifest_path.stem / ckpt_name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "preds.jsonl", "w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved predictions to {out / 'preds.jsonl'}")

    wer_e2e, wer_raw, e2e_err, e2e_w, raw_err, raw_w = compute_wer(preds)
    print(
        f"WER e2e: {wer_e2e:.2f}% ({e2e_err}/{e2e_w} words)\n"
        f"WER raw: {wer_raw:.2f}% ({raw_err}/{raw_w} words)"
    )


if __name__ == "__main__":
    main()
