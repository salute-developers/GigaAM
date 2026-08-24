"""Evaluate a GigaAM checkpoint on every TSV manifest in a directory.

Loads the model once and reuses it across manifests, then writes raw WER per set:

    {
      "uz_dev_general":            {"wer": 15.66123},
      "uz_merged_general_albatross": {"wer": 14.87004}
    }
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from beam_lm import add_beam_args, maybe_attach
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import compute_wer

import gigaam
from gigaam.utils import AudioDataset


def evaluate(model, manifest: Path, args) -> float:
    """Return raw WER (%) for one manifest."""
    ds = AudioDataset(
        str(manifest),
        tokenizer=model.decoding.tokenizer,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        raw_text=False,
        return_tokens=False,
    )
    samples = ds.samples
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
        for wav_pad, wav_lens in tqdm(
            dl, desc=manifest.stem, disable=args.disable_tqdm, leave=False
        ):
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

    if args.save_preds:
        out_dir = manifest.parent / "predictions" / manifest.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "preds.jsonl", "w", encoding="utf-8") as f:
            for r in preds:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _, wer_raw, _, _, raw_err, raw_w = compute_wer(preds)
    print(f"  {manifest.stem:<48} WER raw: {wer_raw:6.2f}%  ({raw_err}/{raw_w} words)")
    return wer_raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--buckets_dir", default="../buckets", help="dir with *.tsv manifests"
    )
    p.add_argument(
        "--manifests",
        nargs="+",
        default=None,
        help="explicit TSVs (overrides --buckets_dir)",
    )
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--model_name", default=None)
    p.add_argument("--out", default="./wer_report.json")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_duration", type=float, default=None)
    p.add_argument("--min_duration", type=float, default=0.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--save_preds", action="store_true", help="also dump preds.jsonl per set"
    )
    p.add_argument("--disable_tqdm", action="store_true", default=False)
    add_beam_args(p)
    args = p.parse_args()

    src = args.checkpoint or args.model_name
    assert src, "Pass --checkpoint or --model_name"

    if args.manifests:
        manifests: List[Path] = [Path(m) for m in args.manifests]
    else:
        manifests = sorted(Path(args.buckets_dir).glob("*.tsv"))
    if not manifests:
        raise SystemExit(f"No .tsv manifests found in {args.buckets_dir}")

    missing = [m for m in manifests if not m.is_file()]
    if missing:
        raise SystemExit(f"Missing manifests: {', '.join(str(m) for m in missing)}")

    print(f"Loading model: {src}")
    model = gigaam.load_model(src, device=args.device)
    # Attached once, before the loop: the LM and the decode pool are expensive to build
    # and are reused across every bucket.
    beam_decoding = maybe_attach(model, args)
    print(f"Evaluating {len(manifests)} manifest(s) on {args.device}\n")

    report: Dict[str, Dict[str, float]] = {}
    try:
        for manifest in manifests:
            wer_raw = evaluate(model, manifest, args)
            report[manifest.stem] = {"wer": round(wer_raw, 5)}
    finally:
        if beam_decoding is not None:
            beam_decoding.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
        f.write("\n")
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
