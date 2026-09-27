#!/usr/bin/env python3
"""Accuracy and PyTorch↔MLX parity on the Golos Crowd sample.

For every model it runs the upstream PyTorch GigaAM (CPU, fp32 — the
reference) and the converted MLX model (fp32 and fp16) on each clip and records:

* transcripts of all three, raw and normalized;
* corpus WER / CER against the Golos references;
* exact transcript agreement MLX vs PyTorch;
* max |encoder_pt − encoder_mlx| per clip (and CTC frame-label agreement).

Models are read from <models>/<name> and <models>/<name>-fp32, as produced by
convert_gigaam_to_mlx.py. Results go to <out>/<name>.jsonl and <out>/summary.json.
"""
import argparse
import json
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

import gigaam
from common import HERE, DEFAULT_DATA, corpus_error_rate, load_corpus, machine_info, normalize
from gigaam_mlx import load_model as load_mlx

MODELS = ["v3_ctc", "v3_rnnt", "v3_e2e_ctc", "v3_e2e_rnnt"]


def mlx_encode(model, audio: np.ndarray):
    mel, lengths = model._compute_features(mx.array(audio))
    encoded, enc_len = model(mel, lengths)
    mx.eval(encoded, enc_len)
    return encoded, int(enc_len[0])


def evaluate(name: str, clips, models_dir: Path):
    pt = gigaam.load_model(name, device="cpu", fp16_encoder=False, use_flash=False)
    pt.eval()
    variants = {"mlx_fp32": load_mlx(models_dir / f"{name}-fp32"), "mlx_fp16": load_mlx(models_dir / name)}
    is_ctc = variants["mlx_fp32"].cfg.head_type == "ctc"

    rows = []
    for clip in clips:
        row = {"id": clip.id, "duration": clip.duration, "reference": clip.reference}
        with torch.no_grad():
            row["pytorch"] = pt.transcribe(str(clip.path)).text
            wav = torch.from_numpy(clip.audio)[None]
            enc_t, len_pt = pt.forward(wav, torch.tensor([wav.shape[-1]]))  # [1, D, T]
            logits_pt = pt.head(enc_t) if is_ctc else None  # [1, T, C]
            enc_pt = enc_t[0].T.numpy()  # [T, D]
        for key, model in variants.items():
            row[key] = model.transcribe(mx.array(clip.audio))
            enc, n = mlx_encode(model, clip.audio)
            enc_np = np.array(enc[0].astype(mx.float32))
            assert n == int(len_pt[0]) and enc_np.shape == enc_pt.shape, (clip.id, n, enc_np.shape, enc_pt.shape)
            row[f"{key}_enc_max_abs_diff"] = float(np.abs(enc_np - enc_pt).max())
            if is_ctc:
                labels_mlx = np.array(mx.argmax(model.head(enc), axis=-1)[0])
                labels_pt = logits_pt[0].argmax(-1).numpy()
                row[f"{key}_ctc_frames_agree"] = int((labels_mlx == labels_pt).sum())
                row["ctc_frames"] = int(labels_pt.shape[0])
        row["enc_abs_max"] = float(np.abs(enc_pt).max())
        rows.append(row)
    return rows


_FORMATTED = re.compile(r"[0-9a-zA-Z]")


def summarize(rows):
    """Corpus metrics. `wer_plain_subset` restricts to clips whose PyTorch
    transcript has no digits or Latin letters: E2E models write "HD", "4K",
    "15-й", phone numbers as digit groups, while Golos references spell them
    in Cyrillic words, and no normalizer maps one onto the other reliably."""
    out = {}
    plain = [r for r in rows if not _FORMATTED.search(r["pytorch"])]
    for key in ["pytorch", "mlx_fp32", "mlx_fp16"]:
        pairs = [(r["reference"], normalize(r[key])) for r in rows]
        out[key] = {
            "wer": corpus_error_rate(pairs, "word"),
            "cer": corpus_error_rate(pairs, "char"),
            "wer_plain_subset": corpus_error_rate(
                [(r["reference"], normalize(r[key])) for r in plain], "word"
            ),
        }
        if key != "pytorch":
            out[key]["exact_match_vs_pytorch"] = sum(r[key] == r["pytorch"] for r in rows)
            diffs = [r[f"{key}_enc_max_abs_diff"] for r in rows]
            out[key]["enc_max_abs_diff"] = {"median": float(np.median(diffs)), "max": float(np.max(diffs))}
            if "ctc_frames" in rows[0]:
                agree = sum(r[f"{key}_ctc_frames_agree"] for r in rows)
                out[key]["ctc_frame_agreement"] = agree / sum(r["ctc_frames"] for r in rows)
    out["enc_abs_max_median"] = float(np.median([r["enc_abs_max"] for r in rows]))
    out["clips"] = len(rows)
    out["plain_subset_clips"] = len(plain)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--models-dir", default=str(HERE.parent / "eval-models"))
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--limit", type=int, default=None, help="first N clips only")
    parser.add_argument("--summarize-only", action="store_true", help="recompute summary.json from saved <out>/<model>.jsonl")
    args = parser.parse_args()

    mx.set_cache_limit(2**30)  # see bench_speed.py: MLX's buffer cache grows with every new input length
    clips = load_corpus(Path(args.data))[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary["machine"] = machine_info()
    summary["corpus"] = {"clips": len(clips), "seconds": sum(c.duration for c in clips)}

    for name in args.models:
        t0 = time.time()
        if args.summarize_only:
            rows = [json.loads(line) for line in open(out / f"{name}.jsonl")]
        else:
            rows = evaluate(name, clips, Path(args.models_dir))
            with open(out / f"{name}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[name] = summarize(rows)
        s = summary[name]
        print(
            f"{name}: WER pt {s['pytorch']['wer']['rate']:.2%} | mlx32 {s['mlx_fp32']['wer']['rate']:.2%} "
            f"| mlx16 {s['mlx_fp16']['wer']['rate']:.2%} | exact32 {s['mlx_fp32']['exact_match_vs_pytorch']}/{len(rows)} "
            f"exact16 {s['mlx_fp16']['exact_match_vs_pytorch']}/{len(rows)} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
