"""Grid-search the KenLM fusion weights (alpha/beta) against raw WER.

The acoustic model runs once and its log-probs are cached; every grid point then
re-decodes from the cache on CPU only. Without this each point would cost a full
forward pass over the dataset.

    python tune_beam_lm.py --checkpoint ... --manifest ./uz_dev_general.tsv \
        --cache_dir ./lm_cache --lm_path ./lm_uz/lm.bin --unigrams_path ./lm_uz/lm.vocab
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from beam_lm import (
    DEFAULT_BEAM_PRUNE_LOGP,
    DEFAULT_BEAM_SIZE,
    DEFAULT_TOKEN_MIN_LOGP,
    _read_unigrams,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import compute_wer

import gigaam
from gigaam.utils import AudioDataset


def build_cache(args, cache_dir: Path) -> None:
    """Run the model once; store per-utterance log-probs sliced to enc_len."""
    src = args.checkpoint or args.model_name
    model = gigaam.load_model(src, device=args.device)
    tokenizer = model.decoding.tokenizer
    if not getattr(tokenizer, "charwise", False):
        raise SystemExit(f"'{src}' is not charwise; beam+LM needs a charwise model.")

    ds = AudioDataset(
        args.manifest, tokenizer=tokenizer, raw_text=False, return_tokens=False
    )
    samples = ds.samples
    if args.max_utts:
        samples = samples[: args.max_utts]
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=AudioDataset.collate,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu",
    )

    chunks: List[np.ndarray] = []
    meta: List[dict] = []
    offset = idx = 0
    with torch.inference_mode():
        for wav, wav_lens in tqdm(
            dl, desc="caching logprobs", disable=args.disable_tqdm
        ):
            if args.max_utts and idx >= args.max_utts:
                break
            enc, enc_len = model(wav.to(args.device), wav_lens.to(args.device))
            log_probs = model.head(encoder_output=enc).float().cpu().numpy()
            lens = enc_len.clamp(min=0, max=log_probs.shape[1]).cpu().tolist()
            for i, n in enumerate(lens):
                if args.max_utts and idx >= args.max_utts:
                    break
                # Sliced here, once: the sweep can never reintroduce the padding bug.
                chunks.append(log_probs[i, :n, :].astype(np.float16))
                s = samples[idx]
                meta.append(
                    {
                        "audio_filepath": s.item,
                        "text": s.text or "",
                        "duration": s.duration,
                        "offset": offset,
                        "length": n,
                    }
                )
                offset += n
                idx += 1

    cache_dir.mkdir(parents=True, exist_ok=True)
    flat = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1), np.float16)
    np.save(cache_dir / "logprobs.npy", flat)
    with open(cache_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(cache_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(list(tokenizer.vocab), f, ensure_ascii=False)
    mb = flat.nbytes / 1e6
    print(
        f"Cached {len(meta)} utts, {flat.shape[0]} frames, {mb:.0f} MB -> {cache_dir}"
    )


def load_cache(cache_dir: Path) -> Tuple[np.ndarray, List[dict], List[str]]:
    flat = np.load(cache_dir / "logprobs.npy", mmap_mode="r")
    meta = [
        json.loads(line)
        for line in open(cache_dir / "meta.jsonl", encoding="utf-8")
        if line.strip()
    ]
    vocab = json.load(open(cache_dir / "vocab.json", encoding="utf-8"))
    return flat, meta, vocab


def sweep(args, cache_dir: Path) -> dict:
    from pyctcdecode import build_ctcdecoder

    flat, meta, vocab = load_cache(cache_dir)
    labels = list(vocab) + [""]
    unigrams = _read_unigrams(args.unigrams_path) if args.unigrams_path else None
    logits = [
        np.ascontiguousarray(
            flat[m["offset"] : m["offset"] + m["length"], :], np.float32
        )
        for m in meta
    ]
    print(
        f"Sweeping {len(args.alphas)}x{len(args.betas)} points over {len(logits)} utts"
    )

    results, best = {}, None
    for alpha in args.alphas:
        for beta in args.betas:
            decoder = build_ctcdecoder(
                labels,
                kenlm_model_path=args.lm_path,
                unigrams=unigrams,
                alpha=alpha,
                beta=beta,
            )
            kwargs = dict(
                beam_width=args.beam_size,
                beam_prune_logp=args.beam_prune_logp,
                token_min_logp=args.token_min_logp,
            )
            # The pool must be forked AFTER this grid point's decoder exists: workers
            # reach the LM only through pyctcdecode's class-level model_container,
            # which they inherit at fork time. Forking once up front leaves later
            # decoders' LMs missing in the workers (KeyError on the container hash).
            pool = (
                mp.get_context("fork").Pool(args.workers) if args.workers > 0 else None
            )
            try:
                if pool is not None:
                    texts = decoder.decode_batch(pool, logits, **kwargs)
                else:
                    texts = [decoder.decode(x, **kwargs) for x in logits]
            finally:
                if pool is not None:
                    pool.terminate()
                    pool.join()
            preds = [{"text": m["text"], "pred_text": t} for m, t in zip(meta, texts)]
            wer_raw = compute_wer(preds)[1]
            results[f"{alpha}_{beta}"] = round(wer_raw, 5)
            mark = ""
            if best is None or wer_raw < best[2]:
                best, mark = (alpha, beta, wer_raw), "  <- best"
            print(f"  alpha={alpha:<5} beta={beta:<5} WER raw {wer_raw:6.3f}%{mark}")

    print(f"\nBEST: alpha={best[0]} beta={best[1]} -> WER raw {best[2]:.3f}%")
    return {
        "best": {"alpha": best[0], "beta": best[1], "wer": round(best[2], 5)},
        "grid": results,
        "lm_path": args.lm_path,
        "manifest": args.manifest,
        "beam_size": args.beam_size,
        "n_utts": len(logits),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--cache_dir", default="./lm_cache")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--model_name", default=None)
    p.add_argument("--lm_path", default=None)
    p.add_argument("--unigrams_path", default=None)
    p.add_argument(
        "--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    p.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5, 2.0])
    p.add_argument("--beam_size", type=int, default=DEFAULT_BEAM_SIZE)
    p.add_argument("--beam_prune_logp", type=float, default=DEFAULT_BEAM_PRUNE_LOGP)
    p.add_argument("--token_min_logp", type=float, default=DEFAULT_TOKEN_MIN_LOGP)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max_utts", type=int, default=None, help="cap for a faster sweep")
    p.add_argument("--batch_size", type=int, default=64, help="caching pass only")
    p.add_argument("--num_workers", type=int, default=4, help="caching pass only")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="./lm_tune_report.json")
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--disable_tqdm", action="store_true", default=False)
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    if args.rebuild_cache or not (cache_dir / "logprobs.npy").is_file():
        if not (args.checkpoint or args.model_name):
            raise SystemExit("Building the cache needs --checkpoint or --model_name")
        build_cache(args, cache_dir)
    else:
        print(f"Reusing cache at {cache_dir} (--rebuild_cache to refresh)")

    report = sweep(args, cache_dir)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
        f.write("\n")
    print(f"Saved report to {args.out}")


if __name__ == "__main__":
    main()
