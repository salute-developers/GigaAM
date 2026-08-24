"""Average the weights of several GigaAM Lightning checkpoints into one.

The GigaAM analogue of NeMo's checkpoints_averaging.py. Averaging the last/best few
checkpoints of a run is usually worth a few tenths of a WER point for free.

    python average_checkpoints.py --ckpt_dir ./checkpoints/models/<exp_name>
    python average_checkpoints.py --ckpt_dir <dir> --top_k 3 --out avg.ckpt

The result loads like any other checkpoint:

    gigaam.load_model("<out>.ckpt")
    python eval_buckets.py --checkpoint <out>.ckpt ...

Only `hyper_parameters` and `state_dict` are kept: gigaam.load_model reads nothing else
from a fine-tuned checkpoint, so dropping the optimizer state makes the output ~3x
smaller. Pass --keep_optimizer to resume training from the average instead.
"""

import argparse
import gc
import re
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

# Anchored on the digits: a greedy [0-9.]+ would swallow the dot before ".ckpt".
_WER_RE = re.compile(r"val_wer=(\d+(?:\.\d+)?)")
_STEP_RE = re.compile(r"step=(\d+)")
_MODEL_PREFIXES = ("preprocessor.", "encoder.", "head.")


def parse_wer(path: Path) -> Optional[float]:
    m = _WER_RE.search(path.name)
    return float(m.group(1)) if m else None


def parse_step(path: Path) -> Optional[int]:
    m = _STEP_RE.search(path.name)
    return int(m.group(1)) if m else None


def select_checkpoints(args) -> List[Path]:
    if args.checkpoints:
        paths = [Path(p) for p in args.checkpoints]
    else:
        paths = sorted(Path(args.ckpt_dir).glob("*.ckpt"))
        if args.exclude_last:
            paths = [p for p in paths if not p.name.endswith("-last.ckpt")]
        paths = [p for p in paths if not p.name.startswith(args.out_prefix)]

    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"Missing: {', '.join(str(p) for p in missing)}")
    if not paths:
        raise SystemExit("No checkpoints found to average")

    if args.top_k:
        scored = [(parse_wer(p), p) for p in paths]
        if any(w is None for w, _ in scored):
            raise SystemExit(
                "--top_k needs val_wer=... in every filename; pass --checkpoints instead"
            )
        paths = [p for _, p in sorted(scored, key=lambda x: x[0])[: args.top_k]]

    if len(paths) < 2:
        raise SystemExit(f"Need >=2 checkpoints to average, got {len(paths)}")
    return paths


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt_dir", help="directory of *.ckpt to average")
    src.add_argument("--checkpoints", nargs="+", help="explicit checkpoint list")
    p.add_argument(
        "--out",
        default=None,
        help="output path (default: <ckpt_dir>/averaged-<n>.ckpt)",
    )
    p.add_argument(
        "--top_k", type=int, default=None, help="average only the N best by val_wer"
    )
    p.add_argument("--exclude_last", action="store_true", default=True)
    p.add_argument("--include_last", dest="exclude_last", action="store_false")
    p.add_argument(
        "--keep_optimizer",
        action="store_true",
        help="keep optimizer/scheduler state (3x bigger; only needed to resume training)",
    )
    p.add_argument(
        "--out_prefix", default="averaged", help="skip inputs starting with this"
    )
    args = p.parse_args()

    paths = select_checkpoints(args)
    out = (
        Path(args.out)
        if args.out
        else Path(args.ckpt_dir) / f"{args.out_prefix}-{len(paths)}.ckpt"
    )

    print(f"Averaging {len(paths)} checkpoints:")
    for path in paths:
        wer, step = parse_wer(path), parse_step(path)
        extra = f"  (val_wer={wer}, step={step})" if wer is not None else ""
        print(f"  {path.name}{extra}")

    avg: dict = {}
    ref_keys = None
    template = None
    n_nonfloat = 0

    for i, path in enumerate(tqdm(paths, desc="Averaging", unit="ckpt")):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" not in ckpt:
            raise SystemExit(f"No state_dict in {path}")
        sd = ckpt["state_dict"]

        if i == 0:
            ref_keys = set(sd)
            name = ckpt["hyper_parameters"].get("model_name")
            template = {
                "hyper_parameters": ckpt["hyper_parameters"],
                "pytorch-lightning_version": ckpt.get("pytorch-lightning_version"),
                "epoch": ckpt.get("epoch"),
                "global_step": ckpt.get("global_step"),
            }
            if args.keep_optimizer:
                for k in ("optimizer_states", "lr_schedulers", "loops", "callbacks"):
                    if k in ckpt:
                        template[k] = ckpt[k]
            for k, v in sd.items():
                # Non-float buffers (e.g. BatchNorm num_batches_tracked) can't be
                # meaningfully averaged; take them from the first checkpoint.
                avg[k] = v.clone().float() if v.is_floating_point() else v.clone()
                n_nonfloat += 0 if v.is_floating_point() else 1
        else:
            if set(sd) != ref_keys:
                raise SystemExit(
                    f"Key mismatch in {path.name}: checkpoints are incompatible"
                )
            other = ckpt["hyper_parameters"].get("model_name")
            if other != name:
                raise SystemExit(
                    f"model_name mismatch: {name!r} vs {other!r} in {path.name}"
                )
            for k, v in sd.items():
                if avg[k].shape != v.shape:
                    raise SystemExit(f"Shape mismatch for {k} in {path.name}")
                if v.is_floating_point():
                    avg[k] += v.float()

        del ckpt, sd
        gc.collect()

    for k in avg:
        if avg[k].is_floating_point():
            avg[k] /= len(paths)

    n_params = sum(v.numel() for v in avg.values())
    n_model = sum(1 for k in avg if k.startswith(_MODEL_PREFIXES))
    template["state_dict"] = avg

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(template, out)
    size_gb = out.stat().st_size / 1e9
    print(
        f"\nAveraged {n_params / 1e6:.1f}M params over {len(paths)} checkpoints "
        f"({n_model}/{len(avg)} tensors are model weights"
        + (f", {n_nonfloat} non-float copied as-is" if n_nonfloat else "")
        + ")"
    )
    print(f"=> {out}  ({size_gb:.2f} GB)")
    print(
        f"\nCheck it:\n  python eval_buckets.py --checkpoint {out} --buckets_dir ../buckets"
    )


if __name__ == "__main__":
    main()
