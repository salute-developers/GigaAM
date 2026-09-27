#!/usr/bin/env python3
"""Transcription speed: MLX vs PyTorch on the Golos Crowd sample.

Two workloads, both from in-memory audio (no ffmpeg / file I/O timed):

* ``short`` — the 200 clips one by one (1–13.5 s, median ≈ 3.9 s);
* ``long``  — consecutive clips concatenated into ≈20 s segments
  (kept under the 25 s limit of upstream ``transcribe``).

Each backend gets one warm-up pass over the workload, then ``--repeats`` timed
passes. Reported per pass: wall time and RTFx = audio seconds / wall seconds;
the summary uses the median pass. Model load time is measured separately.

Backends: mlx_fp16 (fp16 file, weights upcast to fp32 at load — the
load_model default), mlx_fp16_stored (fp16 params kept), mlx_fp32 (fp32 file),
pytorch cpu fp32 and pytorch mps (upstream default for accelerators: fp16
encoder under autocast).

Every model/backend pair runs in its own subprocess so memory is returned to
the OS between runs. MLX keeps freed buffers in a cache that, by default, may
grow to most of unified memory when input lengths vary; the cache is capped at
1 GB here (costs ≈4% on the short workload). A run aborts if framework memory
exceeds --max-gb. Peak framework memory is reported: MLX peak allocation,
MPS driver allocation, CPU max RSS.
"""
import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

import gigaam
from common import DEFAULT_DATA, HERE, load_corpus, machine_info
from gigaam_mlx import load_model as load_mlx

MODELS = ["v3_ctc", "v3_rnnt", "v3_e2e_ctc", "v3_e2e_rnnt"]
BACKENDS = ["mlx_fp16", "mlx_fp16_stored", "mlx_fp32", "torch_cpu", "torch_mps"]


def long_segments(clips, target=20.0, limit=24.0):
    segments, cur, dur = [], [], 0.0
    for c in clips:
        if dur + c.duration > limit and cur:
            segments.append(np.concatenate(cur))
            cur, dur = [], 0.0
        cur.append(c.audio)
        dur += c.duration
        if dur >= target:
            segments.append(np.concatenate(cur))
            cur, dur = [], 0.0
    return segments  # the last partial segment is dropped


def make_backend(name, backend, models_dir: Path):
    t0 = time.perf_counter()
    if backend.startswith("mlx"):
        path = models_dir / (name + ("-fp32" if backend == "mlx_fp32" else ""))
        # mlx_fp16: fp16 file, upcast at load (load_model default); _stored keeps fp16 params
        model = load_mlx(path, dtype=None) if backend == "mlx_fp16_stored" else load_mlx(path)

        def run(audio):
            return model.transcribe(mx.array(audio))

    else:
        device = "cpu" if backend == "torch_cpu" else "mps"
        model = gigaam.load_model(name, device=device, fp16_encoder=device != "cpu", use_flash=False)
        model.eval()

        def run(audio):
            with torch.inference_mode():
                wav = torch.from_numpy(audio).to(model._device).to(model._dtype)[None]
                length = torch.full([1], wav.shape[-1], device=model._device)
                encoded, encoded_len = model.forward(wav, length)
                text = model._decode(encoded, encoded_len, length)[0][0]
            if device == "mps":
                torch.mps.synchronize()
            return text

    load_s = time.perf_counter() - t0
    return run, load_s


GB = 2**30


def framework_bytes(backend):
    if backend.startswith("mlx"):
        return mx.get_active_memory() + mx.get_cache_memory()
    if backend == "torch_mps":
        return torch.mps.driver_allocated_memory()
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS


def peak_bytes(backend):
    if backend.startswith("mlx"):
        return mx.get_peak_memory()
    if backend == "torch_mps":
        return torch.mps.driver_allocated_memory()
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def timed_pass(run, audios, backend, max_bytes):
    lat = []
    t0 = time.perf_counter()
    for a in audios:
        s = time.perf_counter()
        run(a)
        lat.append(time.perf_counter() - s)
        if framework_bytes(backend) > max_bytes:
            raise SystemExit(f"{backend}: framework memory above {max_bytes / GB:.1f} GB, aborting")
    return time.perf_counter() - t0, lat


def run_one(name, backend, args, workloads, seconds):
    mx.set_cache_limit(GB)
    run, load_s = make_backend(name, backend, Path(args.models_dir))
    entry = {"load_seconds": load_s}
    max_bytes = args.max_gb * GB
    peak = 0
    for wl, audios in workloads.items():
        timed_pass(run, audios, backend, max_bytes)  # warm-up
        passes = [timed_pass(run, audios, backend, max_bytes) for _ in range(args.repeats)]
        peak = max(peak, peak_bytes(backend))
        walls = [w for w, _ in passes]
        med = statistics.median(walls)
        lat = sorted(passes[walls.index(med)][1])
        entry[wl] = {
            "wall_seconds": walls,
            "median_wall_seconds": med,
            "rtfx_median": seconds[wl] / med,
            "latency_ms_p50": 1000 * lat[len(lat) // 2],
            "latency_ms_p90": 1000 * lat[int(len(lat) * 0.9)],
        }
    entry["peak_memory_gb"] = peak / GB
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--models-dir", default=str(HERE.parent / "eval-models"))
    parser.add_argument("--out", default=str(HERE / "results" / "speed.json"))
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--backends", nargs="+", default=BACKENDS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-gb", type=float, default=6.0, help="abort a run above this framework memory")
    parser.add_argument("--one", help="internal: run a single model/backend and print JSON")
    args = parser.parse_args()

    clips = load_corpus(Path(args.data))
    workloads = {"short": [c.audio for c in clips], "long": long_segments(clips)}
    seconds = {k: sum(len(a) for a in v) / 16000 for k, v in workloads.items()}

    if args.one:
        name, backend = args.one.split("/")
        print(json.dumps(run_one(name, backend, args, workloads, seconds)))
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = json.loads(out.read_text()) if out.exists() else {"runs": {}}
    result["machine"] = machine_info()
    result["protocol"] = {
        "repeats": args.repeats,
        "warmup": "one full pass per workload",
        "torch_threads": torch.get_num_threads(),
        "mlx_cache_limit_gb": 1,
        "isolation": "one subprocess per model/backend",
        "workloads": {k: {"items": len(v), "audio_seconds": seconds[k]} for k, v in workloads.items()},
    }

    for name in args.models:
        for backend in args.backends:
            key = f"{name}/{backend}"
            cmd = [sys.executable, __file__, "--one", key, "--repeats", str(args.repeats),
                   "--max-gb", str(args.max_gb), "--data", args.data, "--models-dir", args.models_dir]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"{key}: FAILED\n{proc.stderr[-2000:]}", flush=True)
                continue
            entry = json.loads(proc.stdout.strip().splitlines()[-1])
            result["runs"][key] = entry
            print(
                f"{name:12s} {backend:9s} load {entry['load_seconds']:5.1f}s | short RTFx {entry['short']['rtfx_median']:6.1f} "
                f"p50 {entry['short']['latency_ms_p50']:6.1f} ms | long RTFx {entry['long']['rtfx_median']:6.1f} "
                f"p50 {entry['long']['latency_ms_p50']:6.1f} ms | peak {entry['peak_memory_gb']:.2f} GB",
                flush=True,
            )
            out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
