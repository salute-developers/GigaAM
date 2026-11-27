#!/usr/bin/env python3
"""Render results/RESULTS.md from summary.json, speed.json and per-clip jsonl."""
import json
import re
from pathlib import Path

from common import HERE, corpus_error_rate, normalize

MODELS = ["v3_ctc", "v3_rnnt", "v3_e2e_ctc", "v3_e2e_rnnt"]
FORMATTED = re.compile(r"[0-9a-zA-Z]")


def pct(x):
    return f"{100 * x:.2f}%"


def main():
    res = HERE / "results"
    summary = json.loads((res / "summary.json").read_text())
    speed = json.loads((res / "speed.json").read_text()) if (res / "speed.json").exists() else None
    rows = {m: {r["id"]: r for r in map(json.loads, open(res / f"{m}.jsonl"))} for m in MODELS}

    # Common subset: clips where neither E2E transcript has digits or Latin letters.
    ids = sorted(rows["v3_ctc"])
    plain = [i for i in ids if not any(FORMATTED.search(rows[m][i]["pytorch"]) for m in ("v3_e2e_ctc", "v3_e2e_rnnt"))]

    m = summary["machine"]
    lines = [
        "# GigaAM MLX eval results",
        "",
        f"Machine: {m['chip']}, {m['memory_gb']} GB, macOS {m['macos']}; "
        f"Python {m['python']}, mlx {m['mlx']}, torch {m['torch']}, torchaudio {m['torchaudio']}.",
        "",
        f"Corpus: Golos Crowd test, {summary['corpus']['clips']} clips, "
        f"{summary['corpus']['seconds']:.2f} s ({summary['v3_ctc']['pytorch']['wer']['ref_units']} reference words); "
        "the sample of github.com/garrrikkotua/gigaam-vs-whisper-russian, verified by SHA-256.",
        "",
        "## Parity: MLX vs PyTorch",
        "",
        "Reference: upstream PyTorch GigaAM on CPU in fp32. MLX: the fp16 file loaded with the `load_model`",
        "default (weights upcast to fp32 at load). The upstream encoder weights are stored in fp16, so fp16",
        "conversion loses nothing there; only the small CTC head weights are fp32 in the checkpoint.",
        "",
        "| Model | Identical transcripts | Encoder max \\|Δ\\| (median / max over clips) | CTC frame labels identical |",
        "|---|---:|---:|---:|",
    ]
    for name in MODELS:
        s = summary[name]["mlx_fp16"]
        frames = pct(s["ctc_frame_agreement"]) if "ctc_frame_agreement" in s else "—"
        lines.append(
            f"| {name} | {s['exact_match_vs_pytorch']}/{summary[name]['clips']} | "
            f"{s['enc_max_abs_diff']['median']:.1e} / {s['enc_max_abs_diff']['max']:.1e} | {frames} |"
        )
    lines += [
        "",
        f"Median max |encoder output| is {summary['v3_ctc']['enc_abs_max_median']:.2f} (v3_ctc) for scale.",
        "",
        "## Accuracy (MLX fp16; identical to PyTorch, see above)",
        "",
        "Normalization for every hypothesis: lowercase, ё→е, numbers → words (num2words), punctuation and",
        "symbols removed. Corpus WER = total word edits / total reference words.",
        "",
        f"The last column restricts to the {len(plain)} clips where neither E2E transcript contains digits",
        "or Latin letters (E2E writes \"HD\", \"4K\", \"15-й\", phone numbers as digit groups; Golos spells them",
        "out in Cyrillic, and no normalizer maps one onto the other reliably).",
        "",
        "| Model | WER (S/I/D) | CER | WER, plain subset |",
        "|---|---:|---:|---:|",
    ]
    for name in MODELS:
        s = summary[name]["mlx_fp16"]
        w = s["wer"]
        sub = corpus_error_rate([(rows[name][i]["reference"], normalize(rows[name][i]["mlx_fp16"])) for i in plain])
        lines.append(
            f"| {name} | {pct(w['rate'])} ({w['substitutions']}/{w['insertions']}/{w['deletions']}) | "
            f"{pct(s['cer']['rate'])} | {pct(sub['rate'])} ({sub['errors']}/{sub['ref_units']}) |"
        )

    if speed:
        p = speed["protocol"]
        wl = p["workloads"]
        lines += [
            "",
            "## Speed",
            "",
            f"In-memory audio, one warm-up pass, then {p['repeats']} timed passes; median pass reported.",
            f"`short`: {wl['short']['items']} clips one by one ({wl['short']['audio_seconds']:.1f} s). "
            f"`long`: {wl['long']['items']} segments of ≈20 s made by concatenating consecutive clips "
            f"({wl['long']['audio_seconds']:.1f} s). RTFx = audio seconds / wall seconds. "
            f"PyTorch CPU uses {p['torch_threads']} threads; PyTorch MPS runs the upstream fp16-autocast encoder. "
            "`mlx_fp16`: fp16 file, weights upcast to fp32 at load (default); `mlx_fp16_stored`: fp16 params kept "
            "(`load_model(..., dtype=None)`); `mlx_fp32`: fp32 file. MLX buffer cache capped at 1 GB; one subprocess "
            "per row. Peak memory: MLX peak allocation / MPS driver allocation / CPU max RSS — not directly comparable.",
            "",
            "| Model | Backend | Load, s | short RTFx | short p50, ms | long RTFx | long p50, ms | Peak mem, GB |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        order = {b: i for i, b in enumerate(["mlx_fp16", "mlx_fp16_stored", "mlx_fp32", "torch_mps", "torch_cpu"])}
        runs = sorted(speed["runs"].items(), key=lambda kv: (MODELS.index(kv[0].split("/")[0]), order[kv[0].split("/")[1]]))
        for key, e in runs:
            name, backend = key.split("/")
            lines.append(
                f"| {name} | {backend} | {e['load_seconds']:.1f} | {e['short']['rtfx_median']:.1f} | "
                f"{e['short']['latency_ms_p50']:.1f} | {e['long']['rtfx_median']:.1f} | {e['long']['latency_ms_p50']:.1f} | "
                f"{e.get('peak_memory_gb', float('nan')):.2f} |"
            )

    (res / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print((res / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
