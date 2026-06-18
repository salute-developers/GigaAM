#!/usr/bin/env python3
"""A/B сравнение сегментации речи: самописный RMS-VAD против silero-vad.

Обе ветки используют один и тот же GigaAM и одну и ту же нарезку по длине
(``asr.split_interval``); меняется только детектор границ речи. Это изолирует
вклад именно VAD. Эталонной расшифровки нет, поэтому судим по прокси-метрикам:
скорость, число и длина кусочков, доля пойманной речи и объём текста. Сами
тексты двух веток складываются рядом для ручного просмотра.

Пример::

    .venv/bin/python mac_transcriber/scripts/ab_vad_eval.py \\
        --max-meetings 3 --limit-seconds 600
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gigaam  # noqa: E402

from mac_transcriber import asr  # noqa: E402

DEFAULT_ROOT = Path(".local/mac_transcriber")
DEFAULT_OUTPUT_ROOT = Path(".local/vad_ab_eval")


@dataclass(frozen=True)
class Interval:
    """Границы куска речи в сэмплах (16 кГц)."""

    start: int
    end: int


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    meetings = select_meetings(args.root, args.meeting, args.max_meetings)
    if not meetings:
        raise SystemExit(f"Не нашёл audio.m4a под {args.root}/meetings")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # silero грузим один раз; GigaAM кешируется внутри asr.transcribe_segments.
    from silero_vad import load_silero_vad

    silero_model = load_silero_vad()

    results: list[dict[str, Any]] = []
    for meeting_dir in meetings:
        print(f"== {meeting_dir.name}", flush=True)
        result = run_meeting(
            meeting_dir=meeting_dir,
            silero_model=silero_model,
            output_root=output_root,
            limit_seconds=args.limit_seconds,
            model_name=args.model,
            cache_dir=args.cache_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
        results.append(result)
        write_json(output_root / "summary.json", {"results": results})
        write_summary_markdown(output_root / "summary.md", results)
        print_meeting_result(result)

    print(f"\nsummary: {output_root / 'summary.md'}", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--meeting",
        action="append",
        help="Id встречи (имя папки). Можно повторять. По умолчанию — авто-выбор.",
    )
    parser.add_argument("--max-meetings", type=int, default=2)
    parser.add_argument(
        "--limit-seconds",
        type=float,
        default=600.0,
        help="Обрезать каждый файл до N секунд (0 = весь файл).",
    )
    parser.add_argument("--model", default="v3_e2e_rnnt")
    parser.add_argument("--cache-dir", default="/tmp/gigaam-cache")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args(argv)


def select_meetings(
    root: Path, names: list[str] | None, max_meetings: int
) -> list[Path]:
    meetings_dir = root.expanduser().resolve() / "meetings"
    if names:
        chosen = [meetings_dir / name for name in names]
        return [path for path in chosen if (path / "input" / "audio.m4a").exists()]
    with_audio = sorted(
        path.parent.parent for path in meetings_dir.glob("*/input/audio.m4a")
    )
    return with_audio[:max_meetings]


def run_meeting(
    *,
    meeting_dir: Path,
    silero_model: Any,
    output_root: Path,
    limit_seconds: float,
    model_name: str,
    cache_dir: str,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    audio_path = meeting_dir / "input" / "audio.m4a"
    wav = gigaam.load_audio(str(audio_path)).cpu().float()
    if limit_seconds and limit_seconds > 0:
        wav = wav[: int(limit_seconds * asr.SR)]
    total_samples = wav.numel()
    audio_seconds = total_samples / asr.SR

    # rms считаем один раз: нужен обеим веткам для split_interval и для метрик.
    mask, rms, _threshold = asr.activity_mask(wav)

    case_dir = output_root / meeting_dir.name
    case_dir.mkdir(parents=True, exist_ok=True)

    variants: dict[str, dict[str, Any]] = {}
    for variant_name, detector in (
        ("homemade", lambda: homemade_intervals(mask, rms, total_samples)),
        ("silero", lambda: silero_intervals(wav, silero_model, rms, total_samples)),
    ):
        vad_start = time.perf_counter()
        intervals = detector()
        vad_seconds = time.perf_counter() - vad_start

        segments = build_segments(wav, intervals, track_name=audio_path.name)
        asr_start = time.perf_counter()
        segments = asr.transcribe_segments(
            segments,
            model_name=model_name,
            cache_dir=cache_dir,
            device=device,
            batch_size=batch_size,
        )
        asr_seconds = time.perf_counter() - asr_start

        transcript = render_transcript(segments)
        (case_dir / f"{variant_name}.md").write_text(transcript, encoding="utf-8")
        variants[variant_name] = summarize_variant(
            intervals=intervals,
            segments=segments,
            audio_seconds=audio_seconds,
            vad_seconds=vad_seconds,
            asr_seconds=asr_seconds,
            transcript=transcript,
        )

    comparison = compare_variants(variants["homemade"], variants["silero"])
    return {
        "meeting_id": meeting_dir.name,
        "audio_seconds": round(audio_seconds, 1),
        "limit_seconds": limit_seconds,
        "variants": variants,
        "comparison": comparison,
    }


def homemade_intervals(mask: Any, rms: Any, total_samples: int) -> list[Interval]:
    """Production-путь: energy mask -> merge/pad/min-len -> split по длине."""
    intervals: list[Interval] = []
    for start, end in asr.mask_to_intervals(mask, total_samples):
        for part_start, part_end in asr.split_interval(start, end, rms):
            intervals.append(Interval(part_start, part_end))
    return intervals


def silero_intervals(
    wav: Any, silero_model: Any, rms: Any, total_samples: int
) -> list[Interval]:
    """silero находит речь, дальше — те же merge/pad/min-len/split, что у самодела."""
    from silero_vad import get_speech_timestamps

    raw = get_speech_timestamps(wav, silero_model, sampling_rate=asr.SR)
    speech = [(int(item["start"]), int(item["end"])) for item in raw]
    return refine_intervals(speech, rms, total_samples)


def refine_intervals(
    raw: list[tuple[int, int]], rms: Any, total_samples: int
) -> list[Interval]:
    """Повторяет постобработку ``asr.mask_to_intervals`` для произвольных границ."""
    merge_gap = int(asr.MERGE_GAP_S * asr.SR)
    merged: list[list[int]] = []
    for start, end in sorted(raw):
        if not merged or start - merged[-1][1] > merge_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    pad = int(asr.PAD_S * asr.SR)
    min_len = int(asr.MIN_SEGMENT_S * asr.SR)
    intervals: list[Interval] = []
    for start, end in merged:
        start_sample = max(0, start - pad)
        end_sample = min(total_samples, end + pad)
        if end_sample - start_sample < min_len:
            continue
        for part_start, part_end in asr.split_interval(start_sample, end_sample, rms):
            intervals.append(Interval(part_start, part_end))
    return intervals


def build_segments(
    wav: Any, intervals: list[Interval], *, track_name: str
) -> list[asr.Segment]:
    segments: list[asr.Segment] = []
    for interval in intervals:
        segments.append(
            asr.Segment(
                speaker="Speaker",
                track=track_name,
                start=interval.start / asr.SR,
                end=interval.end / asr.SR,
                wav=wav[interval.start : interval.end].clone(),
            )
        )
    segments.sort(key=lambda segment: (segment.start, segment.end))
    return segments


def summarize_variant(
    *,
    intervals: list[Interval],
    segments: list[asr.Segment],
    audio_seconds: float,
    vad_seconds: float,
    asr_seconds: float,
    transcript: str,
) -> dict[str, Any]:
    lengths = [(interval.end - interval.start) / asr.SR for interval in intervals]
    speech_seconds = sum(lengths)
    text = " ".join(segment.text for segment in segments if segment.text)
    return {
        "n_segments": len(intervals),
        "speech_seconds": round(speech_seconds, 1),
        "coverage": round(speech_seconds / audio_seconds, 3) if audio_seconds else 0.0,
        "seg_len_mean": round(statistics.mean(lengths), 2) if lengths else 0.0,
        "seg_len_median": round(statistics.median(lengths), 2) if lengths else 0.0,
        "seg_len_min": round(min(lengths), 2) if lengths else 0.0,
        "seg_len_max": round(max(lengths), 2) if lengths else 0.0,
        "vad_seconds": round(vad_seconds, 2),
        "asr_seconds": round(asr_seconds, 1),
        "chars": len(text),
        "words": len(text.split()),
    }


def compare_variants(
    homemade: dict[str, Any], silero: dict[str, Any]
) -> dict[str, Any]:
    return {
        "n_segments_delta": silero["n_segments"] - homemade["n_segments"],
        "speech_seconds_delta": round(
            silero["speech_seconds"] - homemade["speech_seconds"], 1
        ),
        "coverage_delta": round(silero["coverage"] - homemade["coverage"], 3),
        "asr_seconds_delta": round(silero["asr_seconds"] - homemade["asr_seconds"], 1),
        "words_delta": silero["words"] - homemade["words"],
    }


def render_transcript(segments: list[asr.Segment]) -> str:
    lines = []
    for segment in segments:
        stamp = f"[{asr.fmt_time(segment.start)} - {asr.fmt_time(segment.end)}]"
        lines.append(f"{stamp} {segment.text}".rstrip())
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_summary_markdown(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# VAD A/B: самодел (RMS) против silero-vad",
        "",
        "| Встреча | Аудио, с | Вариант | Кусков | Речь, с | Покрытие | Длина med/max | VAD, с | ASR, с | Слов |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for result in results:
        for variant_name in ("homemade", "silero"):
            variant = result["variants"][variant_name]
            lines.append(
                f"| {result['meeting_id'][:18]} | {result['audio_seconds']} | "
                f"{variant_name} | {variant['n_segments']} | {variant['speech_seconds']} | "
                f"{variant['coverage']} | {variant['seg_len_median']}/{variant['seg_len_max']} | "
                f"{variant['vad_seconds']} | {variant['asr_seconds']} | {variant['words']} |"
            )
        comparison = result["comparison"]
        lines.append(
            f"| {result['meeting_id'][:18]} | | **delta (silero−самодел)** | "
            f"{comparison['n_segments_delta']} | {comparison['speech_seconds_delta']} | "
            f"{comparison['coverage_delta']} | | | {comparison['asr_seconds_delta']} | "
            f"{comparison['words_delta']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_meeting_result(result: dict[str, Any]) -> None:
    homemade = result["variants"]["homemade"]
    silero = result["variants"]["silero"]
    print(
        f"   {result['meeting_id'][:18]}: "
        f"homemade segs={homemade['n_segments']} cov={homemade['coverage']} "
        f"asr={homemade['asr_seconds']}s words={homemade['words']} | "
        f"silero segs={silero['n_segments']} cov={silero['coverage']} "
        f"asr={silero['asr_seconds']}s words={silero['words']}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
