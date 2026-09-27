"""Shared helpers for the MLX eval: corpus loading, text normalization, WER."""
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
from num2words import num2words

HERE = Path(__file__).parent
DEFAULT_DATA = HERE / "data" / "golos_crowd_200"
sys.path.insert(0, str(HERE.parent))  # gigaam_mlx


@dataclass
class Clip:
    id: int
    path: Path
    duration: float
    reference: str
    audio: np.ndarray  # float32, 16 kHz mono


def load_corpus(data_dir: Path = DEFAULT_DATA) -> List[Clip]:
    clips = []
    for line in open(data_dir / "manifest.jsonl"):
        rec = json.loads(line)
        path = data_dir / rec["audio"]
        audio, sr = sf.read(path, dtype="int16")
        assert sr == 16000 and audio.ndim == 1, path
        clips.append(
            Clip(
                id=rec["id"],
                path=path,
                duration=rec["duration"],
                reference=rec["reference"],
                # same scaling as gigaam.preprocess.load_audio (s16le / 32768)
                audio=audio.astype(np.float32) / 32768.0,
            )
        )
    return clips


# ─────────────────────────── normalization ───────────────────────────
# Golos references are lowercase Cyrillic words, no punctuation, no "ё",
# numbers spelled out. CTC/RNNT output already looks like that; E2E models
# output cased text with punctuation and digits. Every hypothesis goes through
# the same function, so char-wise outputs are (near) unchanged.

_NUMBER = re.compile(r"\d+(?:[   ]\d{3})*(?:[.,]\d+)?")


def _number_to_words(match: re.Match) -> str:
    token = re.sub(r"[   ]", "", match.group(0)).replace(",", ".")
    try:
        value = float(token) if "." in token else int(token)
        return " " + num2words(value, lang="ru") + " "
    except (ValueError, OverflowError, NotImplementedError):
        return " " + token + " "


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = _NUMBER.sub(_number_to_words, text)
    text = re.sub(r"[^\w\s]|_", " ", text)  # punctuation, hyphens, symbols
    return " ".join(text.split())


# ─────────────────────────── metrics ───────────────────────────


def edit_ops(ref: List[str], hyp: List[str]) -> Dict[str, int]:
    """Levenshtein alignment counts: substitutions, insertions, deletions."""
    n, m = len(ref), len(hyp)
    # dp[i][j] = (cost, S, I, D)
    prev = [(j, 0, j, 0) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, 0, i)]
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                best = prev[j - 1]
            else:
                c_sub, c_ins, c_del = prev[j - 1], cur[j - 1], prev[j]
                best = min(
                    (c_sub[0] + 1, c_sub[1] + 1, c_sub[2], c_sub[3]),
                    (c_ins[0] + 1, c_ins[1], c_ins[2] + 1, c_ins[3]),
                    (c_del[0] + 1, c_del[1], c_del[2], c_del[3] + 1),
                )
            cur.append(best)
        prev = cur
    cost, s, i_, d = prev[m]
    return {"errors": cost, "substitutions": s, "insertions": i_, "deletions": d}


def corpus_error_rate(pairs, unit: str = "word") -> Dict[str, float]:
    """Corpus-level rate: total edits / total reference units (not a mean of
    per-utterance rates), matching the garrrikkotua benchmark's verify.py."""
    total = {"errors": 0, "substitutions": 0, "insertions": 0, "deletions": 0}
    ref_units = 0
    for ref, hyp in pairs:
        r = ref.split() if unit == "word" else list(ref.replace(" ", ""))
        h = hyp.split() if unit == "word" else list(hyp.replace(" ", ""))
        for k, v in edit_ops(r, h).items():
            total[k] += v
        ref_units += len(r)
    return {**total, "ref_units": ref_units, "rate": total["errors"] / ref_units}


def machine_info() -> Dict[str, str]:
    import platform
    import subprocess
    from importlib.metadata import version

    def sysctl(key):
        return subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True
        ).stdout.strip()

    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "memory_gb": str(int(sysctl("hw.memsize") or 0) // 2**30),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "mlx": version("mlx"),
        "torch": version("torch"),
        "torchaudio": version("torchaudio"),
    }
