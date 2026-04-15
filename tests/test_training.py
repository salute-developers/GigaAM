import gc
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import soundfile as sf

import gigaam
from gigaam.preprocess import SAMPLE_RATE
from gigaam.utils import download_long_audio

TRAIN_UTILS_DIR = Path(__file__).resolve().parents[1] / "train_utils"
MAX_SAMPLES = 2


def _write_manifest(
    manifest_path: Path, audio_paths: list[Path], texts: list[str]
) -> None:
    rows = ["path\tduration\ttranscription"]
    for audio_path, text in zip(audio_paths, texts):
        duration = sf.info(audio_path).duration
        rows.append(f"{audio_path}\t{duration:.3f}\t{text}")
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _run_python(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _extract_e2e_wer(output: str) -> float:
    match = re.search(r"WER e2e:\s+([0-9.]+)%", output)
    assert match, f"WER e2e not found in output:\n{output}"
    return float(match.group(1))


def _clear_gigaam_cache_checkpoints() -> None:
    cache_dir = Path.home() / ".cache" / "gigaam"
    if not cache_dir.is_dir():
        return
    for ckpt in cache_dir.glob("*.ckpt"):
        ckpt.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def pseudo_labeled_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("pseudo_labels")
    audio_dir = root / "audio"
    audio_dir.mkdir()

    audio_path = download_long_audio()
    model = gigaam.load_model("v3_ctc", device="cpu", fp16_encoder=False)
    labels = model.transcribe_longform(audio_path, fr_batch_size=8, fr_num_workers=0)
    del model
    gc.collect()
    audio, sr = sf.read(audio_path, dtype="float32")
    assert sr == SAMPLE_RATE

    assert labels.segments

    segment_paths = []
    texts = []
    for idx, labeled in enumerate(labels.segments[:MAX_SAMPLES]):
        segment_path = audio_dir / f"{idx:02d}.wav"
        start_idx = int(labeled.start * SAMPLE_RATE)
        end_idx = int(labeled.end * SAMPLE_RATE)
        sf.write(segment_path, audio[start_idx:end_idx], SAMPLE_RATE)
        segment_paths.append(segment_path)
        texts.append(str(labeled.text).lower())

    manifest_path = root / "manifest.tsv"
    _write_manifest(manifest_path, segment_paths, texts)
    return manifest_path


@pytest.mark.parametrize(
    ("model_name", "extra_args", "max_steps", "e2e_threshold", "min_e2e_gain"),
    [
        ("v3_e2e_ctc", ["--lr", "5e-4", "--activation_checkpointing"], 3, 20.0, 20.0),
        (
            "v3_e2e_rnnt",
            ["--rnnt_subbatch_size", "1", "--lr", "5e-4", "--activation_checkpointing"],
            6,
            20.0,
            20.0,
        ),
    ],
)
def test_training_and_eval_on_cpu(
    tmp_path: Path,
    pseudo_labeled_dataset: Path,
    model_name: str,
    extra_args: list[str],
    max_steps: int,
    e2e_threshold: float,
    min_e2e_gain: float,
) -> None:
    exp_name = f"pytest_{model_name}"
    output_dir = tmp_path / "artifacts"

    try:
        _run_training_case(
            model_name,
            extra_args,
            max_steps,
            e2e_threshold,
            min_e2e_gain,
            pseudo_labeled_dataset,
            output_dir,
            exp_name,
        )
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)


def _run_training_case(
    model_name: str,
    extra_args: list[str],
    max_steps: int,
    e2e_threshold: float,
    min_e2e_gain: float,
    pseudo_labeled_dataset: Path,
    output_dir: Path,
    exp_name: str,
) -> None:
    baseline_eval = _run_python(
        [
            "eval.py",
            "--model_name",
            model_name,
            "--eval_manifest",
            str(pseudo_labeled_dataset),
            "--batch_size",
            str(MAX_SAMPLES),
            "--num_workers",
            "0",
            "--device",
            "cpu",
            "--disable_tqdm",
        ],
        cwd=TRAIN_UTILS_DIR,
    )
    baseline_e2e = _extract_e2e_wer(baseline_eval.stdout)

    # Сlear gigaam cache checkpoints to save disk space in github actions
    _clear_gigaam_cache_checkpoints()

    train_cmd = [
        "train.py",
        "--model_name",
        model_name,
        "--train_manifest",
        str(pseudo_labeled_dataset),
        "--val_manifest",
        str(pseudo_labeled_dataset),
        "--output_dir",
        str(output_dir),
        "--exp_name",
        exp_name,
        "--batch_size",
        "1",
        "--eval_batch_size",
        str(MAX_SAMPLES),
        "--num_workers",
        "0",
        "--precision",
        "32",
        "--max_steps",
        str(max_steps),
        "--val_check_steps",
        str(max_steps),
        "--disable_tqdm",
        "--log_every_n_steps",
        "1",
        "--skip_initial_validation",
        "--save_top_k",
        "1",
    ]
    _run_python(train_cmd + extra_args, cwd=TRAIN_UTILS_DIR)

    ckpt_dir = output_dir / "models" / exp_name
    checkpoints = sorted(ckpt_dir.glob("*.ckpt"))
    assert checkpoints, f"No checkpoints found in {ckpt_dir}"

    eval_run = _run_python(
        [
            "eval.py",
            "--checkpoint",
            str(checkpoints[0]),
            "--eval_manifest",
            str(pseudo_labeled_dataset),
            "--batch_size",
            str(MAX_SAMPLES),
            "--num_workers",
            "0",
            "--device",
            "cpu",
            "--disable_tqdm",
        ],
        cwd=TRAIN_UTILS_DIR,
    )

    e2e_wer = _extract_e2e_wer(eval_run.stdout)
    assert e2e_wer <= e2e_threshold
    assert baseline_e2e - e2e_wer >= min_e2e_gain


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
