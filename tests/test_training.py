import csv
import gc
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gigaam  # noqa: E402
from gigaam.preprocess import SAMPLE_RATE  # noqa: E402
from gigaam.utils import download_long_audio  # noqa: E402

TRAIN_UTILS_DIR = Path(__file__).resolve().parents[1] / "train_utils"
MAX_SAMPLES = 2


def _write_manifest(
    manifest_path: Path, audio_paths: list[Path], texts: list[str]
) -> None:
    rows = ["path\tduration\ttranscription\ttranscription_e2e\ttranscription_raw"]
    for audio_path, text in zip(audio_paths, texts):
        duration = sf.info(audio_path).duration
        rows.append(f"{audio_path}\t{duration:.3f}\t{text}\t{text}\t{text}")
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _run_python(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        return subprocess.run(
            [sys.executable, *args],
            cwd=cwd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(f"Subprocess failed:\n{error.stdout}\n{error.stderr}")


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
        "--accelerator",
        "cpu",
        "--devices",
        "1",
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


def _head_out_classes(model: "gigaam.GigaAMASR") -> int:
    """Number of output classes of the ASR head."""
    if hasattr(model.head, "decoder_layers"):  # CTCHead
        return model.head.decoder_layers[0].out_channels
    return model.head.joint.joint_net[-1].out_features  # RNNTHead


@pytest.mark.parametrize(
    ("head", "extra_args"),
    [
        ("ctc", ["--raw_text"]),  # normalized charwise labels
        ("rnnt", ["--rnnt_subbatch_size", "1"]),  # e2e mode: verbatim labels
    ],
)
def test_ssl_finetune_cpu(
    tmp_path: Path, pseudo_labeled_dataset: Path, head: str, extra_args: list[str]
) -> None:
    """
    Fine-tune a new head on top of an SSL backbone, then verify
    the checkpoint is self-contained: it reloads via ``gigaam.load_model`` as a ``GigaAMASR``.
    """
    model_name = "multilingual_ssl"
    exp_name = f"pytest_ssl_ft_{head}"
    output_dir = tmp_path / "artifacts"
    vocab_path = tmp_path / "vocab.json"

    try:
        _run_python(
            [
                "train.py",
                "--model_name",
                model_name,
                "--head",
                head,
                "--train_manifest",
                str(pseudo_labeled_dataset),
                "--val_manifest",
                str(pseudo_labeled_dataset),
                "--output_dir",
                str(output_dir),
                "--exp_name",
                exp_name,
                "--build_vocab_from_manifest",
                "--save_vocab",
                str(vocab_path),
                "--batch_size",
                "1",
                "--eval_batch_size",
                str(MAX_SAMPLES),
                "--num_workers",
                "0",
                "--precision",
                "32",
                "--accelerator",
                "cpu",
                "--devices",
                "1",
                "--max_steps",
                "2",
                "--val_check_steps",
                "2",
                "--disable_tqdm",
                "--log_every_n_steps",
                "1",
                "--skip_initial_validation",
                "--save_top_k",
                "1",
                "--freeze_encoder_epochs",
                "-1",
            ]
            + extra_args,
            cwd=TRAIN_UTILS_DIR,
        )

        ckpt_dir = output_dir / "models" / exp_name
        checkpoints = sorted(ckpt_dir.glob("*.ckpt"))
        assert checkpoints, f"No checkpoints found in {ckpt_dir}"

        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        assert " " in vocab and len(vocab) > 1, f"Unexpected derived vocab: {vocab}"

        model = gigaam.load_model(str(checkpoints[0]), device="cpu")
        assert isinstance(model, gigaam.GigaAMASR)
        assert list(model.decoding.tokenizer.vocab) == vocab
        assert model.decoding.blank_id == len(vocab)
        assert _head_out_classes(model) == len(vocab) + 1

        with open(pseudo_labeled_dataset) as f:
            short_audio = next(
                row["path"]
                for row in csv.DictReader(f, delimiter="\t")
                if float(row["duration"]) <= 25
            )
        assert isinstance(str(model.transcribe(short_audio)), str)
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_finetune_manifest_targets(tmp_path):
    from gigaam.decoding import Tokenizer
    from train_utils.utils import FineTuneDataset

    manifest = tmp_path / "labels.tsv"
    manifest.write_text(
        "path\tduration\ttranscription_e2e\ttranscription_raw\n"
        "a.wav\t1\tA\ta\nb.wav\t1\tB\t\n"
        "c.wav\t1\t\tb\nd.wav\t1\t\t\n"
    )
    tokenizers = {"e2e": Tokenizer(["A", "B"]), "raw": Tokenizer(["b", "a"])}
    dataset = FineTuneDataset(str(manifest), tokenizers)
    assert len(dataset) == 3
    assert dataset.targets[0]["e2e"].tolist() == [0]
    assert dataset.targets[0]["raw"].tolist() == [1]
    assert len(FineTuneDataset(str(manifest), {"raw": tokenizers["raw"]})) == 2
    manifest.write_text("path\tduration\ttranscription_e2e\na.wav\t1\t\n")
    with pytest.raises(ValueError, match="No eligible"):
        FineTuneDataset(str(manifest), tokenizers)
    manifest.write_text("path\tduration\ttranscription\na.wav\t1\ta\n")
    with pytest.raises(ValueError, match="transcription_e2e or transcription_raw"):
        FineTuneDataset(str(manifest), tokenizers)


@pytest.mark.parametrize(
    "reference, hypothesis, expected",
    [
        ("Привет, мир!", "привет мир", (100.0, 0.0, 2, 2, 0, 2)),
        ("Ютуб", "youtube", (100.0, 0.0, 1, 1, 0, 1)),
        ("a b", "a", (50.0, 50.0, 1, 2, 1, 2)),
        ("a", "a b c", (200.0, 200.0, 2, 1, 2, 1)),
        ("a", "", (100.0, 100.0, 1, 1, 1, 1)),
        ("  a  b  ", "a b", (0.0, 0.0, 0, 2, 0, 2)),
    ],
)
def test_compute_wer(reference, hypothesis, expected):
    from train_utils.utils import compute_wer

    assert compute_wer([{"text": reference, "pred_text": hypothesis}]) == expected


def test_finetune_eval_retains_nonempty_all_oov_references(tmp_path):
    from gigaam.decoding import Tokenizer
    from train_utils.utils import FineTuneDataset, compute_wer

    manifest = tmp_path / "all_oov.tsv"
    sf.write(tmp_path / "clip.wav", [0.0] * 800, SAMPLE_RATE)
    manifest.write_text(
        "path\tduration\ttranscription_e2e\nclip.wav\t1\tΩЖ\n",
        encoding="utf-8",
    )
    dataset = FineTuneDataset(
        str(manifest), {"e2e": Tokenizer(["A"])}, keep_empty_targets=True
    )
    assert len(dataset) == 1
    assert dataset.texts == [{"e2e": "ΩЖ"}]
    assert dataset.targets[0]["e2e"].numel() == 0
    labels = dataset.collate_fn([dataset[0]])[2]
    eligible = [i for i in range(len(labels["e2e"][0])) if dataset.texts[i]["e2e"]]
    assert eligible == [0]
    result = compute_wer([{"text": dataset.texts[eligible[0]]["e2e"], "pred_text": ""}])
    assert result[2:4] == (1, 1)


@pytest.mark.parametrize("kind", ["ctc", "rnnt"])
def test_dual_finetune_gradients_and_reload(tmp_path, kind):
    import torch
    from test_streaming import tiny_model

    from train_utils.module import GigaAMFineTuner
    from train_utils.ssl_finetune import _head_cfg

    cfg = tiny_model().cfg
    cfg.heads, cfg.decodings = {}, {}
    for name, vocab in (("e2e", ["A", "B", "C"]), ("raw", ["b", "a"])):
        head, decoding = _head_cfg(kind, vocab, 8, 8, 1, 8)
        cfg.heads[name], cfg.decodings[name] = head, decoding
    del cfg.head, cfg.decoding
    model = gigaam.GigaAMDualASR(cfg)
    model.default_language = "ru"
    tuner = GigaAMFineTuner(model, 2, cli_args={"model_name": "tiny"})
    encoded, lengths = tuner._encode(torch.randn(2, 771), torch.tensor([771, 700]))
    targets = torch.tensor([[0], [1]])
    labels = {
        "e2e": (targets, torch.tensor([1, 1])),
        "raw": (targets, torch.tensor([0, 0])),
    }
    loss = tuner._losses(encoded, lengths, labels)
    loss.backward()
    assert loss.isfinite()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in tuner.heads["e2e"].parameters()
    )
    if kind != "ctc":
        for module in (tuner.heads["e2e"].decoder, tuner.heads["e2e"].joint):
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in module.parameters()
            )
    assert all(p.grad is None for p in tuner.heads["raw"].parameters())
    raw_before = [p.detach().clone() for p in tuner.heads["raw"].parameters()]
    torch.optim.AdamW(tuner.parameters(), lr=1e-3, weight_decay=0.1).step()
    for before, parameter in zip(raw_before, tuner.heads["raw"].parameters()):
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in tuner.encoder.parameters()
    )
    tuner.zero_grad(set_to_none=True)
    encoded, lengths = tuner._encode(torch.randn(2, 771), torch.tensor([771, 700]))
    labels["raw"] = (targets, torch.tensor([1, 0]))
    mixed_loss = tuner._losses(encoded, lengths, labels)
    ignored_targets = targets.clone()
    ignored_targets[1] = 99
    labels["raw"] = (ignored_targets, torch.tensor([1, 0]))
    torch.testing.assert_close(tuner._losses(encoded, lengths, labels), mixed_loss)
    mixed_loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in tuner.heads["raw"].parameters()
    )
    checkpoint = tmp_path / "finetuned.ckpt"
    torch.save(
        {"state_dict": tuner.state_dict(), "hyper_parameters": dict(tuner.hparams)},
        checkpoint,
    )
    restored = gigaam.load_model(str(checkpoint), device="cpu", fp16_encoder=False)
    assert isinstance(restored, gigaam.GigaAMDualASR)
    assert restored.default_language == "ru"
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor)
    assert (
        restored.decodings["e2e"].tokenizer.vocab
        != restored.decodings["raw"].tokenizer.vocab
    )
