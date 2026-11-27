import csv
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import editdistance
import soundfile as sf
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, TQDMProgressBar
from tqdm.auto import tqdm

from gigaam.preprocess import SAMPLE_RATE
from gigaam.utils import AudioDataset, normalize_raw_text


class FineTuneDataset(AudioDataset):
    def __init__(
        self, manifest, tokenizers, raw_text=False, keep_empty_targets=False, **kwargs
    ):
        self.tokenizers = tokenizers
        self.encoders = {
            name: self._make_encoder(tokenizer)
            for name, tokenizer in tokenizers.items()
        }
        self.targets = []
        self.texts = []
        self.normalize_raw = raw_text
        self.keep_empty_targets = keep_empty_targets
        super().__init__(manifest, **kwargs)

    def _load_manifest(self, manifest_path, min_duration, max_duration):
        directory = Path(manifest_path).resolve().parent
        skipped = 0
        with open(manifest_path, encoding="utf-8") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if not {"transcription_e2e", "transcription_raw"}.intersection(
                reader.fieldnames or []
            ):
                raise ValueError(
                    "Fine-tuning requires transcription_e2e or transcription_raw column"
                )
            for row in reader:
                duration = float(row["duration"])
                if not self._duration_ok(duration, min_duration, max_duration):
                    continue
                targets, texts = {}, {}
                for name, encode in self.encoders.items():
                    text = (row.get(f"transcription_{name}") or "").strip()
                    if name == "raw" and self.normalize_raw:
                        text = normalize_raw_text(text)
                    texts[name] = text
                    targets[name] = torch.tensor(encode(text), dtype=torch.long)
                if not self.keep_empty_targets and not any(
                    target.numel() for target in targets.values()
                ):
                    skipped += 1
                    continue
                path = Path(row["path"])
                self._append_sample(str(directory / path), duration)
                self.targets.append(targets)
                self.texts.append(texts)
        print(f"Skipped {skipped} rows without eligible targets in {manifest_path}")
        if not self.samples:
            raise ValueError(f"No eligible fine-tuning samples in {manifest_path}")

    def __getitem__(self, idx):
        return self._load_audio(self.samples[idx].item), self.targets[idx]

    def collate_fn(self, batch):
        wavs, targets = zip(*batch)
        wav_pad, wav_lens = self.collate(list(wavs))
        labels = {
            name: self.collate([target[name] for target in targets])
            for name in self.tokenizers
        }
        return wav_pad, wav_lens, labels


@lru_cache(maxsize=1)
def _word_substitutions() -> Dict[Tuple[str, ...], str]:
    path = Path(__file__).resolve().parents[1] / "assets" / "word_substitutions.tsv"
    with path.open(encoding="utf-8") as stream:
        return {
            tuple(normalize_raw_text(spelling).split()): normalize_raw_text(canonical)
            for canonical, variant in csv.reader(stream, delimiter="\t")
            for spelling in (canonical, variant)
        }


def normalize_word_substitutions(text: str) -> str:
    replacements = _word_substitutions()
    max_length = max(map(len, replacements))
    words, result, i = text.split(), [], 0
    while i < len(words):
        for length in range(min(max_length, len(words) - i), 0, -1):
            phrase = tuple(words[i : i + length])
            if phrase in replacements:
                result.append(replacements[phrase])
                i += length
                break
        else:
            result.append(words[i])
            i += 1
    return " ".join(result)


def compute_wer(
    preds: List[Dict[str, Any]],
    language: Optional[str] = None,
    word_substitutions: bool = True,
) -> Tuple[float, float, int, int, int, int]:
    """
    Word error rates for prediction rows with ``text`` (reference) and ``pred_text`` (hypothesis).

    Returns (wer_e2e_pct, wer_raw_pct, e2e_err, e2e_words, raw_err, raw_words).
    """
    e2e_err = e2e_w = raw_err = raw_w = 0
    for r in preds:
        ref_s, hyp_s = r["text"].strip(), r["pred_text"].strip()
        rw, hw = ref_s.split(), hyp_s.split()
        e2e_err += editdistance.eval(rw, hw)
        e2e_w += len(rw)
        nr = normalize_raw_text(ref_s, language=language)
        nh = normalize_raw_text(hyp_s, language=language)
        if word_substitutions:
            nr = normalize_word_substitutions(nr)
            nh = normalize_word_substitutions(nh)
        rr, hh = nr.split(), nh.split()
        raw_err += editdistance.eval(rr, hh)
        raw_w += len(rr)
    return (
        e2e_err / max(e2e_w, 1) * 100,
        raw_err / max(raw_w, 1) * 100,
        e2e_err,
        e2e_w,
        raw_err,
        raw_w,
    )


def save_split(ds, split: str, out_dir: str, max_dur: float, workers: int) -> List[str]:
    audio_dir = Path(out_dir) / "audio" / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    n = len(ds)
    w = max(1, min(workers, n))

    def process_one(i: int) -> Optional[str]:
        sample = ds[i]
        text = sample["text"].strip()
        arr, sr = sample["audio"]["array"], sample["audio"]["sampling_rate"]
        if len(arr) / sr > max_dur:
            return None
        rel_p = f"audio/{split}/{i:06d}.wav"
        p = audio_dir / f"{i:06d}.wav"
        if not p.exists():
            sf.write(str(p), arr, sr)
        return f"{rel_p}\t{len(arr) / sr:.3f}\t{text}"

    with ThreadPoolExecutor(max_workers=w) as ex:
        lines = list(
            tqdm(
                ex.map(process_one, range(n)),
                total=n,
                desc=f"{split} ({n})",
            )
        )
    return [ln for ln in lines if ln is not None]


def load_tonebooks(out_dir: str, max_duration: float = 30.0, workers: int = 8) -> Path:
    """
    Load ToneBooks and save E2E train / val manifests to the output directory.
    """

    from datasets import Audio, load_dataset

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading Vikhrmodels/ToneBooks...")
    ds = load_dataset("Vikhrmodels/ToneBooks")
    train, val = ds["train"], ds.get("validation") or ds.get("test")
    train = train.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    val = val.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    print(f"Splits: {list(ds.keys())}, train={len(train)}, val={len(val)}")

    for fname, rows in (
        (
            "manifest_train.tsv",
            save_split(train, "train", out_dir, max_duration, workers),
        ),
        ("manifest_val.tsv", save_split(val, "val", out_dir, max_duration, workers)),
    ):
        path = out / fname
        path.write_text(
            "path\tduration\ttranscription_e2e\n" + "\n".join(rows) + "\n",
            encoding="utf-8",
        )
        print(f"  {path} ({len(rows)} samples)")

    print(f"\nDone! Manifests at {out}")
    return out


def load_fleurs(
    out_dir: str,
    lang: str,
    dataset: str = "google/fleurs",
    train_split: str = "train",
    val_split: str = "validation",
    max_duration: Optional[float] = 30.0,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    workers: int = 8,
    test_split: Optional[str] = "test",
    max_test_samples: Optional[int] = None,
) -> Path:
    """Load official FLEURS audio and verbatim/normalized train, val, test labels."""
    import io
    import tarfile
    import urllib.request

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_dir = out / "source"
    source_dir.mkdir(exist_ok=True)
    base_url = f"https://huggingface.co/datasets/{dataset}/resolve/main/data/{lang}"
    splits = [
        (train_split, "train", max_train_samples),
        (val_split, "val", max_val_samples),
    ]
    if test_split is not None:
        splits.append((test_split, "test", max_test_samples))
    for hf_split, split, limit in splits:
        hf_split = "dev" if hf_split == "validation" else hf_split
        source_path = source_dir / f"{hf_split}.tsv"
        if not source_path.exists():
            with urllib.request.urlopen(
                f"{base_url}/{hf_split}.tsv", timeout=90
            ) as response:
                source_path.write_bytes(response.read())
        with source_path.open(encoding="utf-8", newline="") as stream:
            source_rows = list(
                csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
            )
        rows = []
        for row in source_rows:
            if len(row) < 6 or Path(row[1]).name != row[1]:
                raise ValueError(f"Invalid FLEURS row in {source_path}: {row!r}")
            if max_duration is None or int(row[5]) / SAMPLE_RATE <= max_duration:
                rows.append(row)
        if limit is not None:
            rows = rows[:limit]
        audio_dir = out / "audio" / split
        audio_dir.mkdir(parents=True, exist_ok=True)
        missing = {row[1] for row in rows if not (audio_dir / row[1]).is_file()}
        if missing:
            with urllib.request.urlopen(
                f"{base_url}/audio/{hf_split}.tar.gz", timeout=180
            ) as response:
                with tarfile.open(fileobj=response, mode="r|gz") as archive:
                    for member in archive:
                        name = Path(member.name).name
                        if not member.isfile() or name not in missing:
                            continue
                        audio_file = archive.extractfile(member)
                        if audio_file is None:
                            continue
                        audio, sample_rate = sf.read(
                            io.BytesIO(audio_file.read()), dtype="float32"
                        )
                        sf.write(audio_dir / name, audio, sample_rate)
                        missing.remove(name)
                        if not missing:
                            break
            if missing:
                raise FileNotFoundError(f"Missing FLEURS audio: {sorted(missing)}")

        def make_row(row):
            path = audio_dir / row[1]
            info = sf.info(path)
            if not info.frames:
                raise ValueError(f"Empty FLEURS audio: {path}")
            return [
                str(path.relative_to(out)),
                info.duration,
                row[2],
                normalize_raw_text(row[3]),
            ]

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            manifest_rows = list(executor.map(make_row, rows))
        manifest = out / f"manifest_{split}.tsv"
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(
                ["path", "duration", "transcription_e2e", "transcription_raw"]
            )
            writer.writerows(manifest_rows)
        print(f"{manifest}: {len(manifest_rows)} samples")
    return out


class StepProgressBar(TQDMProgressBar):

    def __init__(self, steps_per_epoch: Optional[int] = None):
        super().__init__()
        self._steps_per_epoch = steps_per_epoch

    def on_train_epoch_start(self, trainer: Trainer, *args: Any) -> None:
        super().on_train_epoch_start(trainer, *args)
        acc = trainer.accumulate_grad_batches
        if acc <= 1:
            return
        if self._steps_per_epoch:
            remaining = (
                trainer.max_steps - trainer.global_step
                if trainer.max_steps > 0
                else self._steps_per_epoch
            )
            total = min(self._steps_per_epoch, remaining)
        else:
            total = (self.train_progress_bar.total or 0) // acc
        self.train_progress_bar.reset(total=total)

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ):
        acc = trainer.accumulate_grad_batches
        if acc <= 1:
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
            return
        if (batch_idx + 1) % acc == 0:
            self.train_progress_bar.n = (batch_idx + 1) // acc
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
            self.train_progress_bar.refresh()


class EpochTimeLogger(Callback):
    def on_train_epoch_start(self, trainer: Any, pl_module: Any):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer: Any, pl_module: Any):
        duration = time.time() - self.start_time
        if trainer.is_global_zero:
            print(f"[epoch {trainer.current_epoch}] time: {duration:.2f} sec")


def _fmt_float(v: float) -> str:
    return f"{v:g}".replace("+0", "+").replace("-0", "-")


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._-") or "exp"


def build_exp_name(args) -> str:
    """
    Create a unique experiment name based on the command line arguments.
    Ignores arguments which do not affect training dynamics.
    """
    if args.exp_name:
        return _sanitize_name(args.exp_name)
    parts = [f"{args.model_name.replace('_', '')}"]
    parts += [f"lr{_fmt_float(args.lr)}", f"wd{_fmt_float(args.weight_decay)}"]
    parts.append(f"b{args.batch_size}")
    if args.accumulate_grad_batches > 1:
        parts.append(f"agb{args.accumulate_grad_batches}")
    if args.devices > 1:
        parts.append(f"{args.devices}gpu")
    if args.max_steps is not None:
        parts.append(f"{args.max_steps}steps")
        parts.append(f"vcs{args.val_check_steps}")
    else:
        parts.append(f"{args.max_epochs}ep")
        if args.val_check_interval != 1.0:
            parts.append(f"vci{_fmt_float(args.val_check_interval)}")
    if args.warmup_ratio != 0.1:
        parts.append(f"wmp{_fmt_float(args.warmup_ratio)}")
    if args.freeze_encoder_epochs == -1:
        parts.append("frenc")
    elif args.freeze_encoder_epochs > 0:
        parts.append(f"frenc{args.freeze_encoder_epochs}")
    if args.activation_checkpointing:
        parts.append("acckpt")
    if args.val_first_batches is not None:
        parts.append(f"vfb{args.val_first_batches}")
    if args.raw_text:
        parts.append("raw")
    parts.append(f"dur{_fmt_float(args.min_duration)}-{_fmt_float(args.max_duration)}s")
    if args.gradient_clip_val != 1.0:
        parts.append(f"gc{_fmt_float(args.gradient_clip_val)}")
    if args.precision != "16":
        parts.append(f"pr-{str(args.precision).replace('-', '')}")
    if args.seed != 42:
        parts.append(f"seed{args.seed}")
    if args.disable_spec_augment:
        parts.append("nospecaug")
    if not args.disable_spec_augment:
        if args.freq_masks != 2:
            parts.append(f"fm{args.freq_masks}")
        if args.freq_width != 27:
            parts.append(f"fw{args.freq_width}")
        if args.time_masks != 2:
            parts.append(f"tm{args.time_masks}")
        if args.time_width != 20:
            parts.append(f"tw{args.time_width}")
    return _sanitize_name("_".join(parts))


def prepare_experiment_dirs(output_dir: str, exp_name: str) -> Tuple[str, str]:
    model_dir = os.path.join(output_dir, "models", exp_name)
    tb_dir = os.path.join(output_dir, "tb_logs")
    if os.path.isdir(model_dir) and os.listdir(model_dir):
        warnings.warn(
            f"Checkpoint dir is not empty: {model_dir}. Checkpoints may be overwritten.",
            stacklevel=1,
        )
    os.makedirs(model_dir, exist_ok=True)
    return model_dir, tb_dir
