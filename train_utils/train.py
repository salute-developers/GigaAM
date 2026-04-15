"""Fine-tune a GigaAM pretrained model."""

import argparse
import warnings

import pytorch_lightning as pl
import torch
from module import GigaAMFineTuner
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from utils import (
    EpochTimeLogger,
    StepProgressBar,
    build_exp_name,
    prepare_experiment_dirs,
)

import gigaam
from gigaam.utils import AudioDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--train_manifest", required=True)
    p.add_argument("--val_manifest", required=True)
    p.add_argument("--output_dir", default="./checkpoints")
    p.add_argument("--exp_name", default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--rnnt_subbatch_size", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_duration", type=float, default=20.0)
    p.add_argument("--min_duration", type=float, default=0.1)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--gradient_clip_val", type=float, default=1.0)
    p.add_argument("--precision", default="32")
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--activation_checkpointing", action="store_true")
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--raw_text", action="store_true")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--val_check_interval", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--val_check_steps", type=int, default=None)
    p.add_argument("--val_first_batches", type=int, default=None)
    p.add_argument("--log_every_n_steps", type=int, default=25)
    p.add_argument("--disable_tqdm", action="store_true", default=False)
    p.add_argument("--skip_initial_validation", action="store_true", default=False)
    p.add_argument("--save_top_k", type=int, default=2)
    p.add_argument("--disable_spec_augment", action="store_true")
    p.add_argument("--freq_masks", type=int, default=2)
    p.add_argument("--freq_width", type=int, default=27)
    p.add_argument("--time_masks", type=int, default=2)
    p.add_argument("--time_width", type=int, default=20)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = p.parse_args()
    assert (args.max_steps is not None) ^ (
        args.max_epochs is not None
    ), "Either --max_steps or --max_epochs must be provided, but not both"
    step_mode = args.max_steps is not None
    if step_mode:
        assert args.val_check_steps, "--max_steps requires --val_check_steps"
    assert not (
        args.max_epochs is not None and args.val_check_steps is not None
    ), "Use --val_check_interval instead of --val_check_steps with epoch mode"
    return args


def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    step_mode = args.max_steps is not None
    exp_name = build_exp_name(args)
    print(f"Experiment: {exp_name}")
    model_dir, tb_dir = prepare_experiment_dirs(args.output_dir, exp_name)

    print(f"Loading pretrained {args.model_name} ...")
    model = gigaam.load_model(args.model_name, fp16_encoder=False, device="cpu")
    assert isinstance(
        model, gigaam.GigaAMASR
    ), "Fine-tuning expects an ASR model (GigaAMASR)"

    if args.activation_checkpointing:
        model.encoder.activation_checkpointing = True
        print("Encoder: activation checkpointing on (per Conformer layer)")
    tokenizer = model.decoding.tokenizer
    blank_id = model.decoding.blank_id
    orig_model_name = model.cfg.model_name
    is_e2e = "e2e" in orig_model_name
    print(
        f"Mode: {'rnnt' if 'rnnt' in orig_model_name else 'ctc'} | vocab={len(tokenizer)}, blank={blank_id}"
    )

    if args.raw_text and is_e2e:
        raise ValueError("--raw_text is only for non-e2e models (charwise vocab)")
    if not is_e2e and not args.raw_text:
        warnings.warn(
            "Non-e2e model without --raw_text: text won't be normalized. "
            "Consider --raw_text to strip punctuation to vocab chars.",
            stacklevel=1,
        )

    ds_kw = dict(
        tokenizer=tokenizer,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        raw_text=args.raw_text,
        return_tokens=True,
    )
    train_ds = AudioDataset(args.train_manifest, **ds_kw)
    val_ds = AudioDataset(args.val_manifest, **ds_kw)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    dl_kw = dict(num_workers=args.num_workers, pin_memory=True)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=train_ds.collate_fn,
        **dl_kw,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=val_ds.collate_fn,
        **dl_kw,
    )

    lit = GigaAMFineTuner(
        model=model,
        blank_id=blank_id,
        lr=args.lr,
        freeze_encoder=args.freeze_encoder,
        rnnt_subbatch_size=args.rnnt_subbatch_size,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        log_every_n_steps=args.log_every_n_steps,
        spec_augment=not args.disable_spec_augment,
        freq_masks=args.freq_masks,
        freq_width=args.freq_width,
        time_masks=args.time_masks,
        time_width=args.time_width,
        cli_args=vars(args),
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=model_dir,
        filename=f"gigaam-{args.model_name}-" + "{epoch:02d}-{step:06d}-{val_wer:.4f}",
        monitor="val_wer",
        mode="min",
        save_top_k=max(1, args.save_top_k),
    )

    trainer_kw = dict(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy="ddp" if args.devices > 1 else "auto",
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=(
            [ckpt_cb, EpochTimeLogger()]
            + (
                [StepProgressBar(args.val_check_steps if step_mode else None)]
                if not args.disable_tqdm
                else []
            )
        ),
        logger=TensorBoardLogger(save_dir=tb_dir, name=exp_name),
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=0,
        default_root_dir=args.output_dir,
        limit_val_batches=(
            args.val_first_batches if args.val_first_batches is not None else 1.0
        ),
        enable_progress_bar=not args.disable_tqdm,
    )
    if step_mode:
        trainer_kw.update(
            max_steps=args.max_steps,
            max_epochs=-1,
            limit_train_batches=args.val_check_steps * args.accumulate_grad_batches,
        )
    else:
        trainer_kw.update(
            max_epochs=args.max_epochs, val_check_interval=args.val_check_interval
        )

    trainer = pl.Trainer(**trainer_kw)
    if not args.skip_initial_validation:
        print("Running initial validation...")
        trainer.validate(lit, val_dl)
    trainer.fit(lit, train_dl, val_dl, ckpt_path=args.resume_from_checkpoint)
    print(f"Best: {ckpt_cb.best_model_path}")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
