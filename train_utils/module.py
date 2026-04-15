import math
from typing import Any, Dict, List, Optional, Tuple

import editdistance
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torchaudio.functional import rnnt_loss as _ta_rnnt_loss
from torchaudio.transforms import FrequencyMasking, TimeMasking

from gigaam.decoder import CTCHead, RNNTHead
from gigaam.model import GigaAMASR


class GigaAMFineTuner(pl.LightningModule):

    def __init__(
        self,
        model: GigaAMASR,
        blank_id: int,
        lr: float = 1e-4,
        freeze_encoder: bool = False,
        rnnt_subbatch_size: int = 0,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        log_every_n_steps: int = 10,
        spec_augment: bool = False,
        freq_masks: int = 2,
        freq_width: int = 27,
        time_masks: int = 2,
        time_width: int = 20,
        cli_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "cli_args"])
        if cli_args is not None:
            self.save_hyperparameters(cli_args)
        self.preprocessor = model.preprocessor
        self.encoder = model.encoder
        self.head = model.head
        self._decoding = model.decoding
        self._tokenizer = model.decoding.tokenizer
        self._blank_id = blank_id
        self._rnnt_subbatch_size = rnnt_subbatch_size
        self._freeze_encoder = freeze_encoder

        self._spec_augment = spec_augment
        if spec_augment:
            self._freq_aug = nn.ModuleList(
                [FrequencyMasking(freq_width) for _ in range(freq_masks)]
            )
            self._time_aug = nn.ModuleList(
                [TimeMasking(time_width) for _ in range(time_masks)]
            )

        self._rnnt_vocab_size = 0
        if isinstance(self.head, CTCHead):
            self.mode = "ctc"
            self._ctc = nn.CTCLoss(blank=blank_id, reduction="none", zero_infinity=True)
        elif isinstance(self.head, RNNTHead):
            self.mode = "rnnt"
            self._rnnt_vocab_size = int(model.cfg.head.joint.num_classes)
            assert (
                blank_id + 1 == self._rnnt_vocab_size
            ), f"blank_id={blank_id} != joint V={self._rnnt_vocab_size}"
        else:
            raise ValueError(f"Unsupported head: {type(self.head)}")

        self._lr, self._wd = lr, weight_decay
        self._warmup_ratio = warmup_ratio
        self._log_every, self._last_log_step = log_every_n_steps, -1

        for p in self.preprocessor.parameters():
            p.requires_grad = False
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self._val_errors = self._val_words = 0

    def train(self, mode: bool = True) -> "GigaAMFineTuner":
        super().train(mode)
        # We keep it verbose for the freezed preprocessor
        self.preprocessor.eval()
        if self._freeze_encoder:
            self.encoder.eval()
        return self

    def on_train_epoch_start(self):
        self.train()

    def _ctc_loss(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lens: Tensor,
        target_lens: Tensor,
    ) -> Tensor:
        return self._ctc(
            log_probs.transpose(0, 1),
            targets.long(),
            input_lens.long(),
            target_lens.long(),
        ).mean()

    def _rnnt_loss(
        self, logits: Tensor, targets: Tensor, logit_lens: Tensor, target_lens: Tensor
    ) -> Tensor:
        return _ta_rnnt_loss(
            logits=logits.float(),
            targets=targets.int(),
            logit_lengths=logit_lens.int(),
            target_lengths=target_lens.int(),
            blank=self._blank_id,
            reduction="mean",
            fused_log_softmax=True,
        )

    def _encode(self, wavs: Tensor, wav_lens: Tensor) -> Tuple[Tensor, Tensor]:
        amp_dev = next(self.parameters()).device.type
        with torch.amp.autocast(amp_dev, enabled=False):
            features, feat_lens = self.preprocessor(wavs.float(), wav_lens)
        if self.training and self._spec_augment:
            for aug in self._freq_aug:
                features = aug(features)
            for aug in self._time_aug:
                features = aug(features)
        return self.encoder(features, feat_lens)

    def _rnnt_joint(self, encoded: Tensor, tokens: Tensor) -> Tensor:
        head = self.head
        assert isinstance(head, RNNTHead)
        enc_t = encoded.float().transpose(1, 2)
        dec, jnt = head.decoder, head.joint
        bos = torch.zeros(
            enc_t.size(0), 1, dec.pred_hidden, device=enc_t.device, dtype=enc_t.dtype
        )
        pred_out, _ = dec.lstm(
            torch.cat([bos, dec.embed(tokens)], dim=1).transpose(0, 1)
        )
        return jnt.joint_net(
            jnt.enc(enc_t).unsqueeze(2)
            + jnt.pred(pred_out.transpose(0, 1)).unsqueeze(1)
        )

    def _rnnt_forward(
        self, encoded: Tensor, enc_lens: Tensor, tokens: Tensor, tok_lens: Tensor
    ) -> Tensor:
        B = encoded.size(0)
        subb = self._rnnt_subbatch_size or B

        # Ensure int32-safe indexing in _ta_rnnt_loss
        max_t = int(enc_lens.max().item())
        max_u1 = int(tok_lens.max().item()) + 1
        vocab_size = self._rnnt_vocab_size
        while subb > 1 and subb * max_t * max_u1 * vocab_size >= (1 << 31):
            subb = max(1, subb // 2)

        losses: List[torch.Tensor] = []
        weights: List[int] = []
        for i in range(0, B, subb):
            s = slice(i, min(i + subb, B))
            enc_i, enc_lens_i = encoded[s], enc_lens[s]
            tok_i, tok_lens_i = tokens[s], tok_lens[s]
            enc_i = enc_i[:, :, : int(enc_lens_i.max().item())].contiguous()
            tok_i = tok_i[:, : int(tok_lens_i.max().item())].contiguous()
            logits_i = self._rnnt_joint(enc_i, tok_i)  # [b, T, U+1, V]
            T, U1 = logits_i.shape[1:3]
            enc_lens_i = enc_lens_i.clamp(min=1, max=T)
            tok_lens_i = tok_lens_i.clamp(min=1, max=U1 - 1)
            loss_i = self._rnnt_loss(logits_i, tok_i, enc_lens_i, tok_lens_i)
            losses.append(loss_i * (enc_i.size(0)))
            weights.append(enc_i.size(0))
            del logits_i

        return torch.stack(losses).sum() / sum(weights)

    def _batch_wer(
        self, hyps: List[str], tokens: Tensor, tok_lens: Tensor
    ) -> Tuple[int, int]:
        errors = words = 0
        for i, hyp in enumerate(hyps):
            ref = self._tokenizer.decode(tokens[i, : tok_lens[i]].tolist()).split()
            hyp_w = hyp.split()
            errors += editdistance.eval(ref, hyp_w)
            words += max(len(ref), 1)
        return errors, words

    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        wavs, wav_lens, tokens, tok_lens = batch
        encoded, enc_lens = self._encode(wavs, wav_lens)

        if self.mode == "ctc":
            log_probs = self.head(encoded)
            loss = self._ctc_loss(log_probs, tokens, enc_lens, tok_lens)
        else:
            loss = self._rnnt_forward(encoded, enc_lens, tokens, tok_lens)
        self.log("train/loss", loss, prog_bar=True)

        if (
            self.global_step % self._log_every == 0
            and self.global_step != self._last_log_step
        ):
            self._last_log_step = self.global_step
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("train/lr", lr)
            with torch.no_grad():
                res = self._decoding.decode(
                    self.head, encoded.detach().float(), enc_lens
                )
                errs, wds = self._batch_wer([h[0] for h in res], tokens, tok_lens)
                train_wer = errs / max(wds, 1)
                self.log("train/wer", train_wer)
        return loss

    def validation_step(self, batch: Tuple[Tensor, ...], batch_idx: int):
        wavs, wav_lens, tokens, tok_lens = batch
        encoded, enc_lens = self._encode(wavs, wav_lens)

        if self.mode == "ctc":
            log_probs = self.head(encoded)
            loss = self._ctc_loss(log_probs, tokens, enc_lens, tok_lens)
        else:
            loss = self._rnnt_forward(encoded, enc_lens, tokens, tok_lens)

        res = self._decoding.decode(self.head, encoded, enc_lens)
        errs, wds = self._batch_wer([h[0] for h in res], tokens, tok_lens)
        self._val_errors += errs
        self._val_words += wds
        self.log("val/loss", loss, sync_dist=True)

    def on_validation_epoch_end(self):
        errs_t = torch.tensor(self._val_errors, device=self.device, dtype=torch.float64)
        wds_t = torch.tensor(self._val_words, device=self.device, dtype=torch.float64)
        if self.trainer.world_size > 1 and dist.is_initialized():
            dist.all_reduce(errs_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(wds_t, op=dist.ReduceOp.SUM)
        self._val_errors = self._val_words = 0
        if wds_t.item() <= 0:
            return
        wer = (errs_t / wds_t).item()
        # All ranks must log val_wer so ModelCheckpoint sees monitor on every process.
        self.log("val/wer", wer, sync_dist=False)
        self.log("val_wer", wer, logger=False, sync_dist=False)
        if self.trainer.is_global_zero:
            print(
                f"  [val] step={self.global_step} epoch={self.current_epoch}  "
                f"val/loss={self.trainer.callback_metrics.get('val/loss', 0):.6f}  "
                f"val/wer={wer:.4f}"
            )

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, Any]]]:
        opt = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self._lr,
            weight_decay=self._wd,
        )
        total = self.trainer.estimated_stepping_batches
        warmup = max(1, int(self._warmup_ratio * total))
        decay = max(1, total - warmup)
        print(f"  LR: {warmup} warmup + {decay} cosine = {total} steps")

        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            return max(0.0, 0.5 * (1 + math.cos(math.pi * (step - warmup) / decay)))

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return [opt], [{"scheduler": sch, "interval": "step"}]
