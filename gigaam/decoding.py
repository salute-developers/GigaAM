from typing import List, Optional, Tuple

import torch
from sentencepiece import SentencePieceProcessor
from torch import Tensor

from .decoder import CTCHead, RNNTHead


class Tokenizer:
    """
    Tokenizer for converting between text and token IDs.
    The tokenizer can operate either character-wise or using a pre-trained SentencePiece model.
    """

    def __init__(self, vocab: List[str], model_path: Optional[str] = None):
        self.charwise = model_path is None
        if self.charwise:
            self.vocab = vocab
        else:
            self.model = SentencePieceProcessor()
            self.model.load(model_path)

    def decode(self, tokens: List[int]) -> str:
        """
        Convert a list of token IDs back to a string.
        """
        if self.charwise:
            return "".join(self.vocab[tok] for tok in tokens)
        return self.model.decode(tokens)

    def __len__(self):
        """
        Get the total number of tokens in the vocabulary.
        """
        return len(self.vocab) if self.charwise else len(self.model)

    def id_to_str(self, token_id: int) -> str:
        """
        Convert a single token ID to its string representation.
        """
        if self.charwise:
            return self.vocab[token_id]
        return self.model.IdToPiece(token_id)


class CTCGreedyDecoding:
    """
    Class for performing greedy decoding of CTC outputs.
    """

    def __init__(self, vocabulary: List[str], model_path: Optional[str] = None):
        self.tokenizer = Tokenizer(vocabulary, model_path)
        self.blank_id = len(self.tokenizer)

    @torch.inference_mode()
    def decode(
        self,
        head: "CTCHead",
        encoded: Tensor,
        lengths: Tensor,
    ) -> List[Tuple[str, List[int], List[int]]]:
        """
        CTC greedy decode: returns (text, token_ids, token_frames) per sample.
        Token frames are time indices (0..T-1) where a token is emitted.
        """
        log_probs = head(encoder_output=encoded)
        C = log_probs.shape[-1]
        assert (
            C == len(self.tokenizer) + 1
        ), f"Num classes {C} != len(vocab)+1 {len(self.tokenizer)+1}"
        labels = log_probs.argmax(dim=-1)

        B, T = labels.shape
        device = labels.device
        lengths = lengths.to(device=device).clamp(min=0, max=T)

        skip_mask = labels != self.blank_id
        skip_mask[:, 1:] &= labels[:, 1:] != labels[:, :-1]

        time = torch.arange(T, device=device)[None, :]
        skip_mask &= time < lengths[:, None]

        idx = skip_mask.nonzero(as_tuple=False)
        batch_idx = idx[:, 0]
        token_frames_flat = idx[:, 1]
        token_ids_flat = labels[skip_mask]

        counts = torch.bincount(batch_idx, minlength=B).cpu().tolist()
        ids_splits = token_ids_flat.cpu().split(counts)
        fr_splits = token_frames_flat.cpu().split(counts)

        return [
            (self.tokenizer.decode(ids.tolist()), ids.tolist(), fr.tolist())
            for ids, fr in zip(ids_splits, fr_splits)
        ]


class RNNTGreedyDecoding:
    """
    Class for performing greedy decoding of RNN-T outputs.
    """

    def __init__(
        self,
        vocabulary: List[str],
        model_path: Optional[str] = None,
        max_symbols_per_step: int = 10,
    ):
        self.tokenizer = Tokenizer(vocabulary, model_path)
        self.blank_id = len(self.tokenizer)
        self.max_symbols = max_symbols_per_step

    @staticmethod
    def _cat_states(states):
        """Pack per-sample LSTM states into batched (h, c)."""
        hs = [s[0] for s in states]
        cs = [s[1] for s in states]
        return torch.cat(hs, dim=1), torch.cat(cs, dim=1)

    @staticmethod
    def _split_state(state):
        """Unpack batched (h, c) into per-sample states."""
        h, c = state
        b = h.shape[1]
        return [(h[:, i : i + 1], c[:, i : i + 1]) for i in range(b)]

    @torch.inference_mode()
    def decode(
        self,
        head: "RNNTHead",
        encoded: Tensor,
        enc_len: Tensor,
    ) -> List[Tuple[str, List[int], List[int]]]:
        """
        RNN-T greedy decode: returns (text, token_ids, token_frames) per sample.
        Token frames are encoder time indices where tokens are emitted.
        """
        x = encoded.transpose(1, 2)  # [B, T, D]
        B, T, _ = x.shape
        device = x.device

        hyps: List[List[int]] = [[] for _ in range(B)]
        token_frames: List[List[int]] = [[] for _ in range(B)]
        last_label: List[Optional[Tensor]] = [None] * B
        dec_state: List[Optional[Tuple[Tensor, Tensor]]] = [None] * B

        def emit_batch(batch_idx: List[int], t: int, fresh: bool) -> List[int]:
            """One batched predictor+joint step; returns samples that emitted non-blank."""
            idx = torch.tensor(batch_idx, device=device, dtype=torch.long)
            f = x[idx, t : t + 1, :]  # [b, 1, D]

            if fresh:
                g, hidden = head.decoder.predict(None, None, batch_size=len(batch_idx))
            else:
                labels = torch.cat([last_label[i] for i in batch_idx], dim=0)  # [b, 1]
                state = self._cat_states([dec_state[i] for i in batch_idx])
                g, hidden = head.decoder.predict(
                    labels, state, batch_size=len(batch_idx)
                )

            k = head.joint.joint(f, g)[:, 0, 0, :].argmax(dim=-1)  # [b]
            emit = k.ne(self.blank_id)

            if not emit.any():
                return []

            hidden_parts = self._split_state(hidden)
            out = []

            for p in emit.nonzero(as_tuple=False).squeeze(1).tolist():
                bi = batch_idx[p]
                tok = int(k[p])

                hyps[bi].append(tok)
                token_frames[bi].append(t)
                last_label[bi] = k[p : p + 1].view(1, 1)
                dec_state[bi] = hidden_parts[p]
                out.append(bi)

            return out

        enc_len = enc_len.cpu()
        for t in range(T):
            active = (t < enc_len).nonzero(as_tuple=False).squeeze(1).tolist()
            if not active:
                break

            for _ in range(self.max_symbols):
                if not active:
                    break

                fresh = [i for i in active if dec_state[i] is None]
                stateful = [i for i in active if dec_state[i] is not None]

                next_active = []
                if fresh:
                    next_active.extend(emit_batch(fresh, t, fresh=True))
                if stateful:
                    next_active.extend(emit_batch(stateful, t, fresh=False))

                if not next_active:
                    break

                active = next_active

        return [(self.tokenizer.decode(h), h, tf) for h, tf in zip(hyps, token_frames)]
