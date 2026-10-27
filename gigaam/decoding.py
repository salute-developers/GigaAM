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
        ), f"Num classes {C} != len(vocab)+1 {len(self.tokenizer) + 1}"
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

        The whole batch is decoded at once: predictor outputs and LSTM states
        stay batched tensors and are updated with a mask of emitting samples,
        so every step costs one predictor and one joint call for the batch.
        """
        x = encoded.transpose(1, 2)  # [B, T, D]
        B = x.shape[0]
        enc_len = enc_len.to(x.device)

        # Predictor output and state after the last emitted label (none yet).
        g, state = head.decoder.predict(None, None, batch_size=B)

        step_tokens: List[Tensor] = []
        step_masks: List[Tensor] = []
        step_frames: List[int] = []

        for t in range(int(enc_len.max()) if B else 0):
            active = enc_len > t
            for _ in range(self.max_symbols):
                k = head.joint.joint(x[:, t : t + 1, :], g)[:, 0, 0, :].argmax(dim=-1)
                emit = active & k.ne(self.blank_id)
                if not emit.any():
                    break

                step_tokens.append(k)
                step_masks.append(emit)
                step_frames.append(t)

                g_new, state_new = head.decoder.predict(
                    k.unsqueeze(1), state, batch_size=B
                )
                g = torch.where(emit[:, None, None], g_new, g)
                state = tuple(
                    torch.where(emit[None, :, None], new, old)
                    for new, old in zip(state_new, state)
                )
                active = emit

        if not step_tokens:
            return [(self.tokenizer.decode([]), [], []) for _ in range(B)]

        tokens = torch.stack(step_tokens, dim=1).cpu()  # [B, steps]
        masks = torch.stack(step_masks, dim=1).cpu()  # [B, steps]
        frames = torch.tensor(step_frames)  # [steps]

        out = []
        for i in range(B):
            hyp = tokens[i][masks[i]].tolist()
            out.append((self.tokenizer.decode(hyp), hyp, frames[masks[i]].tolist()))
        return out
