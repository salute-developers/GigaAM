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
    ) -> List[Tuple[List[int], List[int]]]:
        """
        CTC greedy decode: returns (token_ids, token_frames) per sample.
        Token frames are time indices (0..T-1) where a token is emitted.
        """
        log_probs = head(encoder_output=encoded)
        assert (
            log_probs.ndim == 3
        ), f"Expected log_probs [B,T,C], got {tuple(log_probs.shape)}"
        B, T, C = log_probs.shape
        assert (
            C == len(self.tokenizer) + 1
        ), f"Num classes {C} != len(vocab)+1 {len(self.tokenizer)+1}"

        labels = log_probs.argmax(dim=-1)
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

        return [(ids.tolist(), fr.tolist()) for ids, fr in zip(ids_splits, fr_splits)]


class RNNTGreedyDecoding:
    def __init__(
        self,
        vocabulary: List[str],
        model_path: Optional[str] = None,
        max_symbols_per_step: int = 10,
    ):
        self.tokenizer = Tokenizer(vocabulary, model_path)
        self.blank_id = len(self.tokenizer)
        self.max_symbols = max_symbols_per_step

    def _greedy_decode(
        self,
        head: "RNNTHead",
        x: Tensor,
        seqlen: Tensor,
    ) -> Tuple[List[int], List[int]]:
        """
        Greedy decode a single sequence.
        Returns (token_ids, token_frames).
        Token frames are encoder time indices t where a token is emitted.
        """
        T = int(seqlen.item()) if torch.is_tensor(seqlen) else int(seqlen)

        hyp: List[int] = []
        token_frames: List[int] = []
        dec_state: Optional[Tensor] = None

        last_label: Optional[Tensor] = None

        last_label_buf = torch.empty((1, 1), device=x.device, dtype=torch.long)

        for t in range(T):
            f = x[t, :, :].unsqueeze(1)
            new_symbols = 0

            while new_symbols < self.max_symbols:
                g, hidden = head.decoder.predict(last_label, dec_state)
                k = int(head.joint.joint(f, g)[0, 0, 0, :].argmax(0).item())

                if k == self.blank_id:
                    break

                hyp.append(k)
                token_frames.append(t)

                dec_state = hidden
                last_label_buf.fill_(k)
                last_label = last_label_buf
                new_symbols += 1

        return hyp, token_frames

    @torch.inference_mode()
    def decode(
        self,
        head: "RNNTHead",
        encoded: Tensor,
        enc_len: Tensor,
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Decode RNN-T outputs for a batch.
        Returns (token_ids, token_frames) per sample.
        """
        B = encoded.shape[0]
        encoded = encoded.transpose(1, 2)

        results: List[Tuple[List[int], List[int]]] = []
        for i in range(B):
            inseq = encoded[i, :, :].unsqueeze(1)
            results.append(self._greedy_decode(head, inseq, enc_len[i]))
        return results
