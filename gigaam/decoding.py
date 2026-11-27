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
        states=None,
    ) -> List[Tuple[str, List[int], List[int]]]:
        """
        CTC greedy decode: returns (text, token_ids, token_frames) per sample.
        Token frames are time indices (0..T-1) where a token is emitted.
        """
        log_probs = head(encoder_output=encoded)
        if log_probs.ndim == 2:
            return self.decode_labels(log_probs, lengths, states)
        C = log_probs.shape[-1]
        assert (
            C == len(self.tokenizer) + 1
        ), f"Num classes {C} != len(vocab)+1 {len(self.tokenizer) + 1}"
        labels = log_probs.argmax(dim=-1)
        return self.decode_labels(labels, lengths, states)

    def decode_labels(self, labels: Tensor, lengths: Tensor, states=None):
        labels = labels.cpu()
        B, T = labels.shape
        lengths = lengths.cpu().clamp(min=0, max=T)

        skip_mask = labels != self.blank_id
        if states is not None and T:
            previous = labels.new_tensor(
                [
                    self.blank_id if state.previous is None else state.previous
                    for state in states
                ]
            )
            skip_mask[:, 0] &= labels[:, 0] != previous
        skip_mask[:, 1:] &= labels[:, 1:] != labels[:, :-1]

        time = torch.arange(T, device=labels.device)[None, :]
        skip_mask &= time < lengths[:, None]

        idx = skip_mask.nonzero(as_tuple=False)
        batch_idx = idx[:, 0]
        token_frames_flat = idx[:, 1]
        token_ids_flat = labels[skip_mask]

        counts = torch.bincount(batch_idx, minlength=B).tolist()
        ids_splits = token_ids_flat.split(counts)
        fr_splits = token_frames_flat.split(counts)

        results = []
        for index, (ids, frames) in enumerate(zip(ids_splits, fr_splits)):
            tokens, positions = ids.tolist(), frames.tolist()
            if states is not None:
                state = states[index]
                state.tokens.extend(tokens)
                state.frames.extend(frame + state.frame_offset for frame in positions)
                count = int(lengths[index])
                if count:
                    state.previous = int(labels[index, count - 1])
                state.frame_offset += count
            results.append((self.tokenizer.decode(tokens), tokens, positions))
        return results


class RNNTGreedyDecoding:
    """
    Class for performing greedy decoding of RNN-T outputs.
    """

    def __init__(
        self,
        vocabulary: Optional[List[str]] = None,
        model_path: Optional[str] = None,
        max_symbols_per_step: int = 10,
    ):
        self.tokenizer = Tokenizer(vocabulary or [], model_path)
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
        states=None,
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
        last_label = (
            [state.previous for state in states] if states is not None else [None] * B
        )
        dec_state = (
            [state.hidden for state in states] if states is not None else [None] * B
        )

        prediction = (
            [state.prediction for state in states] if states is not None else [None] * B
        )
        prediction_hidden = (
            [state.prediction_hidden for state in states]
            if states is not None
            else [None] * B
        )

        def predict_batch(batch_idx: List[int], fresh: bool):
            if fresh:
                g, hidden = head.decoder.predict(None, None, batch_size=len(batch_idx))
            else:
                labels = torch.tensor(
                    [[last_label[i]] for i in batch_idx], device=device
                )
                state = self._cat_states([dec_state[i] for i in batch_idx])
                g, hidden = head.decoder.predict(
                    labels, state, batch_size=len(batch_idx)
                )
            for i, value, hidden_part in zip(
                batch_idx, g.split(1), self._split_state(hidden)
            ):
                prediction[i] = value
                prediction_hidden[i] = hidden_part

        # Blank frames leave the prediction unchanged, so score a window at once.
        window = 8 if getattr(head.joint, "supports_frame_windows", False) else 1
        offsets = torch.arange(window, device=device)
        enc_len = enc_len.cpu()
        limits = enc_len.clamp(max=T).tolist()
        positions = [0] * B
        repeats = [0] * B
        active = [i for i, limit in enumerate(limits) if limit > 0]
        while active and self.max_symbols > 0:
            missing = [i for i in active if prediction[i] is None]
            fresh = [i for i in missing if last_label[i] is None]
            stateful = [i for i in missing if last_label[i] is not None]
            if fresh:
                predict_batch(fresh, fresh=True)
            if stateful:
                predict_batch(stateful, fresh=False)

            rows = torch.tensor(active, device=device)
            start = torch.tensor([positions[i] for i in active], device=device)
            frames = (start[:, None] + offsets).clamp_max(T - 1)
            features = x[rows[:, None], frames]
            predicted = torch.cat([prediction[i] for i in active])
            labels = head.joint.joint(features, predicted)[:, :, 0, :].argmax(-1)
            for i, labels_row in zip(active, labels.tolist()):
                count = min(window, limits[i] - positions[i])
                for delta, token in enumerate(labels_row[:count]):
                    if token != self.blank_id:
                        break
                else:
                    positions[i] += count
                    repeats[i] = 0
                    continue
                positions[i] += delta
                repeats[i] = 1 if delta else repeats[i] + 1
                hyps[i].append(token)
                token_frames[i].append(positions[i])
                last_label[i] = token
                dec_state[i] = prediction_hidden[i]
                prediction[i] = prediction_hidden[i] = None
                if repeats[i] >= self.max_symbols:
                    positions[i] += 1
                    repeats[i] = 0
            active = [i for i in active if positions[i] < limits[i]]

        if states is not None:
            for index, state in enumerate(states):
                state.tokens.extend(hyps[index])
                state.frames.extend(
                    frame + state.frame_offset for frame in token_frames[index]
                )
                if last_label[index] is not None:
                    state.previous = last_label[index]
                state.hidden = dec_state[index]
                state.prediction = prediction[index]
                state.prediction_hidden = prediction_hidden[index]
                state.frame_offset += int(enc_len[index])
        return [(self.tokenizer.decode(h), h, tf) for h, tf in zip(hyps, token_frames)]
