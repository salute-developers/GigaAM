"""CTC beam search decoding with optional KenLM shallow fusion (pyctcdecode).

``CTCBeamLMDecoding`` is a drop-in replacement for ``gigaam.decoding.CTCGreedyDecoding``:
same duck-typed surface (``tokenizer``, ``blank_id``, ``decode(head, encoded, lengths)``),
so it can be swapped in after ``gigaam.load_model``::

    model = gigaam.load_model(...)
    model.decoding = CTCBeamLMDecoding(model.decoding.tokenizer.vocab, lm_path="lm.bin")

Attach it on inference paths only. ``model.decoding`` is also used by the training and
validation WER logging in ``module.py``, which must stay greedy.
"""

import multiprocessing as mp
import warnings
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from gigaam.decoding import Tokenizer

DEFAULT_ALPHA = 0.5
DEFAULT_BETA = 1.5
DEFAULT_BEAM_SIZE = 100
DEFAULT_BEAM_PRUNE_LOGP = -10.0
DEFAULT_TOKEN_MIN_LOGP = -5.0

# pyctcdecode reads unigrams out of an .arpa itself; from a compiled binary it cannot.
_ARPA_SUFFIXES = (".arpa", ".arpa.gz")


def _read_unigrams(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [w for w in (line.strip() for line in f) if w]


class CTCBeamLMDecoding:
    """Beam search CTC decoding, optionally fused with a word-level KenLM."""

    def __init__(
        self,
        vocabulary: List[str],
        model_path: Optional[str] = None,
        lm_path: Optional[str] = None,
        unigrams_path: Optional[str] = None,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        beam_size: int = DEFAULT_BEAM_SIZE,
        beam_prune_logp: float = DEFAULT_BEAM_PRUNE_LOGP,
        token_min_logp: float = DEFAULT_TOKEN_MIN_LOGP,
        num_workers: int = 0,
    ):
        if model_path is not None:
            raise ValueError(
                "CTCBeamLMDecoding supports charwise vocabularies only; "
                "a SentencePiece model_path would need '▁' handling."
            )
        try:
            from pyctcdecode import build_ctcdecoder
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "Beam search needs pyctcdecode: pip install 'gigaam[lm]' "
                "(or: pip install pyctcdecode kenlm)"
            ) from exc

        self.tokenizer = Tokenizer(vocabulary, None)
        self.blank_id = len(self.tokenizer)
        self._char_to_id = {c: i for i, c in enumerate(vocabulary)}
        self._space_id = self._char_to_id.get(" ")

        unigrams = _read_unigrams(unigrams_path) if unigrams_path else None
        if lm_path and unigrams is None and not lm_path.endswith(_ARPA_SUFFIXES):
            warnings.warn(
                f"'{lm_path}' looks like a compiled KenLM binary and no unigrams_path "
                "was given. pyctcdecode cannot recover the unigram set from a binary, "
                "so partial-word scoring silently degrades. Pass unigrams_path "
                "(lm.vocab from build_kenlm.py) or point lm_path at the .arpa.",
                stacklevel=2,
            )

        # Blank must land on index len(vocabulary) to match CTCHead's class order.
        labels = list(vocabulary) + [""]
        self._decoder = build_ctcdecoder(
            labels,
            kenlm_model_path=lm_path,
            unigrams=unigrams,
            alpha=alpha,
            beta=beta,
        )
        self.beam_size = beam_size
        self.beam_prune_logp = beam_prune_logp
        self.token_min_logp = token_min_logp

        # pyctcdecode keeps LMs in a class-level container and strips them in
        # __getstate__ (kenlm.Model is unpicklable), so workers only see the LM if
        # they inherit it. Under 'spawn' the LM is silently lost.
        self._pool = (
            mp.get_context("fork").Pool(num_workers) if num_workers > 0 else None
        )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> "CTCBeamLMDecoding":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _to_contract(
        self, text: str, text_frames: List[Tuple[str, Tuple[int, int]]]
    ) -> Tuple[str, List[int], List[int]]:
        """Rebuild greedy's (text, token_ids, token_frames) from word-level spans.

        frames_to_words derives a word's span from its first char's frame and its last
        char's frame + 1, so anchoring those two reproduces pyctcdecode's span exactly.
        Interior chars are interpolated; a one-char word can only ever span one frame.
        """
        token_ids: List[int] = []
        token_frames: List[int] = []
        for word_idx, (word, (start, end)) in enumerate(text_frames):
            if not word:
                continue
            if word_idx and self._space_id is not None:
                token_ids.append(self._space_id)
                token_frames.append(start)
            last = max(start, end - 1)
            if len(word) == 1:
                frames = [start]
            else:
                step = (last - start) / (len(word) - 1)
                frames = [int(round(start + step * k)) for k in range(len(word))]
            for char, frame in zip(word, frames):
                char_id = self._char_to_id.get(char)
                if char_id is None:
                    continue
                token_ids.append(char_id)
                token_frames.append(frame)
        return text, token_ids, token_frames

    @torch.inference_mode()
    def decode(
        self,
        head: Any,
        encoded: Tensor,
        lengths: Tensor,
    ) -> List[Tuple[str, List[int], List[int]]]:
        log_probs = head(encoder_output=encoded)
        num_classes = log_probs.shape[-1]
        assert (
            num_classes == len(self.tokenizer) + 1
        ), f"Num classes {num_classes} != len(vocab)+1 {len(self.tokenizer) + 1}"

        # .float(): encoded comes back fp16 under CUDA autocast (model.py).
        probs = log_probs.float().cpu().numpy()
        max_frames = probs.shape[1]
        lens = lengths.clamp(min=0, max=max_frames).cpu().tolist()
        # Slice off padding: greedy masks it via `time < lengths`, beam search would
        # otherwise decode the padded tail into spurious tokens.
        logits_list = [
            np.ascontiguousarray(probs[i, : lens[i], :]) for i in range(len(lens))
        ]

        kwargs = dict(
            beam_width=self.beam_size,
            beam_prune_logp=self.beam_prune_logp,
            token_min_logp=self.token_min_logp,
        )
        nonempty = [i for i, n in enumerate(lens) if n > 0]

        results: List[Tuple[str, List[int], List[int]]] = [("", [], [])] * len(lens)
        if not nonempty:
            return results

        if self._pool is not None:
            # decode_beams_batch yields OutputBeamMPSafe: (text, text_frames, ...)
            batch = self._decoder.decode_beams_batch(
                self._pool, [logits_list[i] for i in nonempty], **kwargs
            )
            beams = [(b[0][0], b[0][1]) if b else ("", []) for b in batch]
        else:
            # decode_beams yields OutputBeam: (text, lm_state, text_frames, ...)
            beams = []
            for i in nonempty:
                out = self._decoder.decode_beams(logits_list[i], **kwargs)
                beams.append((out[0][0], out[0][2]) if out else ("", []))

        for i, (text, text_frames) in zip(nonempty, beams):
            results[i] = self._to_contract(text, text_frames)
        return results


def add_beam_args(parser: Any) -> None:
    """Register the shared beam/LM flags on an eval script's ArgumentParser."""
    group = parser.add_argument_group("beam search / LM")
    group.add_argument(
        "--beam_search", action="store_true", help="beam search without an LM"
    )
    group.add_argument("--lm_path", default=None, help="KenLM .arpa or .bin")
    group.add_argument(
        "--unigrams_path", default=None, help="unigram list; required for a binary LM"
    )
    group.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="LM weight")
    group.add_argument(
        "--beta", type=float, default=DEFAULT_BETA, help="word insertion bonus"
    )
    group.add_argument("--beam_size", type=int, default=DEFAULT_BEAM_SIZE)
    group.add_argument("--beam_prune_logp", type=float, default=DEFAULT_BEAM_PRUNE_LOGP)
    group.add_argument("--token_min_logp", type=float, default=DEFAULT_TOKEN_MIN_LOGP)
    group.add_argument(
        "--lm_workers", type=int, default=0, help="decode pool size; 0 = in-process"
    )


def maybe_attach(model: Any, args: Any) -> Optional[CTCBeamLMDecoding]:
    """Swap model.decoding for beam search if the CLI asked for it.

    Returns the decoder (so the caller can close() its pool), or None when the run
    stays greedy.
    """
    if not (getattr(args, "beam_search", False) or getattr(args, "lm_path", None)):
        return None
    if not getattr(model.decoding.tokenizer, "charwise", False):
        raise ValueError(
            "Beam search requires a charwise model (e.g. multilingual_ctc)"
        )

    decoding = CTCBeamLMDecoding(
        model.decoding.tokenizer.vocab,
        lm_path=args.lm_path,
        unigrams_path=args.unigrams_path,
        alpha=args.alpha,
        beta=args.beta,
        beam_size=args.beam_size,
        beam_prune_logp=args.beam_prune_logp,
        token_min_logp=args.token_min_logp,
        num_workers=args.lm_workers,
    )
    model.decoding = decoding
    lm_desc = (
        f"LM={args.lm_path} alpha={args.alpha} beta={args.beta}"
        if args.lm_path
        else "no LM"
    )
    print(f"Decoding: beam search (beam={args.beam_size}, {lm_desc})")
    return decoding
