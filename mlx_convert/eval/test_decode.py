"""decode_sentencepiece must reproduce SentencePieceProcessor.decode exactly.

Run: pytest mlx_convert/eval/test_decode.py (needs converted E2E models in
mlx_convert/eval-models/, see convert_gigaam_to_mlx.py).
"""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from gigaam_mlx import decode_sentencepiece  # noqa: E402


@pytest.mark.parametrize("name", ["v3_e2e_ctc", "v3_e2e_rnnt"])
def test_matches_sentencepiece(name):
    spm = pytest.importorskip("sentencepiece")
    model_dir = ROOT / "eval-models" / name
    if not (model_dir / "tokenizer.model").exists():
        pytest.skip(f"{model_dir} not converted")
    sp = spm.SentencePieceProcessor(model_file=str(model_dir / "tokenizer.model"))
    cfg = json.loads((model_dir / "config.json").read_text())
    pieces, control = cfg["vocabulary"], cfg["tokenizer_control_token_ids"]
    assert len(pieces) == sp.get_piece_size()

    rng = random.Random(0)
    for _ in range(20000):
        ids = [rng.randrange(len(pieces)) for _ in range(rng.randint(1, 30))]
        assert decode_sentencepiece(ids, pieces, control) == sp.decode(ids), ids
