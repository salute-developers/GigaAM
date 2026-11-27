# MLX evaluation: parity, accuracy, speed

Checks the MLX port against upstream PyTorch GigaAM on a fixed, verifiable corpus.
Latest numbers: [`results/RESULTS.md`](results/RESULTS.md).

## Corpus

200 utterances (823.85 s) from the Golos Crowd test split — the sample published by
[garrrikkotua/gigaam-vs-whisper-russian](https://github.com/garrrikkotua/gigaam-vs-whisper-russian).
`golos_crowd_200.json` holds the row ids and per-clip SHA-256 from that repository; the audio is not
redistributed here (Golos licence) — `prepare_golos.py` downloads the Hugging Face mirror
`bond005/sberdevices_golos_10h_crowd` at a pinned revision and verifies every hash.

References are lowercase Cyrillic, no punctuation, numbers spelled out. CTC/RNNT output already has
that form; E2E output (cased, punctuated, digits) is normalized before scoring, and the report adds
WER on the subset where no E2E transcript contains digits or Latin letters (see `common.normalize`).

## Run

```bash
cd mlx_convert
uv venv .venv && uv pip install --python .venv/bin/python -e "..[torch]" \
    mlx safetensors soundfile pyarrow num2words huggingface_hub pytest

# models: fp16 (default) and fp32 copies
for m in v3_ctc v3_rnnt v3_e2e_ctc v3_e2e_rnnt; do
  .venv/bin/python convert_gigaam_to_mlx.py --model $m --output eval-models/$m
  .venv/bin/python convert_gigaam_to_mlx.py --model $m --output eval-models/$m-fp32 --dtype float32
done

.venv/bin/python eval/prepare_golos.py          # corpus → eval/data/golos_crowd_200
.venv/bin/python -m pytest eval/test_decode.py  # SentencePiece decoder == sentencepiece
.venv/bin/python eval/run_eval.py               # parity + WER/CER  → results/summary.json
.venv/bin/python eval/bench_speed.py --repeats 3  # speed          → results/speed.json
.venv/bin/python eval/make_report.py            # → results/RESULTS.md
```

`run_eval.py` compares, per clip, the transcript and the encoder output of MLX with PyTorch on CPU
in fp32. `bench_speed.py` runs each model/backend in its own subprocess with MLX's buffer cache
capped at 1 GB and aborts above `--max-gb` of framework memory; without the cap MLX keeps freed
buffers for every new input length and can grow to most of unified memory.
