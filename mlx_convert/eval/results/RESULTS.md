# GigaAM MLX eval results

Machine: Apple M4 Pro, 24 GB, macOS 26.6.2; Python 3.12.13, mlx 0.32.2, torch 2.14.0, torchaudio 2.11.0.

Corpus: Golos Crowd test, 200 clips, 823.85 s (1001 reference words); the sample of github.com/garrrikkotua/gigaam-vs-whisper-russian, verified by SHA-256.

## Parity: MLX vs PyTorch

Reference: upstream PyTorch GigaAM on CPU in fp32. MLX: the fp16 file loaded with the `load_model`
default (weights upcast to fp32 at load). The upstream encoder weights are stored in fp16, so fp16
conversion loses nothing there; only the small CTC head weights are fp32 in the checkpoint.

| Model | Identical transcripts | Encoder max \|Δ\| (median / max over clips) | CTC frame labels identical |
|---|---:|---:|---:|
| v3_ctc | 200/200 | 3.0e-06 / 8.1e-05 | 100.00% |
| v3_rnnt | 200/200 | 3.2e-06 / 1.7e-05 | — |
| v3_e2e_ctc | 200/200 | 3.9e-06 / 8.7e-05 | 100.00% |
| v3_e2e_rnnt | 200/200 | 3.3e-06 / 1.0e-05 | — |

Median max |encoder output| is 1.54 (v3_ctc) for scale.

## Accuracy (MLX fp16; identical to PyTorch, see above)

Normalization for every hypothesis: lowercase, ё→е, numbers → words (num2words), punctuation and
symbols removed. Corpus WER = total word edits / total reference words.

The last column restricts to the 168 clips where neither E2E transcript contains digits
or Latin letters (E2E writes "HD", "4K", "15-й", phone numbers as digit groups; Golos spells them
out in Cyrillic, and no normalizer maps one onto the other reliably).

| Model | WER (S/I/D) | CER | WER, plain subset |
|---|---:|---:|---:|
| v3_ctc | 1.70% (15/1/1) | 0.30% | 1.35% (11/815) |
| v3_rnnt | 1.50% (12/2/1) | 0.24% | 1.23% (10/815) |
| v3_e2e_ctc | 6.49% (46/13/6) | 3.14% | 2.70% (22/815) |
| v3_e2e_rnnt | 5.69% (42/10/5) | 3.19% | 2.09% (17/815) |

## Speed

In-memory audio, one warm-up pass, then 3 timed passes; median pass reported.
`short`: 200 clips one by one (823.8 s). `long`: 38 segments of ≈20 s made by concatenating consecutive clips (815.0 s). RTFx = audio seconds / wall seconds. PyTorch CPU uses 8 threads; PyTorch MPS runs the upstream fp16-autocast encoder. `mlx_fp16`: fp16 file, weights upcast to fp32 at load (default); `mlx_fp16_stored`: fp16 params kept (`load_model(..., dtype=None)`); `mlx_fp32`: fp32 file. MLX buffer cache capped at 1 GB; one subprocess per row. Peak memory: MLX peak allocation / MPS driver allocation / CPU max RSS — not directly comparable.

| Model | Backend | Load, s | short RTFx | short p50, ms | long RTFx | long p50, ms | Peak mem, GB |
|---|---|---:|---:|---:|---:|---:|---:|
| v3_ctc | mlx_fp16 | 0.1 | 201.4 | 20.0 | 285.2 | 74.4 | 1.81 |
| v3_ctc | mlx_fp16_stored | 0.0 | 167.4 | 24.5 | 270.0 | 79.6 | 1.35 |
| v3_ctc | mlx_fp32 | 0.2 | 199.5 | 19.9 | 286.8 | 76.0 | 1.81 |
| v3_ctc | torch_mps | 1.0 | 200.6 | 20.2 | 268.0 | 80.9 | 1.63 |
| v3_ctc | torch_cpu | 1.0 | 10.0 | 402.6 | 41.2 | 507.2 | 2.01 |
| v3_rnnt | mlx_fp16 | 0.1 | 90.9 | 43.4 | 102.4 | 208.9 | 1.82 |
| v3_rnnt | mlx_fp16_stored | 0.0 | 72.8 | 54.6 | 87.8 | 249.2 | 1.35 |
| v3_rnnt | mlx_fp32 | 0.2 | 86.6 | 45.6 | 97.0 | 222.1 | 1.82 |
| v3_rnnt | torch_mps | 1.1 | 37.1 | 104.1 | 40.5 | 515.6 | 0.66 |
| v3_rnnt | torch_cpu | 1.0 | 9.8 | 412.7 | 39.0 | 542.1 | 2.03 |
| v3_e2e_ctc | mlx_fp16 | 0.1 | 204.5 | 19.7 | 288.8 | 74.9 | 1.81 |
| v3_e2e_ctc | mlx_fp16_stored | 0.0 | 168.4 | 24.1 | 266.1 | 80.8 | 1.35 |
| v3_e2e_ctc | mlx_fp32 | 0.2 | 198.7 | 20.7 | 279.3 | 77.2 | 1.81 |
| v3_e2e_ctc | torch_mps | 1.1 | 202.1 | 20.1 | 271.1 | 79.2 | 0.63 |
| v3_e2e_ctc | torch_cpu | 1.0 | 10.0 | 403.4 | 41.4 | 503.6 | 2.02 |
| v3_e2e_rnnt | mlx_fp16 | 0.1 | 98.8 | 41.4 | 112.4 | 191.2 | 1.82 |
| v3_e2e_rnnt | mlx_fp16_stored | 0.0 | 79.9 | 50.5 | 94.1 | 224.6 | 1.35 |
| v3_e2e_rnnt | mlx_fp32 | 0.2 | 92.2 | 43.1 | 104.3 | 206.2 | 1.82 |
| v3_e2e_rnnt | torch_mps | 1.1 | 44.5 | 87.7 | 48.2 | 445.9 | 0.66 |
| v3_e2e_rnnt | torch_cpu | 1.0 | 10.0 | 409.8 | 41.4 | 516.9 | 2.04 |
