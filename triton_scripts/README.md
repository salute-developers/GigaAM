# GigaAM on Triton Inference Server

Serve `v4_ctc`, `v4_rnnt`, or `multilingual_v2_rnnt` for full-context inference or packet streaming.

## Export and serve

Run from `triton_scripts`. For TensorRT, use the same TensorRT version to build
and serve the plans. Put `trtexec` on PATH or set `TRTEXEC`.

```bash
python run_convert_onnx.py v4_ctc --contexts 160 --repository repos
python run_convert_trt.py repos/v4_ctc_dual_dynamic_trt_fp16/1/graphs/*.onnx
docker build -t gigaam-triton .
docker run --rm --gpus all --shm-size=1g -p 8000:8000 \
  -v "$(pwd)/repos:/models" -v "$(pwd)/..:/opt/gigaam_repo" \
  -e PYTHONPATH=/opt/gigaam_repo gigaam-triton \
  tritonserver --model-repository=/models
```

The default provider is TensorRT FP16. For ONNX Runtime CUDA, add
`--provider CUDAExecutionProvider` to the export and skip plan generation.
`--contexts 160` creates a dynamic endpoint with 160 ms as the default. The
client can select another context or request full-context inference.

## Send audio

Install `tritonclient[http]` on the client and use the Triton server's HTTP
address. Audio is mono FP32 at 16 kHz.

```python
from tritonclient.http import InferenceServerClient
from run_client import infer_dual_packet, infer_dual_packets

model = "v4_ctc_dual_dynamic_trt_fp16"
with InferenceServerClient("triton.example.com:8000") as client:
    text = infer_dual_packet(wave, model, client=client, stream_id="clip-id",
                             start=True, final=True, streaming=False)
    first = infer_dual_packets({"stream-a": a1, "stream-b": b1}, model,
                               client=client, start=True, future_context_ms=160)
    last = infer_dual_packets({"stream-a": a2, "stream-b": b2}, model,
                              client=client, final=True)
```

A request supports up to four distinct streams; each packet may contain
multiple complete chunks, within the engine shape profile. Use stream IDs
unique across clients, send each stream's packets in order, and route each
stream to the same replica until `final=True` flushes and releases its state.
Concatenate `raw` and `e2e` deltas separately. For multilingual RNNT, pass
`language="auto"` or a language code when starting the stream.
Independent clip batches can run concurrently with separate HTTP client
handles. Keep at most one request in flight for each live stream.

## Streaming throughput

Mean RTFx on English FLEURS audio at 160, 320, and 640 ms contexts on NVIDIA
H100. One pass sends a whole clip at the selected finite context. Long
packets are 1.5–3 s and short packets are 0.2–0.8 s. Torch uses its FP16
encoder, and TensorRT uses FP16 plans.

### `v4_ctc`

| Runtime | One pass | Long packets | Short packets |
| --- | ---: | ---: | ---: |
| Torch, BS1 | 497.1 | 57.7 | 27.8 |
| ONNX CUDA, BS1 | 686.9 | 72.3 | 26.7 |
| ONNX CUDA, BS4 | 2,091.8 | 143.2 | 58.1 |
| Triton / ONNX CUDA, BS≤4 | 1,820.9 | 135.4 | 53.2 |
| Triton / TensorRT FP16, BS≤4 | 3,280.4 | 196.8 | 99.7 |

### `v4_rnnt`

| Runtime | One pass | Long packets | Short packets |
| --- | ---: | ---: | ---: |
| Torch, BS1 | 318.7 | 55.4 | 27.7 |
| ONNX CUDA, BS1 | 332.1 | 88.4 | 30.3 |
| ONNX CUDA, BS4 | 792.1 | 195.2 | 62.3 |
| Triton / ONNX CUDA, BS≤4 | 776.6 | 195.2 | 57.5 |
| Triton / TensorRT FP16, BS≤4 | 2,579.0 | 317.9 | 101.9 |

### `multilingual_v2_rnnt` (auto language)

| Runtime | One pass | Long packets | Short packets |
| --- | ---: | ---: | ---: |
| Torch, BS1 | 204.0 | 34.5 | 14.5 |
| ONNX CUDA, BS1 | 214.0 | 51.5 | 17.2 |
| ONNX CUDA, BS4 | 477.2 | 124.0 | 39.2 |
| Triton / ONNX CUDA, BS≤4 | 489.4 | 121.8 | 36.8 |
| Triton / TensorRT FP16, BS≤4 | 1,443.4 | 204.6 | 55.4 |
