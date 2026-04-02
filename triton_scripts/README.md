# Triton Inference Server Setup

This setup supports all ASR models from the GigaAM family. Inference is implemented through a Triton ensemble: the client sends WAV files and receives transcribed texts. CTC models are converted to ONNX/TRT entirely, while RNNT models are split into encoder (ONNX/TRT) and decoder/joint components that run in Python using onnxruntime.

## Prerequisites

Navigate to the triton_scripts directory:
```bash
cd triton_scripts
```

## 0. Build Docker Image

Build the Triton Inference Server Docker image:
```bash
docker build -t gigaam-triton .
```

## 1. Convert Models to ONNX

Convert models to ONNX format. This creates `.onnx` checkpoints and configs:
```bash
python run_convert_onnx.py <model_version>  # e.g., v3_ctc, v3_e2e_rnnt
```

**Note:** The script saves model configs to the preprocessing directory. For `v3` family models, preprocessing differs from earlier versions. Since Triton uses a shared preprocessing model, you can only use models with the same preprocessing simultaneously (either all `v3` models or all earlier models). The preprocessing is determined by the last model converted to ONNX.

## 2. Convert ONNX to TensorRT

Convert ONNX models to TensorRT format. This converts the version of the corresponding CTC/RNNT model that was last converted to ONNX. Run inside the TensorRT Docker container:
```bash
docker run --gpus all -it --rm -v $(pwd):/workspace nvcr.io/nvidia/tensorrt:24.10-py3
# inside the container:
bash run_convert_trt.sh <ctc | rnnt>
```

## 3. Start Triton Server

Run the Triton Inference Server:
```bash
docker run --gpus all --ipc=host -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$(pwd)/repos:/models" \
  -v "$(pwd)/..:/opt/gigaam_repo" \
  -e PYTHONPATH=/opt/gigaam_repo \
  gigaam-triton \
  tritonserver --model-repository=/models --exit-on-error=false
```

Python backend models (e.g. [`rnnt_postprocessing`](repos/rnnt_postprocessing/1/model.py)) do `import gigaam`. The package lives next to this directory, under `gigaam_repo/gigaam/`.

**Note:** For ONNX models, the default configuration uses `instance_group [{ kind: KIND_GPU }]`. To enable CPU execution, update the `instance_group` to `KIND_CPU` in the following model config files [`ctc`](repos/ctc_encoder_onnx/config.pbtxt), [`rnnt`](repos/gigaam_encoder_onnx/config.pbtxt).

## 4. Run Client

Run inference using the client:
```bash
python run_client.py <model_type> <backend> <wav_file1> [wav_file2] ...
```

Arguments:
- `model_type`: `ctc` or `rnnt`
- `backend`: `onnx` or `trt`
- `wav_file1`, `wav_file2`, ...: Paths to WAV files

Examples:
```bash
python run_client.py rnnt onnx example.wav
python run_client.py ctc trt audio1.wav audio2.wav audio3.wav
```

## Benchmark

Forward pass time in seconds on CUDA for the first 4 segments from `long_example.wav` (VAD-segmented, ~65s total audio). For torch/onnx — both single-sample and batched inference are shown.

| Backend       | v3_ctc          | v3_e2e_rnnt     |
|:--------------|:----------------|:----------------|
| triton/trt    | 0.034 ± 0.000   | 0.403 ± 0.008   |
| triton/onnx   | 0.046 ± 0.001   | 0.413 ± 0.005   |
| onnx (batch)  | 0.037 ± 0.004   | 0.949 ± 0.017   |
| onnx          | 0.047 ± 0.001   | 1.093 ± 0.045   |
| torch (batch) | 0.036 ± 0.002   | 0.919 ± 0.002   |
| torch         | 0.112 ± 0.003   | 1.008 ± 0.001   |
