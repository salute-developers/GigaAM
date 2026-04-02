import sys
from typing import List

import librosa
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput

SAMPLE_RATE = 16000


def infer_ensemble(
    wav_paths: List[str],
    model_type: str = "ctc",
    backend: str = "onnx",
    triton_url: str = "localhost:8000",
    timeout: int = 60,
) -> List[str]:
    """
    Run inference on ensemble model.

    Args:
        wav_paths: List of paths to WAV files
        model_type: Type of model - "ctc" or "rnnt"
        backend: Backend type - "onnx" or "trt"
        triton_url: Triton server URL (with or without http:// prefix)
        timeout: Request timeout in seconds

    Returns:
        List of transcribed texts
    """
    if model_type not in ["ctc", "rnnt"]:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'ctc' or 'rnnt'")
    if backend not in ["onnx", "trt"]:
        raise ValueError(f"Invalid backend: {backend}. Must be 'onnx' or 'trt'")

    model_name = f"gigaam_{model_type}_{backend}"

    if triton_url.startswith("http://"):
        triton_url = triton_url[7:]
    elif triton_url.startswith("https://"):
        triton_url = triton_url[8:]

    if triton_url.startswith("localhost"):
        triton_url = triton_url.replace("localhost", "127.0.0.1")

    audio_arrays: List[np.ndarray] = []
    audio_lengths: List[int] = []

    for wav_path in wav_paths:
        audio = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)[0]
        audio_arrays.append(audio.astype(np.float32))
        audio_lengths.append(len(audio))

    audio_batch = np.concatenate(audio_arrays).astype(np.float32)
    audio_lengths_array = np.array(audio_lengths, dtype=np.int64)

    client = InferenceServerClient(
        url=triton_url,
        connection_timeout=timeout,
        network_timeout=timeout,
    )

    input_audio = InferInput("audio_batch", audio_batch.shape, "FP32")
    input_audio.set_data_from_numpy(audio_batch)

    input_lengths = InferInput("audio_lengths", audio_lengths_array.shape, "INT64")
    input_lengths.set_data_from_numpy(audio_lengths_array)

    response = client.infer(model_name, [input_audio, input_lengths])

    texts_bytes = response.as_numpy("texts")
    texts = [text_bytes.decode("utf-8") for text_bytes in texts_bytes]

    return texts


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python run_client.py <model_type> <backend> "
            "<wav_file1> [wav_file2] ..."
        )
        print("  model_type: ctc | rnnt")
        print("  backend: onnx | trt")
        sys.exit(1)

    model_type = sys.argv[1]
    backend = sys.argv[2]
    wav_files = sys.argv[3:]

    if not wav_files:
        print("Error: No WAV files provided")
        sys.exit(1)

    texts = infer_ensemble(wav_files, model_type=model_type, backend=backend)

    for wav_file, text in zip(wav_files, texts):
        print(f"{wav_file}: {text}")
