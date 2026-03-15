"""Test GigaAM MLX model."""
import time
import sys

sys.path.insert(0, ".")

from gigaam_mlx import load_model, load_audio

print("Loading GigaAM MLX model...")
t0 = time.time()
model = load_model("./gigaam-v3-ctc-mlx")
load_time = time.time() - t0
print(f"Model loaded in {load_time:.2f}s")

print("\nLoading audio...")
audio = load_audio("test_ru.wav")
print(f"Audio: {audio.shape[0]} samples ({audio.shape[0]/16000:.2f}s)")

print("\nTranscribing...")
t0 = time.time()
result = model.transcribe(audio)
transcribe_time = time.time() - t0

print(f"\nGigaAM v3_ctc (MLX): {result}")
print(f"Transcription time: {transcribe_time:.2f}s")
