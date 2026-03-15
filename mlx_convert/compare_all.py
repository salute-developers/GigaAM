"""Full comparison: Whisper CPP vs GigaAM PyTorch vs GigaAM MLX."""
import subprocess
import time
import sys

sys.path.insert(0, "..")
sys.path.insert(0, ".")

AUDIO = "test_ru.wav"

print("=" * 70)
print("COMPARISON: Russian speech transcription")
print("Audio: GigaAM example.wav (~11.3s, Pushkin poetry)")
print("=" * 70)

# ── 1. Whisper CPP ──
print("\n📢 1. Whisper CPP (small, Russian)")
t0 = time.time()
result = subprocess.run(
    ["whisper-cli", "-m", "ggml-small.bin", "-l", "ru", "-f", AUDIO, "-nt"],
    capture_output=True, text=True
)
whisper_time = time.time() - t0
whisper_text = ""
for line in result.stdout.strip().split("\n"):
    line = line.strip()
    if line and not line.startswith("["):
        whisper_text += line + " "
    elif line.startswith("["):
        # extract text after timestamp
        parts = line.split("]", 1)
        if len(parts) > 1:
            whisper_text += parts[1].strip() + " "
whisper_text = whisper_text.strip()
print(f"   Text: {whisper_text}")
print(f"   Time: {whisper_time:.2f}s")

# ── 2. GigaAM PyTorch ──
print("\n📢 2. GigaAM v3_ctc (PyTorch, CPU)")
import gigaam
model_pt = gigaam.load_model("v3_ctc", device="cpu", fp16_encoder=False, use_flash=False)
t0 = time.time()
pt_text = model_pt.transcribe(AUDIO)
pt_time = time.time() - t0
print(f"   Text: {pt_text}")
print(f"   Time: {pt_time:.2f}s")

# ── 3. GigaAM MLX ──
print("\n📢 3. GigaAM v3_ctc (MLX, Apple Silicon)")
from gigaam_mlx import load_model, load_audio
model_mlx = load_model("./gigaam-v3-ctc-mlx")
audio = load_audio(AUDIO)

# Warmup
_ = model_mlx.transcribe(audio)

t0 = time.time()
mlx_text = model_mlx.transcribe(audio)
mlx_time = time.time() - t0
print(f"   Text: {mlx_text}")
print(f"   Time: {mlx_time:.2f}s")

# ── Summary ──
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Model':<35} {'Time':>8}  Text")
print("-" * 70)
print(f"{'Whisper CPP (small)':<35} {whisper_time:>7.2f}s  {whisper_text[:60]}...")
print(f"{'GigaAM v3_ctc (PyTorch/CPU)':<35} {pt_time:>7.2f}s  {pt_text[:60]}...")
print(f"{'GigaAM v3_ctc (MLX/M4)':<35} {mlx_time:>7.2f}s  {mlx_text[:60]}...")
print()

# Reference text (Pushkin)
ref = "ничьих не требуя похвал счастлив уж я надеждой сладкой что дева с трепетом любви посмотрит может быть украдкой на песни грешные мои у лукоморья дуб зеленый"
print(f"Reference: {ref}")
