"""Test GigaAM MLX model in float32."""
import time
import sys
sys.path.insert(0, ".")

from gigaam_mlx import load_model, load_audio

print("Loading GigaAM MLX fp32 model...")
model = load_model("./gigaam-v3-ctc-mlx-fp32")

audio = load_audio("test_ru.wav")

# warmup
_ = model.transcribe(audio)

t0 = time.time()
result = model.transcribe(audio)
elapsed = time.time() - t0

print(f"GigaAM v3_ctc (MLX fp32): {result}")
print(f"Time: {elapsed:.2f}s")

# Compare with reference
ref = "ничьих не требуя похвал счастлив уж я надеждой сладкой что дева с трепетом любви посмотрит может быть украдкой на песни грешные мои у лукоморья дуб зеленый"
print(f"Reference:                {ref}")
print(f"Match: {result == ref}")
