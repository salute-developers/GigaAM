"""Test original PyTorch GigaAM for baseline comparison."""
import sys, time
sys.path.insert(0, "..")
import gigaam

model = gigaam.load_model("v3_ctc", device="cpu", fp16_encoder=False, use_flash=False)

t0 = time.time()
result = model.transcribe("test_ru.wav")
elapsed = time.time() - t0

print(f"GigaAM v3_ctc (PyTorch): {result}")
print(f"Time: {elapsed:.2f}s")
