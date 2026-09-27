#!/usr/bin/env python3
"""Download the Golos Crowd eval sample and verify it byte-for-byte.

The sample is the 200-utterance Golos Crowd test subset published by
https://github.com/garrrikkotua/gigaam-vs-whisper-russian. Its manifest stores
a SHA-256 per clip; `golos_crowd_200.json` keeps those ids and hashes. The
audio itself is not redistributed here (Golos licence), so this script pulls
the Hugging Face mirror, takes the listed rows and checks every hash.

Output: <out>/wav/<id>.wav and <out>/manifest.jsonl with
{"id", "audio", "duration", "reference"} per line.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(HERE / "data" / "golos_crowd_200"))
    args = parser.parse_args()

    spec = json.loads((HERE / "golos_crowd_200.json").read_text())
    parquet = hf_hub_download(
        spec["dataset"], spec["file"], repo_type="dataset", revision=spec["revision"]
    )
    table = pq.read_table(parquet)
    audio, text = table.column("audio"), table.column("transcription")

    out = Path(args.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.jsonl", "w") as manifest:
        for item in spec["items"]:
            row = item["id"]
            data = audio[row].as_py()["bytes"]
            digest = hashlib.sha256(data).hexdigest()
            if digest != item["sha256"]:
                raise SystemExit(f"row {row}: sha256 {digest} != {item['sha256']}")
            path = out / "wav" / f"{row:06d}.wav"
            path.write_bytes(data)
            info = sf.info(io.BytesIO(data))
            if info.samplerate != 16000 or info.channels != 1:
                raise SystemExit(f"row {row}: {info.samplerate} Hz, {info.channels} ch")
            record = {
                "id": row,
                "audio": str(path.relative_to(out)),
                "duration": info.frames / info.samplerate,
                "reference": text[row].as_py(),
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = sum(
        json.loads(line)["duration"] for line in open(out / "manifest.jsonl")
    )
    print(f"{len(spec['items'])} clips verified, {total:.2f} s of audio → {out}")


if __name__ == "__main__":
    main()
