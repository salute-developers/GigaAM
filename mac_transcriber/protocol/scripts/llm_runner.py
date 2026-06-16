#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оркестратор пайплайна протоколов:
inventory -> chunks -> extraction -> validate(loop) -> merge ->
protocol -> gate(loop) -> render.

Окружение:
  ANTHROPIC_API_KEY  ключ API (никогда не хардкодить)
  PROTOCOL_MODEL     модель, по умолчанию claude-sonnet-4-6

Запуск:
  python3 scripts/llm_runner.py transcript.md --meta meta.json -o build/

meta.json (готовит вызывающий код, не LLM):
  {"meeting": {"doc_id": "PRT-260612-01", "title": "...",
   "date_human": "...", "date_short": "12.06.2026", "time_human": "...",
   "duration": "52:34", "segments": 662, "source": "...",
   "transcript": "...", "coverage": "coverage.json", "generator": "..."},
   "participants": ["Дарья", "Азиз", "..."]}
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

MODEL = os.environ.get("PROTOCOL_MODEL", "claude-sonnet-4-6")
client = anthropic.Anthropic()


def strip_fences(t: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip(), flags=re.M).strip()


def ask_json(prompt_file: str, payload: dict, max_tokens: int = 8000) -> dict:
    system = (PROMPTS / prompt_file).read_text(encoding="utf-8")
    user = json.dumps(payload, ensure_ascii=False)
    err = None
    for _ in range(3):
        msg = client.messages.create(
            model=MODEL, max_tokens=max_tokens, temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}])
        text = "".join(b.text for b in msg.content if b.type == "text")
        try:
            return json.loads(strip_fences(text))
        except json.JSONDecodeError as e:
            err = e
            user += ("\n\nПредыдущий ответ не распарсился как JSON. "
                     "Верни строго один JSON-объект без пояснений и markdown.")
    raise RuntimeError(f"{prompt_file}: невалидный JSON после 3 попыток: {err}")


def tool(*cmd) -> int:
    r = subprocess.run([sys.executable, str(SCRIPTS / "protocol_pipeline.py"),
                        *map(str, cmd)], capture_output=True, text=True)
    print(r.stdout, end="")
    if r.stderr:
        print(r.stderr, end="", file=sys.stderr)
    return r.returncode


def save(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1),
                    encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("--meta", help="meta.json с блоком meeting и participants")
    ap.add_argument("-o", "--out", default="build")
    ap.add_argument("--chunk-size", type=int, default=60)
    ap.add_argument("--overlap", type=int, default=6)
    ap.add_argument("--force", action="store_true",
                    help="продолжить при несовпадении числа сегментов")
    args = ap.parse_args()

    build = Path(args.out)
    build.mkdir(parents=True, exist_ok=True)
    seg = build / "segments.json"
    ext_p = build / "extraction.json"
    gaps = build / "gaps.json"
    cov = build / "coverage.json"
    proto_p = build / "protocol.json"

    # 1. Инвентаризация (детерминированная)
    rc = tool("inventory", args.transcript, "-o", seg)
    if rc != 0 and not args.force:
        sys.exit("inventory: число сегментов не совпало с заявленным. "
                 "Чини сквозные ID на этапе ASR или запусти с --force.")

    # 2. Чанки
    chunks_dir = build / "chunks"
    tool("chunks", seg, "--size", args.chunk_size,
         "--overlap", args.overlap, "-o", chunks_dir)

    # 3. Extraction по чанкам
    classification, facts = [], []
    for ch in sorted(chunks_dir.glob("chunk-*.json")):
        res = ask_json("phase1.txt", json.loads(ch.read_text(encoding="utf-8")))
        classification += res.get("classification", [])
        facts += res.get("facts", [])
        print(f"[extract] {ch.name}: +{len(res.get('facts', []))} фактов")
    ext = {"classification": classification, "facts": facts}
    save(ext_p, ext)

    # 4. Цикл полноты классификации
    for i in range(3):
        if tool("validate", seg, ext_p, "--gaps", gaps) == 0:
            break
        res = ask_json("phase1b.txt", json.loads(gaps.read_text(encoding="utf-8")))
        ext["classification"] += res.get("classification", [])
        ext["facts"] += res.get("facts", [])
        save(ext_p, ext)
        print(f"[validate] итерация {i + 1}: дозаполнено")
    else:
        sys.exit("validate: дыры не закрылись за 3 итерации")

    # 5. Слияние дублей
    merged = ask_json("phase2.txt", {"facts": ext["facts"]}, max_tokens=16000)
    save(build / "facts_merged.json", merged)

    # 6. Сборка протокола
    proto = ask_json("phase3.txt", {"facts": merged["facts"]}, max_tokens=16000)
    proto.setdefault("merged", {}).update(merged.get("merged", {}))
    if args.meta:
        meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
        proto["meeting"] = {**proto.get("meeting", {}), **meta.get("meeting", {})}
        if meta.get("participants"):
            proto["participants"] = meta["participants"]
    save(proto_p, proto)

    # 7. Цикл полноты протокола
    for i in range(2):
        if tool("gate", ext_p, proto_p, "-o", cov) == 0:
            break
        orphans = json.loads(cov.read_text(encoding="utf-8"))["orphans_required"]
        proto = ask_json("phase3.txt", {
            "facts": merged["facts"],
            "fix_orphans": orphans,
            "current_protocol": proto,
        }, max_tokens=16000)
        proto.setdefault("merged", {}).update(merged.get("merged", {}))
        save(proto_p, proto)
        print(f"[gate] итерация {i + 1}: сироты встроены")
    else:
        sys.exit("gate: остались факты вне протокола")

    # 8. Рендер
    from render_protocol import render
    proto["kpis"] = [
        {"num": len(proto.get("decisions", [])), "cap": "Решений"},
        {"num": len(proto.get("tasks", [])), "cap": "Задач"},
        {"num": len(proto.get("questions", [])), "cap": "Вопросов"},
        {"num": len(proto.get("risks", [])), "cap": "Рисков"},
    ]
    pdf = build / "protocol.pdf"
    render(template_dir=str(ROOT / "templates"),
           out_pdf=str(pdf), data=proto)
    print(f"[done] {pdf} · {cov}")


if __name__ == "__main__":
    main()
