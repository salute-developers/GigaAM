# -*- coding: utf-8 -*-
"""
Контур полноты для генерации протоколов из транскриптов.

Идея: LLM никогда не видит весь транскрипт сразу и никогда не является
последней инстанцией по полноте. Полноту гарантирует код:

  1. inventory   код парсит транскрипт и присваивает S-ID (ground truth)
  2. chunks      код режет на окна с перекрытием для extraction-фазы
  3. validate    код проверяет, что LLM классифицировала КАЖДЫЙ сегмент
                 чанка; недостающие ID уходят на повторный запрос
  4. gate        код проверяет, что каждый substantive-факт попал в
                 протокол; иначе билд падает со списком дыр

Пайплайн: inventory -> (LLM extraction по чанкам) -> validate ->
          (LLM merge) -> (LLM protocol) -> gate -> render_protocol.py

Использование:
  python3 protocol_pipeline.py inventory transcript.md -o segments.json
  python3 protocol_pipeline.py chunks segments.json --size 60 --overlap 6 -o chunks/
  python3 protocol_pipeline.py validate segments.json extraction.json
  python3 protocol_pipeline.py gate extraction.json protocol.json -o coverage.json

Коды выхода: 0 полнота подтверждена, 2 есть дыры (список в stdout/json).
"""

import argparse
import json
import re
import sys
from pathlib import Path

SEG_RE = re.compile(
    r"^(?:\[(S\d{4})(?:\s*-\s*(S\d{4}))?\]\s*)?"
    r"\[(\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2})\]\s*\*\*(.+?):\*\*\s*(.*)$"
)

SUBSTANTIVE = "substantive"
SUPPORTING = "supporting"
NOISE = "noise"
CLASSES = {SUBSTANTIVE, SUPPORTING, NOISE}


# ---------------------------------------------------------------- inventory

def parse_transcript(path: str) -> list[dict]:
    """Детерминированная нумерация. Никакого LLM.

    Если строки несут явные ID из ASR ([S0010] или [S0010-S0012]
    перед таймкодом), используются они; covers хранит весь диапазон
    сырых сегментов, склеенных в реплику. Иначе ID присваиваются
    последовательно. Смешивать режимы в одном файле нельзя."""
    segments = []
    n = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = SEG_RE.match(line.strip())
        if not m:
            continue
        sid_a, sid_b, t0, t1, speaker, text = m.groups()
        if sid_a:
            covers = [f"S{i:04d}" for i in
                      range(int(sid_a[1:]), int((sid_b or sid_a)[1:]) + 1)]
            sid = sid_a
        else:
            n += 1
            sid = f"S{n:04d}"
            covers = [sid]
        segments.append({
            "id": sid, "covers": covers,
            "t0": t0, "t1": t1,
            "speaker": speaker.strip(), "text": text.strip(),
        })
    return segments


def cmd_inventory(args):
    segments = parse_transcript(args.transcript)
    declared = None
    head = Path(args.transcript).read_text(encoding="utf-8")[:2000]
    m = re.search(r"Segments:\s*(\d+)", head)
    if m:
        declared = int(m.group(1))
    raw_count = sum(len(s["covers"]) for s in segments)
    out = {
        "source": args.transcript,
        "count": len(segments),
        "raw_count": raw_count,
        "declared": declared,
        "speakers": sorted({s["speaker"] for s in segments}),
        "segments": segments,
    }
    Path(args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    status = "OK" if declared in (None, len(segments), raw_count) else "MISMATCH"
    print(f"[inventory] {len(segments)} реплик, {raw_count} сырых сегментов "
          f"(заявлено: {declared}) {status} -> {args.out}")
    if status == "MISMATCH":
        sys.exit(2)


# ------------------------------------------------------------------- chunks

def cmd_chunks(args):
    inv = json.loads(Path(args.segments).read_text(encoding="utf-8"))
    segs = inv["segments"]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    step = args.size - args.overlap
    chunks = []
    i = 0
    while i < len(segs):
        chunk = segs[i:i + args.size]
        chunks.append(chunk)
        i += step
        if i + args.overlap >= len(segs):
            break
    # Последний чанк добирает хвост
    if chunks and chunks[-1][-1]["id"] != segs[-1]["id"]:
        chunks.append(segs[-(args.size):])
    for k, chunk in enumerate(chunks, 1):
        payload = {
            "chunk": k,
            "ids": [s["id"] for s in chunk],
            "segments": chunk,
        }
        (outdir / f"chunk-{k:02d}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[chunks] {len(chunks)} чанков по <= {args.size} "
          f"с перекрытием {args.overlap} -> {outdir}/")


# ----------------------------------------------------------------- validate

def load_extraction(path: str) -> dict:
    """
    Формат extraction.json (склейка ответов LLM по всем чанкам):
    {
      "classification": [{"id": "S0001", "class": "noise|supporting|substantive"}],
      "facts": [{
          "id": "F-0001",
          "type": "decision|task|question|risk|info",
          "text": "одно предложение с конкретикой",
          "segments": ["S0008", "S0009"],
          "speaker": "Вячеслав",
          "owner": null, "due": null,
          "entities": ["GigaAM", "Mac mini"]
      }]
    }
    При перекрытии чанков один ID может встретиться дважды:
    конфликт классов разрешается в пользу substantive.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_classes(classification: list[dict]) -> tuple[dict, list]:
    rank = {NOISE: 0, SUPPORTING: 1, SUBSTANTIVE: 2}
    resolved, bad = {}, []
    for row in classification:
        sid, cls = row.get("id"), row.get("class")
        if cls not in CLASSES:
            bad.append(row)
            continue
        if sid not in resolved or rank[cls] > rank[resolved[sid]]:
            resolved[sid] = cls
    return resolved, bad


def cmd_validate(args):
    inv = json.loads(Path(args.segments).read_text(encoding="utf-8"))
    ext = load_extraction(args.extraction)
    inventory_ids = [s["id"] for s in inv["segments"]]
    inv_set = set(inventory_ids)

    resolved, bad_rows = resolve_classes(ext.get("classification", []))
    classified = set(resolved)

    missing = sorted(inv_set - classified)          # LLM пропустила сегмент
    unknown = sorted(classified - inv_set)          # LLM выдумала ID

    fact_refs = set()
    facts_no_seg = []
    for f in ext.get("facts", []):
        refs = [r for r in f.get("segments", []) if r in inv_set]
        if not refs:
            facts_no_seg.append(f.get("id"))
        fact_refs.update(refs)

    # substantive-сегмент обязан быть привязан хотя бы к одному факту
    substantive = {sid for sid, c in resolved.items() if c == SUBSTANTIVE}
    unanchored = sorted(substantive - fact_refs)

    problems = {
        "missing_segments": missing,
        "unknown_segment_ids": unknown,
        "invalid_class_rows": bad_rows,
        "substantive_without_fact": unanchored,
        "facts_without_valid_segments": facts_no_seg,
    }
    ok = not any(problems.values())
    print(json.dumps({
        "ok": ok,
        "inventory": len(inv_set),
        "classified": len(classified & inv_set),
        "substantive": len(substantive),
        "facts": len(ext.get("facts", [])),
        "problems": problems,
    }, ensure_ascii=False, indent=1))
    if not ok:
        # Список для повторного запроса к LLM кладём рядом
        gap = [s for s in inv["segments"]
               if s["id"] in set(missing) | set(unanchored)]
        Path(args.gaps).write_text(
            json.dumps({"segments": gap}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"[validate] дыры -> {args.gaps}, отправь их в gap-промпт",
              file=sys.stderr)
        sys.exit(2)
    print("[validate] полнота классификации подтверждена")


# --------------------------------------------------------------------- gate

def cmd_gate(args):
    """
    protocol.json = DATA из render_protocol.py, но каждый пункт
    (decisions/tasks/questions/risks, элементы timeline) несёт
    "facts": ["F-0001", ...]. Дополнительно допускается реестр
    "merged": {"F-0042": "дубликат F-0007"} для осознанно склеенных.
    """
    ext = load_extraction(args.extraction)
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))

    must_appear = {f["id"]: f for f in ext.get("facts", [])
                   if f.get("type") in {"decision", "task", "question", "risk"}}
    info_facts = {f["id"]: f for f in ext.get("facts", [])
                  if f.get("type") == "info"}

    used = set()
    for key in ("decisions", "tasks", "questions", "risks", "timeline"):
        for item in proto.get(key, []):
            used.update(item.get("facts", []))
    merged = set(proto.get("merged", {}))

    orphans = sorted(set(must_appear) - used - merged)
    info_orphans = sorted(set(info_facts) - used - merged)

    coverage = {
        "facts_total": len(must_appear) + len(info_facts),
        "facts_required": len(must_appear),
        "in_protocol": len(used),
        "merged": len(merged),
        "orphans_required": [
            {"id": o, "type": must_appear[o]["type"],
             "text": must_appear[o]["text"],
             "segments": must_appear[o].get("segments", [])}
            for o in orphans],
        "orphans_info": info_orphans,
    }
    Path(args.out).write_text(
        json.dumps(coverage, ensure_ascii=False, indent=1), encoding="utf-8")
    if orphans:
        print(f"[gate] FAIL: {len(orphans)} фактов не попали в протокол, "
              f"список в {args.out}")
        for o in coverage["orphans_required"][:10]:
            print(f"  {o['id']} [{o['type']}] {o['text'][:90]}")
        sys.exit(2)
    print(f"[gate] OK: все обязательные факты отражены, "
          f"info-сирот: {len(info_orphans)} -> {args.out}")


# --------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("inventory", help="транскрипт -> segments.json")
    a.add_argument("transcript")
    a.add_argument("-o", "--out", default="segments.json")
    a.set_defaults(func=cmd_inventory)

    a = sub.add_parser("chunks", help="segments.json -> чанки для extraction")
    a.add_argument("segments")
    a.add_argument("--size", type=int, default=60)
    a.add_argument("--overlap", type=int, default=6)
    a.add_argument("-o", "--out", default="chunks")
    a.set_defaults(func=cmd_chunks)

    a = sub.add_parser("validate", help="проверка полноты классификации")
    a.add_argument("segments")
    a.add_argument("extraction")
    a.add_argument("--gaps", default="gaps.json")
    a.set_defaults(func=cmd_validate)

    a = sub.add_parser("gate", help="проверка, что факты дошли до протокола")
    a.add_argument("extraction")
    a.add_argument("protocol")
    a.add_argument("-o", "--out", default="coverage.json")
    a.set_defaults(func=cmd_gate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
