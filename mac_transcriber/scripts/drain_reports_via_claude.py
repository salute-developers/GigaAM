#!/usr/bin/env python3
"""Слив очереди отчётов через CLAUDE (а не платный LLM-API).

Дроп-ин замена reprocess_blocked.py: берёт встречи, поставленные сервисом в очередь
(blocked_on_ai / blocked_on_quota — при MAC_TRANSCRIBER_REPORT_BACKEND=claude туда
попадает КАЖДАЯ встреча), плюс старые встречи без нормального отчёта (заглушка/local),
и для каждой:

  1. agent_report.prepare  : transcript.json -> agent_input.json (канонич. segment_id)
  2. claude -p             : читает agent_input.json -> пишет ai_payload.json
  3. agent_report.finalize : payload -> рендер report.{md,json,html,typ,pdf}+coverage+health+slack
  4. upload_reports_to_yandex --meeting-dirs : заливка на Я.Диск ('<дата> <заголовок>')
  5. service._write_status(completed)         : финализация (манифест/память) + VPS/Slack подхватит

Idle-дёшево: если очередь пуста — выходим ДО запуска claude (ноль токенов).
Вешается на launchd (ежечасно). claude гоняется с --allowedTools 'Read Write Edit'
и bypassPermissions (без Bash/сети) — он только читает вход и пишет payload.

Использование:
    .venv/bin/python mac_transcriber/scripts/drain_reports_via_claude.py [--dry-run] [--limit N] [--no-upload] [--no-pdf]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# reprocess_blocked не тянет service на импорте, поэтому переиспользуем его
# helper'ы (_read_status и пр.) уже на уровне модуля — env грузим до import service.
import reprocess_blocked as rb  # noqa: E402

INSTRUCTIONS = SCRIPTS_DIR / "report_agent_instructions.md"
UPLOAD_SCRIPT = SCRIPTS_DIR / "upload_reports_to_yandex.py"
CLAUDE_TIMEOUT = int(os.environ.get("MAC_TRANSCRIBER_CLAUDE_TIMEOUT", "1500"))
# Кросс-процессный лок: не даём двум дренажам наложиться (launchd + ручной запуск).
LOCK_PATH = REPO_ROOT / ".local" / "drain_reports.lock"
# Кап попыток генерации для blocked-встреч (защита от вечно падающей встречи).
MAX_ATTEMPTS = 3


def _load_env_files(paths: list[Path]) -> None:
    for path in paths:
        expanded = path.expanduser()
        if not expanded.exists():
            continue
        for line in expanded.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _claude_bin() -> str:
    return (
        os.environ.get("CLAUDE_BIN")
        or shutil.which("claude")
        or str(Path.home() / ".local" / "bin" / "claude")
    )


# Сигналы исчерпания лимитов / недоступности AI в выводе claude (для логов).
LIMIT_MARKERS = (
    "usage limit",
    "rate limit",
    "limit reached",
    "limit will reset",
    "resets at",
    "overloaded",
    "quota",
    "exceeded your",
    "429",
)


def _payload_is_valid(path: Path) -> bool:
    """ai_payload.json существует, валиден и содержит непустые adaptive_sections."""
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and bool(payload.get("adaptive_sections"))


def _classify_generation(payload_valid: bool, payload_exists: bool) -> str:
    """Исход генерации: ok | limit | fail.

    - valid payload                -> "ok"
    - файл есть, но кривой/пустой   -> "fail" (claude отработал, но брак — детерминированно,
                                       идёт в счётчик попыток и под кап)
    - файла НЕТ вообще              -> "limit" (claude не отработал: лимит/авторизация/сеть/
                                       overload — ТРАНЗИЕНТНО: не штрафуем, повторим позже)
    """
    if payload_valid:
        return "ok"
    if payload_exists:
        return "fail"
    return "limit"


def _generate_payload(work: Path, model: str) -> str:
    """claude -p: читает agent_input.json -> пишет ai_payload.json. Возвращает ok|limit|fail."""
    out = work / "ai_payload.json"
    if out.exists():
        out.unlink()
    prompt = (
        f"Прочитай инструкции {INSTRUCTIONS} ПОЛНОСТЬЮ, затем прочитай вход "
        f"{work / 'agent_input.json'} ЦЕЛИКОМ (большой файл — читай несколькими Read). "
        f"Сгенерируй подробный отчёт-payload строго по схеме из инструкций и запиши его "
        f"как валидный JSON через Write в {out}. Запиши ТОЛЬКО этот файл, ничего больше."
    )
    cmd = [
        _claude_bin(),
        "-p",
        prompt,
        "--allowedTools",
        "Read Write Edit",
        "--permission-mode",
        "bypassPermissions",
    ]
    if model:
        cmd += ["--model", model]
    # DISABLE_AUTOUPDATER: фоновый джоб НЕ инициирует само-обновление claude (чтобы
    # неинтерактивный запуск не прыгнул на потенциально несовместимую версию сам по
    # себе). Обновления прилетают из интерактивных сессий пользователя; джоб берёт
    # ту версию, что уже установлена. Авто-апдейт НЕ прерывает работающий процесс —
    # применяется к следующему запуску, так что длинная генерация безопасна.
    env = {**os.environ, "DISABLE_AUTOUPDATER": "1"}
    try:
        res = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            timeout=CLAUDE_TIMEOUT,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
    except subprocess.TimeoutExpired:
        # Таймаут — детерминированная проблема этого прогона (завис/слишком большой вход),
        # считаем fail (идёт под кап), а не лимит.
        print("   claude timeout -> fail", flush=True)
        return "fail"
    tail = ((res.stdout or "") + " " + (res.stderr or "")).strip()
    status = _classify_generation(_payload_is_valid(out), out.exists())
    if status == "limit":
        is_limit = any(m in tail.lower() for m in LIMIT_MARKERS)
        reason = "лимит/недоступность" if is_limit else "не отработал (нет payload)"
        print(f"   claude {reason}; tail: …{tail[-180:]}", flush=True)
    elif status == "fail":
        print(f"   payload невалиден (брак); tail: …{tail[-180:]}", flush=True)
    return status


def _acquire_lock(lock_path: Path):
    """Берём эксклюзивный неблокирующий файловый лок.

    Возвращает открытый дескриптор (его нельзя закрывать до конца процесса) при
    успехе или ``None``, если лок уже держит другой процесс.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_path, "w", encoding="utf-8")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fd.close()
        return None
    return fd


def _health_status(meeting_dir: Path) -> str:
    """Поле ``status`` из ``artifacts/report_health.json`` (или "" если нет/битый)."""
    path = meeting_dir / "artifacts" / "report_health.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(data, dict):
        return ""
    return str(data.get("status") or "")


def _needs_disk_upload(meeting_dir: Path) -> bool:
    """True, если отчёт отрендерен (``artifacts/report.md``), но ещё не залит на Я.Диск."""
    if not (meeting_dir / "artifacts" / "report.md").exists():
        return False
    return rb._read_status(meeting_dir).get("report_disk_uploaded") is not True


def _mark_disk_uploaded(meeting_dir: Path) -> None:
    """Помечаем в status.json, что отчёт успешно залит на Я.Диск (идемпотентно)."""
    status = rb._read_status(meeting_dir)
    status["report_disk_uploaded"] = True
    (meeting_dir / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _bump_attempt(meeting_dir: Path) -> int:
    """Счётчик попыток генерации для blocked-встречи (до запуска claude).

    Возвращает новое значение счётчика.
    """
    status = rb._read_status(meeting_dir)
    count = int(status.get("report_attempt_count") or 0) + 1
    status["report_attempt_count"] = count
    (meeting_dir / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return count


def _attempt_count(meeting_dir: Path) -> int:
    """Текущее число попыток генерации для blocked-встречи."""
    return int(rb._read_status(meeting_dir).get("report_attempt_count") or 0)


def _upload_one(meeting_dir: Path) -> bool:
    """Поштучная заливка одной встречи на Я.Диск. True при rc==0."""
    rc = subprocess.run(
        [
            sys.executable,
            str(UPLOAD_SCRIPT),
            "--meeting-dirs",
            str(meeting_dir),
            "--apply",
        ],
        cwd=str(REPO_ROOT),
        check=False,
    ).returncode
    return rc == 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env-file", type=Path, action="append", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-upload", action="store_true", help="не заливать на Я.Диск")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument(
        "--model", default=os.environ.get("MAC_TRANSCRIBER_CLAUDE_MODEL", "")
    )
    args = ap.parse_args(argv)

    _load_env_files(
        args.env_file if args.env_file is not None else [REPO_ROOT / ".env.local"]
    )

    # Кросс-процессный лок: не даём двум дренажам наложиться. При --dry-run ничего не
    # пишем, поэтому лок не нужен. Дескриптор держим открытым до конца процесса.
    lock_fd = None
    if not args.dry_run:
        lock_fd = _acquire_lock(LOCK_PATH)
        if lock_fd is None:
            print("another drain is running, exit")
            return 0

    # reprocess_blocked содержит готовый отбор очереди — переиспользуем.
    import agent_report as ar
    from mac_transcriber import service

    meetings_root = service.ROOT / "meetings"
    blocked = rb._blocked_meetings(meetings_root)
    regen = rb._needs_regen_meetings(meetings_root, blocked)
    regen_set = {m.resolve() for m in regen}
    queue = blocked + regen
    if args.limit:
        queue = queue[: args.limit]

    # Self-heal: завершённые встречи с отрендеренным отчётом, который ещё не залит на
    # Я.Диск (заливка прошлого прохода упала). Дозальём в конце БЕЗ перегенерации.
    queue_set = {d.resolve() for d in queue}
    reupload_only: list[Path] = []
    if not args.no_upload:
        for status_path in meetings_root.glob("*/status.json"):
            md = status_path.parent
            if md.resolve() in queue_set:
                continue  # эти и так пройдут штатную поштучную заливку
            if rb._read_status(md).get("status") != "completed":
                continue
            if _needs_disk_upload(md):
                reupload_only.append(md)

    if not queue and not reupload_only:
        print("Nothing to do: queue empty.")  # idle: claude не запускаем
        return 0

    print(
        f"{len(queue)} meeting(s): {len(blocked)} blocked, {len(regen)} stale "
        f"(oldest first); {len(reupload_only)} reupload-only"
    )
    for d in queue:
        tag = "regen" if d.resolve() in regen_set else "blocked"
        print(f"  - [{tag}] {d.name}")
    for d in reupload_only:
        print(f"  - [reupload] {d.name}")
    if args.dry_run:
        return 0

    processed: list[Path] = []
    failed = 0
    limit_hit = False
    work_root = Path(tempfile.mkdtemp(prefix="claude_reports_"))
    try:
        for d in queue:
            print(f"\n== {d.name}: prepare -> claude -> finalize", flush=True)
            is_regen = d.resolve() in regen_set
            # Кап тратится ТОЛЬКО на детерминированные сбои (claude отработал, но дал брак;
            # finalize/health упали). Временный лимит счётчик НЕ трогает — см. ниже.
            # Для blocked проверяем уже накопленные попытки; regen уже отфильтрован по
            # своему счётчику в _needs_regen_meetings.
            if not is_regen and _attempt_count(d) >= MAX_ATTEMPTS:
                print("   skip: attempt cap reached", flush=True)
                continue
            work = work_root / d.name
            work.mkdir(parents=True, exist_ok=True)

            def _bump_fail(meeting: Path = d, regen: bool = is_regen) -> None:
                """Засчитать ДЕТЕРМИНИРОВАННЫЙ сбой (идёт под кап)."""
                rb._bump_regen_count(meeting) if regen else _bump_attempt(meeting)

            try:
                ar.cmd_prepare(
                    argparse.Namespace(meeting_dir=str(d), work_dir=str(work))
                )
                status = _generate_payload(work, args.model)
            except Exception as exc:  # noqa: BLE001
                print(f"   prepare/generate FAILED (left as-is): {exc}", flush=True)
                _bump_fail()
                failed += 1
                continue

            if status == "limit":
                # Лимиты/недоступность AI: НЕ штрафуем и НЕ помечаем — встреча остаётся
                # blocked. Останавливаем прогон: остальные тоже упрутся в лимит. Подберём
                # их следующим часовым запуском, когда лимиты восстановятся.
                print(
                    "   AI лимит/недоступность -> стоп без штрафа; подхватится, "
                    "когда лимиты обновятся",
                    flush=True,
                )
                limit_hit = True
                break
            if status == "fail":
                _bump_fail()
                failed += 1
                continue

            try:
                ar.cmd_finalize(
                    argparse.Namespace(
                        meeting_dir=str(d),
                        work_dir=str(work),
                        out_dir=str(d / "artifacts"),
                        pdf=not args.no_pdf,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                print(f"   finalize FAILED (left as-is): {exc}", flush=True)
                _bump_fail()
                failed += 1
                continue
            # Health-гейт: failed -> отчёт битый, не помечаем completed (повторится).
            # ok/degraded -> норма, финализируем штатно.
            if _health_status(d) == "failed":
                print("   health=failed -> left blocked", flush=True)
                _bump_fail()
                failed += 1
                continue
            # Финализация штатным путём (манифест/память) + статус completed -> VPS/Slack.
            service._write_status(
                d,
                "completed",
                phase="completed",
                progress=1.0,
                message="Report generated (claude)",
                result={"report_generator": "claude"},
            )
            processed.append(d)
            print("   -> completed", flush=True)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    # Заливка отчётов на Я.Диск (титульные имена папок) — ПОШТУЧНО, отметка только при
    # успехе (rc==0). Что не залилось — дозальётся в следующий проход (report_disk_uploaded
    # не выставлен). Сюда же примешиваем reupload-only из self-heal.
    if not args.no_upload:
        to_upload = processed + reupload_only
        if to_upload:
            print(
                f"\n== upload {len(to_upload)} report(s) to Yandex.Disk "
                f"({len(processed)} fresh, {len(reupload_only)} reupload)",
                flush=True,
            )
            for d in to_upload:
                if _upload_one(d):
                    _mark_disk_uploaded(d)
                    print(f"   uploaded: {d.name}", flush=True)
                else:
                    print(
                        f"   upload FAILED (will retry next run): {d.name}", flush=True
                    )

    if limit_hit:
        print(
            "\n⚠ остановлено по лимиту AI — оставшиеся встречи остались в очереди "
            "(blocked) без штрафа; следующий часовой прогон добьёт их, когда лимиты "
            "восстановятся."
        )
    print(f"\nDone. processed={len(processed)} failed={failed} of {len(queue)}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
