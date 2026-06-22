#!/bin/bash
# Деплоится на VPS (slack_zoom). Always-on сторож: каждые ~5 мин пуллит healthz
# Mac-транскрайбера через relay-туннель и алертит в Telegram при падении/восстановлении.
# Ловит то, что Mac сам сообщить не может: Mac уснул / туннель упал / транскрайбер лёг.
#
# Установка на VPS:
#   scp mac_transcriber/scripts/vps_mac_watchdog.sh root@VPS:/root/mac_watchdog.sh
#   ssh root@VPS 'chmod +x /root/mac_watchdog.sh'
#   # создать /root/.mac-watchdog.env с TELEGRAM_TOKEN/TELEGRAM_CHAT_ID (см. ниже)
#   ssh root@VPS '(crontab -l 2>/dev/null; echo "*/5 * * * * /root/mac_watchdog.sh") | crontab -'
#
# /root/.mac-watchdog.env (chmod 600):
#   TELEGRAM_TOKEN=123456:ABC...
#   TELEGRAM_CHAT_ID=987654321
#   RELAY_URL=http://127.0.0.1:18013/healthz   # опц., это дефолт
#   FAIL_THRESHOLD=2                            # опц., алерт после N подряд сбоев
set -uo pipefail

ENV_FILE="${1:-/root/.mac-watchdog.env}"
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && . "$ENV_FILE"
RELAY_URL="${RELAY_URL:-http://127.0.0.1:18013/healthz}"
THRESHOLD="${FAIL_THRESHOLD:-2}"
STATE="/root/.mac-watchdog.state"

notify() {
  [ -n "${TELEGRAM_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ] || return 0
  curl -fsS -m10 "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
    --data-urlencode "text=$1" >/dev/null 2>&1 || true
}

fails="$(cat "$STATE" 2>/dev/null || echo 0)"
case "$fails" in (''|*[!0-9]*) fails=0;; esac

if curl -fsS -m8 "$RELAY_URL" 2>/dev/null | grep -q '"ok"'; then
  # Восстановление: были подряд сбои выше порога -> сообщаем «снова ок».
  if [ "$fails" -ge "$THRESHOLD" ]; then
    notify "✅ Mac-транскрайбер снова доступен (relay $RELAY_URL)."
  fi
  echo 0 >"$STATE"
else
  fails=$((fails + 1))
  echo "$fails" >"$STATE"
  # Алертим РОВНО при пересечении порога (один раз), без спама каждые 5 мин.
  if [ "$fails" -eq "$THRESHOLD" ]; then
    notify "🔴 Mac-транскрайбер недоступен через relay ($RELAY_URL). Пайплайн отчётов стоит: проверь Mac (сон/туннель/сервис)."
  fi
fi
