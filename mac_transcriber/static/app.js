const phases = [
  ["uploaded", "Загрузка"],
  ["segmenting", "Сегментация"],
  ["transcribing", "Распознавание"],
  ["writing_artifacts", "Артефакты"],
  ["completed", "Готово"],
];

const artifactLabels = {
  transcript_md: "Markdown",
  transcript_json: "JSON",
  summary_json: "Summary",
  report_md: "Protocol MD",
  report_json: "Protocol JSON",
  report_html: "HTML",
  report_health: "Report Health",
  coverage_json: "Coverage",
  report_typ: "Typst",
  report_pdf: "PDF",
  segments_tsv: "Segments",
  speaker_track_map: "Tracks",
};

const state = {
  apiKey: readStoredApiKey(),
  config: null,
  meetings: [],
  selectedId: null,
  selectedStatus: null,
  transcript: [],
  transcriptMarkdown: "",
  isUploading: false,
  isTicking: false,
};

const el = {
  uploadForm: document.querySelector("#uploadForm"),
  audioFile: document.querySelector("#audioFile"),
  audioFileName: document.querySelector("#audioFileName"),
  participantFiles: document.querySelector("#participantFiles"),
  participantNames: document.querySelector("#participantNames"),
  apiKeyInput: document.querySelector("#apiKeyInput"),
  authDetails: document.querySelector("#authDetails"),
  startButton: document.querySelector("#startButton"),
  dropZone: document.querySelector("#dropZone"),
  refreshButton: document.querySelector("#refreshButton"),
  meetingList: document.querySelector("#meetingList"),
  healthDot: document.querySelector("#healthDot"),
  healthText: document.querySelector("#healthText"),
  modelBadge: document.querySelector("#modelBadge"),
  deviceBadge: document.querySelector("#deviceBadge"),
  batchBadge: document.querySelector("#batchBadge"),
  progressRing: document.querySelector("#progressRing"),
  progressValue: document.querySelector("#progressValue"),
  jobTitle: document.querySelector("#jobTitle"),
  jobMessage: document.querySelector("#jobMessage"),
  phaseSteps: document.querySelector("#phaseSteps"),
  searchInput: document.querySelector("#searchInput"),
  artifactActions: document.querySelector("#artifactActions"),
  detailPane: document.querySelector("#detailPane"),
  toast: document.querySelector("#toast"),
};

init();

async function init() {
  el.apiKeyInput.value = state.apiKey;
  bindEvents();
  renderPhaseSteps(null);
  renderProgress();
  renderDetail();
  renderArtifactActions();
  await Promise.all([loadConfig(), checkHealth()]);
  await refreshMeetings();
  setInterval(tick, 1800);
}

function bindEvents() {
  el.audioFile.addEventListener("change", () => {
    const file = el.audioFile.files?.[0];
    el.audioFileName.textContent = file ? file.name : "Выберите аудио";
  });

  el.apiKeyInput.addEventListener("input", () => {
    setApiKey(el.apiKeyInput.value.trim());
  });

  el.uploadForm.addEventListener("submit", uploadMeeting);
  el.refreshButton.addEventListener("click", refreshMeetings);
  el.searchInput.addEventListener("input", renderDetail);

  ["dragenter", "dragover"].forEach((eventName) => {
    el.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      el.dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    el.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      el.dropZone.classList.remove("dragover");
    });
  });

  el.dropZone.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer?.files || [];
    if (!file) return;
    if (typeof DataTransfer === "function") {
      const transfer = new DataTransfer();
      transfer.items.add(file);
      el.audioFile.files = transfer.files;
    }
    el.audioFileName.textContent = file.name;
  });
}

async function tick() {
  if (state.isTicking) return;
  state.isTicking = true;
  try {
    await checkHealth();
    await refreshMeetings({ quiet: true });
    if (state.selectedId && isLiveStatus(state.selectedStatus?.status)) {
      await selectMeeting(state.selectedId, { quiet: true });
    }
  } finally {
    state.isTicking = false;
  }
}

async function loadConfig() {
  try {
    state.config = await fetchJson("/api/ui/config", { auth: false });
    el.modelBadge.textContent = state.config.model || "model";
    el.deviceBadge.textContent = `device: ${state.config.device || "cpu"}`;
    el.batchBadge.textContent = `batch: ${state.config.batch_size || 1}`;
    if (state.config.auth_required && !state.apiKey) {
      el.authDetails.open = true;
      showToast("Введите локальный API ключ для запуска и просмотра задач.", { key: "auth-required" });
    }
  } catch (error) {
    showToast(`Не удалось загрузить конфигурацию: ${error.message}`, { key: "config-error" });
  }
}

async function checkHealth() {
  try {
    await fetchJson("/healthz", { auth: false });
    el.healthDot.className = "dot dot-ok";
    el.healthText.textContent = "online";
  } catch {
    el.healthDot.className = "dot dot-bad";
    el.healthText.textContent = "offline";
  }
}

async function uploadMeeting(event) {
  event.preventDefault();
  const audio = el.audioFile.files?.[0];
  if (!audio) {
    showToast("Выберите аудиофайл.");
    return;
  }
  if (!ensureAuthReady()) return;

  state.isUploading = true;
  el.startButton.disabled = true;
  el.startButton.innerHTML = `${icon("loader")} Загружаем`;

  const form = new FormData();
  form.append("file", audio);
  form.append("language_code", "ru");
  form.append("processing_mode", "simple");

  const participantFiles = Array.from(el.participantFiles.files || []);
  participantFiles.forEach((file) => form.append("zoom_participant_files", file));

  const names = parseNames(el.participantNames.value);
  if (names.length) {
    form.append("participants", names.join(","));
    form.append(
      "zoom_participant_tracks_json",
      JSON.stringify(names.map((name) => ({ speaker_name: name }))),
    );
  }

  try {
    const created = await fetchJson("/meetings", {
      method: "POST",
      body: form,
    });
    state.selectedId = created.id;
    state.transcript = [];
    state.transcriptMarkdown = "";
    el.searchInput.value = "";
    showToast("Задача создана, распознавание запущено.");
    await refreshMeetings({ quiet: true });
    await selectMeeting(created.id);
  } catch (error) {
    showToast(error.message, { key: "upload-error" });
  } finally {
    state.isUploading = false;
    el.startButton.disabled = false;
    el.startButton.innerHTML = `${icon("play")} Запустить`;
  }
}

async function refreshMeetings(options = {}) {
  if (!ensureAuthReady(options)) {
    state.meetings = [];
    renderMeetings();
    return;
  }
  try {
    const data = await fetchJson("/meetings");
    state.meetings = Array.isArray(data.meetings) ? data.meetings : [];
    if (!state.selectedId && state.meetings.length) {
      state.selectedId = state.meetings[0].id;
      await selectMeeting(state.selectedId, { quiet: true });
    }
    renderMeetings();
  } catch (error) {
    if (!options.quiet) showToast(error.message, { key: `meetings-${error.message}` });
  }
}

async function selectMeeting(id, options = {}) {
  if (!id || !ensureAuthReady(options)) return;
  state.selectedId = id;
  try {
    state.selectedStatus = await fetchJson(`/meetings/${encodeURIComponent(id)}/status`);
    if (state.selectedStatus.status === "completed") {
      await loadArtifacts(id, options);
    } else {
      state.transcript = [];
      state.transcriptMarkdown = "";
    }
    renderMeetings();
    renderProgress();
    renderDetail();
    renderArtifactActions();
  } catch (error) {
    if (!options.quiet) showToast(error.message, { key: `status-${error.message}` });
  }
}

async function loadArtifacts(id, options = {}) {
  try {
    const [transcriptJson, transcriptMarkdown] = await Promise.allSettled([
      fetchJson(`/meetings/${encodeURIComponent(id)}/artifacts/transcript_json`),
      fetchText(`/meetings/${encodeURIComponent(id)}/artifacts/transcript_md`),
    ]);
    state.transcript =
      transcriptJson.status === "fulfilled" ? normalizeTranscript(transcriptJson.value) : [];
    state.transcriptMarkdown =
      transcriptMarkdown.status === "fulfilled" ? transcriptMarkdown.value : "";
    if (!state.transcript.length && !state.transcriptMarkdown) {
      throw new Error("Transcript artifacts are empty or unavailable");
    }
  } catch (error) {
    state.transcript = [];
    state.transcriptMarkdown = "";
    if (!options.quiet) {
      showToast(`Артефакты пока недоступны: ${error.message}`, { key: `artifacts-${id}` });
    }
  }
}

function renderMeetings() {
  if (!state.meetings.length) {
    el.meetingList.innerHTML = `<div class="empty-state"><p>История пока пустая.</p></div>`;
    return;
  }

  el.meetingList.innerHTML = state.meetings
    .map((meeting) => {
      const name = escapeHtml(meeting.metadata?.source_filename || meeting.id);
      const meta = meeting.result
        ? `${meeting.result.segments || 0} сегм. · ${formatDuration(meeting.result.elapsed_seconds)}`
        : phaseLabel(meeting.phase || meeting.status);
      return `
        <button class="meeting-item ${meeting.id === state.selectedId ? "active" : ""}" data-id="${escapeHtml(meeting.id)}">
          <span class="meeting-top">
            <span class="meeting-name">${name}</span>
            ${statusBadge(meeting.status)}
          </span>
          <span class="meeting-meta">${escapeHtml(meta)}</span>
        </button>
      `;
    })
    .join("");

  el.meetingList.querySelectorAll(".meeting-item").forEach((button) => {
    button.addEventListener("click", () => selectMeeting(button.dataset.id));
  });
}

function renderProgress() {
  const meeting = state.selectedStatus;
  if (!meeting) {
    el.jobTitle.textContent = "Задача не выбрана";
    el.jobMessage.textContent = "Загрузите аудио или выберите задачу из истории.";
    setProgress(0);
    renderPhaseSteps(null);
    return;
  }

  const fileName = meeting.metadata?.source_filename || meeting.id;
  const progress = clamp(Number(meeting.progress || statusProgress(meeting.status)), 0, 1);
  el.jobTitle.textContent = fileName;
  el.jobMessage.textContent = progressMessage(meeting);
  setProgress(progress);
  renderPhaseSteps(meeting.phase || meeting.status);
}

function renderPhaseSteps(activePhase) {
  const activeIndex = phaseIndex(activePhase);
  el.phaseSteps.innerHTML = phases
    .map(([key, label], index) => {
      const className = index < activeIndex ? "done" : index === activeIndex ? "active" : "";
      return `
        <div class="phase-step ${className}">
          <span><i class="phase-dot"></i>${label}</span>
          <span>${index < activeIndex ? "done" : index === activeIndex ? "now" : ""}</span>
        </div>
      `;
    })
    .join("");
}

function renderArtifactActions() {
  const meeting = state.selectedStatus;
  if (!meeting || meeting.status !== "completed") {
    el.artifactActions.innerHTML = "";
    return;
  }

  const artifacts = availableArtifacts(meeting);
  el.artifactActions.innerHTML = `
    <button class="ghost-button" type="button" data-copy-transcript>
      ${icon("copy")} Копировать
    </button>
    ${artifacts
      .map(
        (kind) => `
          <button class="ghost-button" type="button" data-download="${escapeHtml(kind)}">
            ${icon("download")} ${escapeHtml(artifactLabels[kind] || kind)}
          </button>
        `,
      )
      .join("")}
  `;

  const copyButton = el.artifactActions.querySelector("[data-copy-transcript]");
  copyButton?.addEventListener("click", copyTranscript);
  el.artifactActions.querySelectorAll("[data-download]").forEach((button) => {
    button.addEventListener("click", () => downloadArtifact(button.dataset.download));
  });
}

function renderDetail() {
  const meeting = state.selectedStatus;
  if (!meeting) {
    el.detailPane.innerHTML = `
      <div class="empty-state">
        ${icon("activity")}
        <h2>Готов к работе</h2>
        <p>После запуска здесь появится прогресс, сегменты речи и артефакты.</p>
      </div>
    `;
    return;
  }

  if (meeting.status === "failed") {
    el.detailPane.innerHTML = `
      <div class="error-box">${escapeHtml(meeting.error || "Ошибка обработки")}</div>
    `;
    return;
  }

  if (meeting.status !== "completed") {
    el.detailPane.innerHTML = `
      <div class="empty-state">
        ${icon("loader")}
        <h2>${escapeHtml(phaseLabel(meeting.phase || meeting.status))}</h2>
        <p>${escapeHtml(progressMessage(meeting))}</p>
      </div>
    `;
    return;
  }

  const query = el.searchInput.value.trim().toLowerCase();
  const segments = state.transcript.filter((segment) => {
    if (!query) return true;
    return [segment.text, segment.speaker, segment.track]
      .filter(Boolean)
      .join(" ")
      .toLowerCase()
      .includes(query);
  });

  if (segments.length) {
    el.detailPane.innerHTML = `
      <div class="detail-stack">
        ${renderReportAlert(meeting)}
        <div class="transcript">
          ${segments.map(renderSegment).join("")}
        </div>
      </div>
    `;
    return;
  }

  if (state.transcriptMarkdown) {
    el.detailPane.innerHTML = `
      <div class="detail-stack">
        ${renderReportAlert(meeting)}
        <div class="markdown-view">${escapeHtml(state.transcriptMarkdown)}</div>
      </div>
    `;
    return;
  }

  el.detailPane.innerHTML = `
    <div class="empty-state">
      ${icon("file")}
      <h2>Транскрипт пуст</h2>
      <p>Артефакты созданы, но сегменты речи не найдены.</p>
    </div>
  `;
}

function renderReportAlert(meeting) {
  if (!meeting || !meeting.report_health_status || meeting.report_health_status === "ok") return "";
  const alerts = Array.isArray(meeting.report_alerts) ? meeting.report_alerts.filter(Boolean) : [];
  const details = alerts.length
    ? `<ul>${alerts.map((alert) => `<li>${escapeHtml(alert)}</li>`).join("")}</ul>`
    : "<p>Отчёт создан в degraded-состоянии. Скачайте Report Health для деталей.</p>";
  return `
    <div class="report-alert" role="alert">
      ${icon("alert")}
      <div>
        <strong>Отчёт требует проверки: ${escapeHtml(meeting.report_health_status)}</strong>
        ${details}
      </div>
    </div>
  `;
}

function renderSegment(segment) {
  return `
    <article class="segment">
      <header class="segment-head">
        <span class="speaker">${escapeHtml(segment.speaker || "Speaker")}</span>
        <span class="segment-meta">${formatTime(segment.start)} - ${formatTime(segment.end)}</span>
      </header>
      <p class="segment-text">${escapeHtml(segment.text || "")}</p>
    </article>
  `;
}

async function downloadArtifact(kind) {
  if (!state.selectedId || !kind || !ensureAuthReady()) return;
  let url = "";
  try {
    const response = await fetchWithAuth(
      `/meetings/${encodeURIComponent(state.selectedId)}/artifacts/${kind}`,
    );
    if (!response.ok) throw new Error(await errorMessage(response));
    const blob = await response.blob();
    url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = artifactFilename(kind);
    document.body.appendChild(link);
    link.click();
    link.remove();
  } catch (error) {
    showToast(error.message, { key: `download-${kind}-${error.message}` });
  } finally {
    if (url) URL.revokeObjectURL(url);
  }
}

async function copyTranscript() {
  const text =
    state.transcriptMarkdown ||
    state.transcript
      .map((segment) => `[${formatTime(segment.start)}] ${segment.speaker || "Speaker"}: ${segment.text || ""}`)
      .join("\n");
  if (!text) {
    showToast("Транскрипт пока пуст.");
    return;
  }
  try {
    if (!navigator.clipboard?.writeText) throw new Error("Clipboard API недоступен.");
    await navigator.clipboard.writeText(text);
    showToast("Транскрипт скопирован.");
  } catch (error) {
    showToast(error.message || "Не удалось скопировать транскрипт.", { key: "copy-error" });
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetchWithAuth(url, options);
  if (!response.ok) throw new Error(await errorMessage(response));
  if (response.status === 204) return {};
  return response.json();
}

async function fetchText(url, options = {}) {
  const response = await fetchWithAuth(url, options);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.text();
}

function fetchWithAuth(url, options = {}) {
  const headers = new Headers(options.headers || {});
  const includeAuth = options.auth !== false;
  const apiKey = includeAuth ? syncApiKeyFromInput() : state.apiKey;
  if (includeAuth && apiKey) headers.set("Authorization", `Bearer ${apiKey}`);
  return fetch(url, { ...options, headers });
}

async function errorMessage(response) {
  if (response.status === 401) {
    el.authDetails.open = true;
    return "Нужен правильный локальный API ключ.";
  }
  try {
    const data = await response.json();
    return data.detail || response.statusText;
  } catch {
    return response.statusText || "Ошибка запроса";
  }
}

function setProgress(value) {
  const pct = Math.round(value * 100);
  el.progressRing.style.setProperty("--progress", `${pct * 3.6}deg`);
  el.progressValue.textContent = `${pct}%`;
}

function statusProgress(status) {
  if (status === "uploaded") return 0.05;
  if (status === "processing") return 0.18;
  if (status === "completed") return 1;
  if (status === "failed") return 1;
  return 0;
}

function progressMessage(meeting) {
  if (meeting.status === "completed") {
    const result = meeting.result || {};
    const reportStatus =
      meeting.report_health_status && meeting.report_health_status !== "ok"
        ? ` · отчёт: ${meeting.report_health_status}`
        : "";
    return `${result.segments || 0} сегментов · ${result.tracks || 0} треков · ${formatDuration(result.elapsed_seconds)}${reportStatus}`;
  }
  if (meeting.status === "failed") return meeting.error || "Ошибка обработки.";
  if (meeting.phase === "transcribing" && Number.isFinite(meeting.segments_total)) {
    return `${meeting.segments_done || 0} из ${meeting.segments_total || 0} сегментов`;
  }
  return meeting.message || phaseLabel(meeting.phase || meeting.status);
}

function phaseIndex(phase) {
  if (phase === "preparing" || phase === "queued") return 0;
  const index = phases.findIndex(([key]) => key === phase);
  if (index >= 0) return index;
  return phase === "failed" ? phases.length - 1 : 0;
}

function phaseLabel(phase) {
  const labels = {
    uploaded: "Загружено",
    queued: "В очереди",
    preparing: "Подготовка",
    segmenting: "Сегментация",
    transcribing: "Распознавание",
    writing_artifacts: "Запись артефактов",
    completed: "Готово",
    failed: "Ошибка",
    processing: "Обработка",
  };
  return labels[phase] || phase || "Ожидание";
}

function statusBadge(status) {
  const iconName = status === "completed" ? "check" : status === "failed" ? "alert" : "loader";
  return `<span class="badge ${escapeHtml(cssToken(status))}">${icon(iconName)} ${escapeHtml(phaseLabel(status))}</span>`;
}

function isLiveStatus(status) {
  return status === "uploaded" || status === "processing";
}

function parseNames(raw) {
  return raw
    .split(/\n|,/)
    .map((name) => name.trim())
    .filter(Boolean);
}

function artifactFilename(kind) {
  const filenames = {
    transcript_md: "transcript.md",
    transcript_json: "transcript.json",
    summary_json: "summary.json",
    report_md: "report.md",
    report_json: "report.json",
    report_html: "report.html",
    report_health: "report_health.json",
    coverage_json: "coverage.json",
    report_typ: "report.typ",
    report_pdf: "report.pdf",
    segments_tsv: "segments.tsv",
    speaker_track_map: "speaker_track_map.tsv",
  };
  return filenames[kind] || kind;
}

function availableArtifacts(meeting) {
  if (Array.isArray(meeting.artifacts) && meeting.artifacts.length) {
    return meeting.artifacts.filter((kind) => typeof kind === "string" && kind);
  }
  return Object.keys(artifactLabels);
}

function normalizeTranscript(value) {
  if (Array.isArray(value)) return value.filter(isSegmentLike);
  if (Array.isArray(value?.segments)) return value.segments.filter(isSegmentLike);
  if (Array.isArray(value?.transcript)) return value.transcript.filter(isSegmentLike);
  return [];
}

function isSegmentLike(segment) {
  return segment && typeof segment === "object";
}

function ensureAuthReady(options = {}) {
  const apiKey = syncApiKeyFromInput();
  if (!state.config?.auth_required || apiKey) return true;
  el.authDetails.open = true;
  if (!options.quiet) {
    showToast("Введите локальный API ключ для запуска и просмотра задач.", { key: "auth-required" });
  }
  return false;
}

function syncApiKeyFromInput() {
  const currentValue = el.apiKeyInput.value.trim();
  if (currentValue) {
    setApiKey(currentValue);
  } else if (state.apiKey) {
    el.apiKeyInput.value = state.apiKey;
  }
  return state.apiKey;
}

function setApiKey(value) {
  if (value === state.apiKey) return;
  state.apiKey = value;
  writeStoredApiKey(state.apiKey);
}

function readStoredApiKey() {
  try {
    return localStorage.getItem("gigaam-ui-api-key") || "";
  } catch {
    return "";
  }
}

function writeStoredApiKey(value) {
  try {
    if (value) {
      localStorage.setItem("gigaam-ui-api-key", value);
    } else {
      localStorage.removeItem("gigaam-ui-api-key");
    }
  } catch {
    // Ignore storage failures; the in-memory key still works for this tab.
  }
}

function formatDuration(seconds) {
  if (!Number.isFinite(Number(seconds))) return "0с";
  const total = Math.max(0, Math.round(Number(seconds)));
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  return minutes ? `${minutes}м ${rest}с` : `${rest}с`;
}

function formatTime(value) {
  const seconds = Math.max(0, Number(value) || 0);
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.round((seconds % 1) * 10);
  const base = [h, m, s].map((part) => String(part).padStart(2, "0")).join(":");
  return `${base}.${ms}`;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function cssToken(value) {
  return String(value || "unknown").replace(/[^a-z0-9_-]/gi, "-");
}

function icon(name) {
  return `<svg aria-hidden="true"><use href="#icon-${name}"></use></svg>`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

let toastTimer = null;
let lastToastKey = "";
let lastToastAt = 0;

function showToast(message, options = {}) {
  const key = options.key || message;
  const now = Date.now();
  if (key === lastToastKey && now - lastToastAt < 5000) return;
  lastToastKey = key;
  lastToastAt = now;
  el.toast.textContent = message;
  el.toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.toast.classList.remove("visible"), 3600);
}
