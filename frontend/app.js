const form = document.getElementById("pitchForm");
const evaluateBtn = document.getElementById("evaluateBtn");
const refreshBtn = document.getElementById("refreshBtn");
const statusText = document.getElementById("statusText");
const modeBadge = document.getElementById("modeBadge");
const healthBadge = document.getElementById("healthBadge");
const apiBaseUrlInput = document.getElementById("apiBaseUrl");
const evaluationPlaceholder = document.getElementById("evaluationPlaceholder");
const evaluationResults = document.getElementById("evaluationResults");
const summaryCard = document.getElementById("summaryCard");
const strengthsList = document.getElementById("strengthsList");
const weaknessesList = document.getElementById("weaknessesList");
const suggestionsList = document.getElementById("suggestionsList");
const notesList = document.getElementById("notesList");
const rawJson = document.getElementById("rawJson");
const overallKpi = document.getElementById("overallKpi");
const confidenceKpi = document.getElementById("confidenceKpi");
const bandKpi = document.getElementById("bandKpi");
const pitchPreview = document.getElementById("pitchPreview");

const fields = {
  videoFile: document.getElementById("videoFile"),
  title: document.getElementById("title"),
  transcript: document.getElementById("transcript"),
  languageHint: document.getElementById("languageHint"),
  slideText: document.getElementById("slideText"),
  founderName: document.getElementById("founderName"),
  startupName: document.getElementById("startupName"),
  sector: document.getElementById("sector"),
  stage: document.getElementById("stage"),
};

const API_BASE_STORAGE_KEY = "spe_api_base_url";
const DEFAULT_API_BASE_URL =
  globalThis.SPE_API_BASE_URL || "http://127.0.0.1:8000";

let currentScoringMode = "unknown";

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function normalizeBaseUrl(value) {
  const raw = String(value || "")
    .trim()
    .replace(/\/+$/, "");

  if (!raw) {
    return "";
  }

  if (/^https?:\/\//i.test(raw)) {
    return raw;
  }

  return `https://${raw}`;
}

function getApiBaseUrl() {
  const stored = localStorage.getItem(API_BASE_STORAGE_KEY);
  return normalizeBaseUrl(
    stored || apiBaseUrlInput.value || DEFAULT_API_BASE_URL,
  );
}

function setApiBaseUrl(value) {
  const normalized = normalizeBaseUrl(value);
  apiBaseUrlInput.value = normalized;
  if (normalized) {
    localStorage.setItem(API_BASE_STORAGE_KEY, normalized);
  }
  return normalized;
}

function apiPath(path) {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${getApiBaseUrl()}${normalizedPath}`;
}

function listFromTextarea(value) {
  return String(value || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

function getSelectedVideoFile() {
  return fields.videoFile?.files?.[0] || null;
}

function renderPitchPreview(payload) {
  const videoFile = getSelectedVideoFile();
  pitchPreview.innerHTML = `
    <h3>${escapeHtml(payload.title)}</h3>
    <p><strong>Video:</strong> ${escapeHtml(payload.video?.file_name || videoFile?.name || "Required")}</p>
    <p><strong>Founder:</strong> ${escapeHtml(payload.user_details.founder_name || "N/A")}</p>
    <p><strong>Startup:</strong> ${escapeHtml(payload.user_details.startup_name || "N/A")}</p>
    <p><strong>Sector:</strong> ${escapeHtml(payload.user_details.sector || "N/A")}</p>
    <p><strong>Stage:</strong> ${escapeHtml(payload.user_details.stage || "N/A")}</p>
  `;
}

function renderList(container, values, emptyLabel) {
  if (!values || values.length === 0) {
    container.innerHTML = `<p class="small">${escapeHtml(emptyLabel)}</p>`;
    return;
  }

  container.innerHTML = values
    .map((value) => `<div class="list-item">${escapeHtml(value)}</div>`)
    .join("");
}

function renderSummary(summary) {
  const overall = Number(summary.overall_score || 0);
  const confidence = Number(summary.confidence_score || 0);
  const band = summary.investment_band || "-";
  const bandClass = `band-${band}`;

  summaryCard.innerHTML = `
    <h3>Overall Summary</h3>
    <div class="summary-score">
      <span>Score</span>
      <strong>${overall.toFixed(2)}</strong>
      <span>/ 10</span>
    </div>
    <p><strong>Language Detected:</strong> ${escapeHtml(summary.language_detected || "-")}</p>
    <p><strong>Scoring Mode:</strong> ${escapeHtml(summary.scoring_mode || currentScoringMode)}</p>
    <p><strong>Processing Option:</strong> ${escapeHtml(summary.processing_option || "-")}</p>
    <span class="band-pill ${bandClass}">${escapeHtml(band)}</span>
  `;

  overallKpi.textContent = `${overall.toFixed(2)} / 10`;
  confidenceKpi.textContent = `${confidence.toFixed(2)} / 10`;
  bandKpi.textContent = band;
}

function renderDashboard(result) {
  const summary = result.summary || {};
  renderSummary(summary);
  renderList(strengthsList, summary.strengths || [], "No strengths returned.");
  renderList(
    weaknessesList,
    summary.weaknesses || [],
    "No weaknesses returned.",
  );
  renderList(
    suggestionsList,
    summary.suggestions || [],
    "No suggestions returned.",
  );
  renderList(notesList, summary.processing_notes || [], "No notes returned.");
  rawJson.textContent = JSON.stringify(result, null, 2);
}

function setLoading(isLoading) {
  evaluateBtn.disabled = isLoading;
  refreshBtn.disabled = isLoading;
  statusText.textContent = isLoading
    ? "Calling backend..."
    : statusText.textContent;
}

function setHealthState(text, className) {
  healthBadge.textContent = text;
  healthBadge.classList.remove("is-heuristic", "is-neural", "is-error");
  if (className) {
    healthBadge.classList.add(className);
  }
}

function normalizeScoringMode(value) {
  const mode = String(value || "")
    .trim()
    .toLowerCase();
  if (mode.includes("neural")) {
    return "neural-network";
  }
  if (mode.includes("heuristic")) {
    return "heuristic";
  }
  return mode || "unknown";
}

function updateModeBadge(mode) {
  currentScoringMode = normalizeScoringMode(mode);
  modeBadge.textContent = `Mode: ${currentScoringMode}`;
  modeBadge.classList.remove("is-heuristic", "is-neural", "is-error");
  if (currentScoringMode === "neural-network") {
    modeBadge.classList.add("is-neural");
  } else if (currentScoringMode === "heuristic") {
    modeBadge.classList.add("is-heuristic");
  }
}

function toPayload() {
  const slidePoints = listFromTextarea(fields.slideText.value);
  const videoFile = getSelectedVideoFile();
  const titleValue = fields.title.value.trim();
  const titleFromVideo = videoFile?.name
    ? videoFile.name.replace(/\.[^.]+$/, "")
    : "Untitled Pitch";

  return {
    title: titleValue || titleFromVideo,
    transcript: fields.transcript.value.trim(),
    language_hint: fields.languageHint.value.trim() || "en",
    slide_text: slidePoints,
    presenter_profile: {
      founder_name: fields.founderName.value.trim(),
      startup_name: fields.startupName.value.trim(),
      sector: fields.sector.value.trim(),
      stage: fields.stage.value.trim(),
    },
    slides: slidePoints.map((point, index) => ({
      title: `Slide ${index + 1}`,
      content: point,
    })),
    video: videoFile
      ? {
          file_name: videoFile.name,
          file_format: (videoFile.name.split(".").pop() || "mp4").toLowerCase(),
          duration_sec: 60,
          transcript_text: fields.transcript.value.trim(),
        }
      : null,
    user_details: {
      founder_name: fields.founderName.value.trim(),
      startup_name: fields.startupName.value.trim(),
      sector: fields.sector.value.trim(),
      stage: fields.stage.value.trim(),
    },
  };
}

function buildUploadFormData(payload) {
  const videoFile = getSelectedVideoFile();
  if (!videoFile) {
    throw new Error("Please upload a video file before evaluating.");
  }

  const formData = new FormData();
  formData.append("video", videoFile);
  formData.append(
    "title",
    payload.title || videoFile.name.replace(/\.[^.]+$/, ""),
  );
  formData.append("transcript", payload.transcript || "");
  formData.append("language_hint", payload.language_hint || "en");
  formData.append("slide_text", (payload.slide_text || []).join("\n"));
  formData.append("founder_name", payload.user_details?.founder_name || "");
  formData.append("startup_name", payload.user_details?.startup_name || "");
  formData.append("sector", payload.user_details?.sector || "");
  formData.append("stage", payload.user_details?.stage || "");
  return formData;
}

async function refreshBackendStatus() {
  const baseUrl = setApiBaseUrl(getApiBaseUrl());

  if (!baseUrl) {
    setHealthState("Backend: not set", "is-error");
    updateModeBadge("unknown");
    statusText.textContent = "Set your ngrok URL first";
    return;
  }

  try {
    const [healthResponse, modeResponse] = await Promise.all([
      fetch(`${baseUrl}/health`),
      fetch(`${baseUrl}/scoring-mode`),
    ]);

    if (!healthResponse.ok) {
      throw new Error(`Health check failed (${healthResponse.status})`);
    }

    const health = await healthResponse.json();
    const mode = modeResponse.ok ? await modeResponse.json() : {};

    updateModeBadge(mode.scoring_mode || health.scoring_mode);
    setHealthState(`Backend: ${health.status || "ok"}`, "is-neural");
    statusText.textContent = `Connected to ${baseUrl}`;
  } catch (error) {
    console.error("Backend status check failed:", error);
    setHealthState("Backend: offline", "is-error");
    updateModeBadge("unknown");
    statusText.textContent = "Unable to reach backend";
  }
}

async function evaluatePitch(payload) {
  const baseUrl = setApiBaseUrl(getApiBaseUrl());
  if (!baseUrl) {
    evaluationPlaceholder.classList.remove("hidden");
    evaluationPlaceholder.textContent = "Set the backend API URL first.";
    evaluationResults.classList.add("hidden");
    statusText.textContent = "Backend URL missing";
    return;
  }
  setLoading(true);

  try {
    const formData = buildUploadFormData(payload);
    const response = await fetch(`${baseUrl}/evaluate/upload`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const maybeError = await response.text();
      throw new Error(
        `Request failed (${response.status}): ${maybeError.slice(0, 220)}`,
      );
    }

    const result = await response.json();
    renderDashboard(result);
    evaluationPlaceholder.classList.add("hidden");
    evaluationResults.classList.remove("hidden");
    statusText.textContent = "Evaluation complete.";
  } catch (error) {
    evaluationPlaceholder.classList.remove("hidden");
    evaluationPlaceholder.textContent = error.message || "Evaluation failed.";
    evaluationResults.classList.add("hidden");
    statusText.textContent = "Evaluation failed.";
  } finally {
    setLoading(false);
  }
}

Object.values(fields)
  .filter(Boolean)
  .forEach((field) => {
    field.addEventListener("input", () => renderPitchPreview(toPayload()));
  });

fields.videoFile?.addEventListener("change", () =>
  renderPitchPreview(toPayload()),
);

apiBaseUrlInput.value = normalizeBaseUrl(
  localStorage.getItem(API_BASE_STORAGE_KEY) || DEFAULT_API_BASE_URL,
);
apiBaseUrlInput.addEventListener("change", () => {
  setApiBaseUrl(apiBaseUrlInput.value);
  refreshBackendStatus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = toPayload();
  renderPitchPreview(payload);
  await evaluatePitch(payload);
});

refreshBtn.addEventListener("click", refreshBackendStatus);

renderPitchPreview(toPayload());
await refreshBackendStatus();
