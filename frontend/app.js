const form = document.getElementById("pitchForm");
const clearBtn = document.getElementById("clearBtn");
const evaluateBtn = document.getElementById("evaluateBtn");
const statusText = document.getElementById("statusText");
const pitchPreview = document.getElementById("pitchPreview");
const modeBadge = document.getElementById("modeBadge");

// New element references for updated UI
const videoSelect = document.getElementById("videoSelect");
const videoContainer = document.getElementById("videoContainer");
const pitchVideo = document.getElementById("pitchVideo");
const videoTitle = document.getElementById("videoTitle");
const videoDuration = document.getElementById("videoDuration");
const videoRatingPanel = document.getElementById("videoRatingPanel");
const finalScore = document.getElementById("finalScore");
const videoRatingBand = document.getElementById("videoRatingBand");
const videoRatingText = document.getElementById("videoRatingText");

const evaluationPlaceholder = document.getElementById("evaluationPlaceholder");
const evaluationResults = document.getElementById("evaluationResults");
const summaryCard = document.getElementById("summaryCard");
const quantScores = document.getElementById("quantScores");
const modalityWeights = document.getElementById("modalityWeights");
const riskDistribution = document.getElementById("riskDistribution");
const guidanceList = document.getElementById("guidanceList");
const chunkReports = document.getElementById("chunkReports");
const rawJson = document.getElementById("rawJson");
const overallKpi = document.getElementById("overallKpi");
const confidenceKpi = document.getElementById("confidenceKpi");
const bandKpi = document.getElementById("bandKpi");
const outputPanel = document.querySelector(".output-panel");

let selectedVideoFileName = "pitch.mp4";
let selectedVideoDurationSec = 120;
let latestRating = null;
let currentScoringMode = "unknown";

const fields = {
  title: document.getElementById("title"),
  slideText: document.getElementById("slideText"),
  founderName: document.getElementById("founderName"),
  startupName: document.getElementById("startupName"),
  sector: document.getElementById("sector"),
  stage: document.getElementById("stage"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function listFromTextarea(value) {
  return value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

function toPayload() {
  const slidePoints = listFromTextarea(fields.slideText.value);
  const inferredDuration = Number.isFinite(Number(pitchVideo.duration))
    ? Math.max(5, Math.round(Number(pitchVideo.duration)))
    : selectedVideoDurationSec;

  return {
    title: fields.title.value.trim() || "Untitled Pitch",
    transcript: "",
    language_hint: "en-ta",
    presenter_profile: { experience: "Unknown" },
    slide_text: slidePoints,
    video: {
      file_name: selectedVideoFileName,
      file_format: "mp4",
      duration_sec: inferredDuration,
      transcript_text: "",
    },
    slides: slidePoints.map((point, idx) => ({
      title: `Slide ${idx + 1}`,
      content: point,
    })),
    user_details: {
      founder_name: fields.founderName.value.trim(),
      startup_name:
        fields.startupName.value.trim() || fields.title.value.trim(),
      sector: fields.sector.value.trim(),
      stage: fields.stage.value.trim(),
    },
  };
}

function renderPitchPreview(payload) {
  pitchPreview.innerHTML = `
    <h3>${escapeHtml(payload.title)}</h3>
    <p><strong>Founder:</strong> ${escapeHtml(payload.user_details.founder_name || "N/A")}</p>
    <p><strong>Sector:</strong> ${escapeHtml(payload.user_details.sector || "N/A")} | <strong>Stage:</strong> ${escapeHtml(payload.user_details.stage || "N/A")}</p>
    <p><strong>Slides:</strong> ${payload.slides.length}</p>
  `;
}

function rowHtml(name, value) {
  return `<div class="row-item"><span>${escapeHtml(name)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function renderBars(container, points, maxValue, kind) {
  container.innerHTML = "";
  if (!points || points.length === 0) {
    container.innerHTML = '<p class="small">No data.</p>';
    return;
  }

  container.innerHTML = points
    .map((point) => {
      const rawValue = Number(point.value || 0);
      const safeValue = Number.isFinite(rawValue) ? rawValue : 0;
      const percent = Math.max(0, Math.min(100, (safeValue / maxValue) * 100));
      const fillClass = kind === "modality" ? "bar-fill modality" : "bar-fill";
      const displayValue =
        kind === "modality"
          ? `${(safeValue * 100).toFixed(1)}%`
          : safeValue.toFixed(2);
      return `
        <div class="bar-row">
          <div>
            <div class="bar-label">${escapeHtml(point.label)}</div>
            <div class="bar-track"><div class="${fillClass}" style="width:${percent.toFixed(1)}%"></div></div>
          </div>
          <div class="bar-value">${displayValue}</div>
        </div>
      `;
    })
    .join("");
}

function renderGuidance(summary) {
  guidanceList.innerHTML = `
    <div class="guidance-block">
      <p><strong>Strengths:</strong> ${escapeHtml((summary.strengths || []).join(", ") || "-")}</p>
      <p><strong>Weaknesses:</strong> ${escapeHtml((summary.weaknesses || []).join(", ") || "-")}</p>
      <p><strong>Suggestions:</strong> ${escapeHtml((summary.suggestions || []).join(", ") || "-")}</p>
    </div>
  `;
}

function renderSummary(summary) {
  const bandClass = `band-${summary.investment_band}`;
  const overall = Number(summary.overall_score || 0);
  const confidence = Number(summary.confidence_score || 0);
  const scoringMode = currentScoringMode || "unknown";

  summaryCard.innerHTML = `
    <h3>Overall Summary</h3>
    <div class="summary-score">
      <span>Score</span>
      <strong>${overall.toFixed(2)}</strong>
      <span>/ 10</span>
    </div>
    <p><strong>Language Detected:</strong> ${escapeHtml(summary.language_detected)}</p>
    <p><strong>Scoring Mode:</strong> ${escapeHtml(scoringMode)}</p>
    <p><strong>Processing Option:</strong> ${escapeHtml(summary.processing_option || "unknown")}</p>
    <p><strong>Runtime:</strong> ${escapeHtml((summary.processing_notes || []).join(" | ") || "-")}</p>
    <span class="band-pill ${bandClass}">${escapeHtml(summary.investment_band)}</span>
  `;

  overallKpi.textContent = `${overall.toFixed(2)} / 10`;
  confidenceKpi.textContent = `${confidence.toFixed(2)} / 10`;
  bandKpi.textContent = summary.investment_band || "-";
}

function renderRisks(points) {
  if (!points || points.length === 0) {
    riskDistribution.innerHTML =
      '<p class="small">No explicit risk flags detected.</p>';
    return;
  }
  riskDistribution.innerHTML = points
    .map((point) => rowHtml(point.label, String(point.value)))
    .join("");
}

function renderChunks(chunks) {
  chunkReports.innerHTML = "";
  if (!chunks || chunks.length === 0) {
    chunkReports.innerHTML = '<p class="small">No chunk reports.</p>';
    return;
  }

  chunkReports.innerHTML = chunks
    .map((chunk) => {
      const risks = (chunk.risk_flags || []).length
        ? chunk.risk_flags
            .map(
              (risk) =>
                `<span class="chip chip-risk">${escapeHtml(risk)}</span>`,
            )
            .join("")
        : `<span class="chip">No risk flags</span>`;

      const textMetricRows = (chunk.text_metrics || [])
        .map((metric) =>
          rowHtml(metric.name, Number(metric.score || 0).toFixed(2)),
        )
        .join("");

      const avMetricRows = (chunk.av_metrics || [])
        .map((metric) =>
          rowHtml(metric.name, Number(metric.score || 0).toFixed(2)),
        )
        .join("");

      return `
        <article class="chunk-card">
          <div class="chunk-title">
            <h4>Chunk #${escapeHtml(chunk.chunk_id)}</h4>
            <span class="chunk-meta">${escapeHtml(chunk.start_sec)}s-${escapeHtml(chunk.end_sec)}s</span>
          </div>
          <p><strong>Aggregate:</strong> ${Number(chunk.aggregate_score || 0).toFixed(2)} | <strong>Attention:</strong> T ${Number(chunk.attention?.text || 0).toFixed(2)} / V ${Number(chunk.attention?.visual || 0).toFixed(2)} / A ${Number(chunk.attention?.audio || 0).toFixed(2)}</p>
          <div>${risks}</div>
          <details>
            <summary>Expand chunk metrics</summary>
            <div class="chunk-subgrid">
              <div class="list-card">${textMetricRows || '<p class="small">No text metrics.</p>'}</div>
              <div class="list-card">${avMetricRows || '<p class="small">No AV metrics.</p>'}</div>
            </div>
          </details>
        </article>
      `;
    })
    .join("");
}

function showError(message) {
  evaluationPlaceholder.classList.remove("hidden");
  evaluationPlaceholder.textContent = message;
  evaluationResults.classList.add("hidden");
}

function setLoading(isLoading) {
  evaluateBtn.disabled = isLoading;
  outputPanel.classList.toggle("is-loading", isLoading);
  statusText.textContent = isLoading
    ? "Evaluating your pitch..."
    : statusText.textContent;
}

async function evaluatePitch(payload) {
  setLoading(true);

  try {
    const response = await fetch("/evaluate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const maybeError = await response.text();
      throw new Error(
        `Request failed (${response.status}): ${maybeError.slice(0, 220)}`,
      );
    }

    const result = await response.json();

    renderSummary(result.summary || {});
    renderBars(
      quantScores,
      result.dashboard?.quantitative_scores || [],
      10,
      "score",
    );
    renderBars(
      modalityWeights,
      result.dashboard?.modality_weights || [],
      1,
      "modality",
    );
    renderRisks(result.dashboard?.risk_distribution || []);
    renderGuidance(result.summary || {});
    renderChunks(result.chunk_reports || []);
    rawJson.textContent = JSON.stringify(result, null, 2);

    const overall = Number(result.summary?.overall_score || 0);
    latestRating = {
      score: overall.toFixed(2),
      band: result.summary?.investment_band || "-",
    };
    showLatestRatingPanel();

    evaluationPlaceholder.classList.add("hidden");
    evaluationResults.classList.remove("hidden");
    statusText.textContent = "Evaluation complete.";
    evaluationResults.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    showError(
      err.message || "Something went wrong while evaluating the pitch.",
    );
    statusText.textContent = "Evaluation failed.";
  } finally {
    setLoading(false);
  }
}

function clearForm() {
  globalThis.location.reload();
}

function formatDuration(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function hideVideoRatingPanel() {
  videoRatingPanel.classList.add("hidden");
}

function showVideoRatingPanel(score, band) {
  finalScore.textContent = score;
  videoRatingBand.textContent = band || "Evaluating...";
  videoRatingText.textContent = "Rating based on video analysis";
  videoRatingPanel.classList.remove("hidden");
}

function showLatestRatingPanel() {
  if (!latestRating) {
    return;
  }
  showVideoRatingPanel(latestRating.score, latestRating.band);
}

function handleVideoEnded() {
  const bufferTimeMs = 2000;
  setTimeout(() => {
    showLatestRatingPanel();
  }, bufferTimeMs);
}

function handleVideoSelect(event) {
  const videoName = event.target.value;

  if (!videoName) {
    selectedVideoFileName = "pitch.mp4";
    selectedVideoDurationSec = 120;
    pitchVideo.removeAttribute("src");
    pitchVideo.load();
    videoContainer.classList.add("hidden");
    hideVideoRatingPanel();
    return;
  }

  selectedVideoFileName = videoName;
  selectedVideoDurationSec = 120;

  pitchVideo.src = `/videos/${encodeURIComponent(videoName)}`;
  videoTitle.textContent = videoName;
  videoContainer.classList.remove("hidden");
  hideVideoRatingPanel();

  if (!fields.title.value) {
    fields.title.value = videoName.replace(/\.[^.]+$/, "");
    renderPitchPreview(toPayload());
  }

  pitchVideo.play().catch((err) => {
    console.error("Failed to play video:", err);
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = toPayload();
  renderPitchPreview(payload);
  await evaluatePitch(payload);
});

clearBtn.addEventListener("click", () => {
  clearForm();
});

function normalizeScoringMode(value) {
  const mode = String(value || "")
    .trim()
    .toLowerCase();
  if (mode.includes("heuristic")) {
    return "heuristic";
  }
  if (mode.includes("neural")) {
    return "neural-network";
  }
  return mode || "unknown";
}

function updateScoringModeBadge(mode) {
  if (!modeBadge) {
    return;
  }
  currentScoringMode = normalizeScoringMode(mode);
  modeBadge.textContent = `Scoring Mode: ${currentScoringMode}`;
  modeBadge.classList.toggle(
    "is-heuristic",
    currentScoringMode === "heuristic",
  );
  modeBadge.classList.toggle(
    "is-neural",
    currentScoringMode === "neural-network",
  );
}

async function loadScoringMode() {
  try {
    const response = await fetch("/scoring-mode");
    if (!response.ok) {
      throw new Error(
        `Scoring mode endpoint failed with status ${response.status}`,
      );
    }
    const data = await response.json();
    updateScoringModeBadge(data.scoring_mode);
  } catch (err) {
    console.error("Failed to load scoring mode:", err);
    updateScoringModeBadge("unknown");
  }
}

async function loadVideoList() {
  try {
    const response = await fetch("/videos");
    const data = await response.json();

    const selectedValue = videoSelect.value;
    videoSelect.innerHTML = '<option value="">Select a video...</option>';

    data.videos.forEach((video) => {
      const option = document.createElement("option");
      option.value = video;
      option.textContent = video;
      videoSelect.appendChild(option);
    });

    if (selectedValue && data.videos.includes(selectedValue)) {
      videoSelect.value = selectedValue;
    }
  } catch (err) {
    console.error("Failed to load videos:", err);
  }
}

Object.values(fields)
  .filter(Boolean)
  .forEach((field) => {
    field.addEventListener("input", () => renderPitchPreview(toPayload()));
  });

videoSelect.addEventListener("change", handleVideoSelect);

pitchVideo.addEventListener("loadedmetadata", () => {
  const duration = pitchVideo.duration;
  if (Number.isFinite(duration)) {
    selectedVideoDurationSec = Math.max(5, Math.round(duration));
  }
  videoDuration.textContent = `Duration: ${formatDuration(duration)}`;
});

pitchVideo.addEventListener("ended", () => {
  handleVideoEnded();
});

await loadScoringMode();
await loadVideoList();

renderPitchPreview(toPayload());

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
    const uploadResponse = await fetch(`${baseUrl}/evaluate/upload`, {
      method: "POST",
      body: formData,
    });

    let response = uploadResponse;
    if (
      (uploadResponse.status === 404 || uploadResponse.status === 405) &&
      getSelectedVideoFile()
    ) {
      response = await fetch(`${baseUrl}/evaluate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
    }

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
