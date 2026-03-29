const form = document.getElementById("pitchForm");
const clearBtn = document.getElementById("clearBtn");
const evaluateBtn = document.getElementById("evaluateBtn");
const statusText = document.getElementById("statusText");
const pitchPreview = document.getElementById("pitchPreview");

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

// Video player elements
const videoSelect = document.getElementById("videoSelect");
const videoContainer = document.getElementById("videoContainer");
const pitchVideo = document.getElementById("pitchVideo");
const videoTitle = document.getElementById("videoTitle");
const videoDuration = document.getElementById("videoDuration");
const videoRatingPanel = document.getElementById("videoRatingPanel");
const finalScore = document.getElementById("finalScore");
const videoRatingBand = document.getElementById("videoRatingBand");
const videoRatingText = document.getElementById("videoRatingText");

let selectedVideoFileName = "pitch.mp4";
let selectedVideoDurationSec = 120;

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

  summaryCard.innerHTML = `
    <h3>Overall Summary</h3>
    <div class="summary-score">
      <span>Score</span>
      <strong>${overall.toFixed(2)}</strong>
      <span>/ 10</span>
    </div>
    <p><strong>Language Detected:</strong> ${escapeHtml(summary.language_detected)}</p>
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
  Object.values(fields).forEach((field) => {
    field.value = "";
  });
  renderPitchPreview(toPayload());
  statusText.textContent = "Cleared.";
}

function fillSample() {
  fields.title.value = "RetailPulse";
  fields.slideText.value =
    "Problem: stock mismatch creates losses\nSolution: demand forecasting with bilingual AI guidance\nTraction: 12 stores, 18% inventory efficiency gain\nGo-to-market: channel partners and distributor referrals";
  fields.founderName.value = "A. Founder";
  fields.startupName.value = "RetailPulse";
  fields.sector.value = "RetailTech";
  fields.stage.value = "Seed";

  renderPitchPreview(toPayload());
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

Object.values(fields)
  .filter(Boolean)
  .forEach((field) => {
  field.addEventListener("input", () => renderPitchPreview(toPayload()));
  });

// Video player functions
async function loadVideoList() {
  try {
    const response = await fetch("/videos");
    const data = await response.json();

    // Clear current options except the placeholder
    const selectedValue = videoSelect.value;
    videoSelect.innerHTML = '<option value="">Select a video...</option>';

    // Add video options
    data.videos.forEach((video) => {
      const option = document.createElement("option");
      option.value = video;
      option.textContent = video;
      videoSelect.appendChild(option);
    });

    // Restore selection if it still exists
    if (selectedValue && data.videos.includes(selectedValue)) {
      videoSelect.value = selectedValue;
    }
  } catch (err) {
    console.error("Failed to load videos:", err);
  }
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

function handleVideoEnded() {
  // Buffer time of 2 seconds before showing rating
  const bufferTimeMs = 2000;

  setTimeout(() => {
    // Check if we have evaluation results
    if (!evaluationResults.classList.contains("hidden")) {
      const score = overallKpi.textContent.split(" ")[0];
      const band = bandKpi.textContent;
      showVideoRatingPanel(score, band);
    }
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

  // Set video source and show container
  pitchVideo.src = `/videos/${encodeURIComponent(videoName)}`;
  videoTitle.textContent = videoName;
  videoContainer.classList.remove("hidden");
  hideVideoRatingPanel();

  // Update form fields with video name if needed
  if (!fields.title.value) {
    fields.title.value = videoName.replace(/\.[^.]+$/, ""); // Remove extension
    renderPitchPreview(toPayload());
  }

  // Play the video
  pitchVideo.play().catch((err) => {
    console.error("Failed to play video:", err);
  });
}

// Video event listeners
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

// Load videos on page start
await loadVideoList();

renderPitchPreview(toPayload());
