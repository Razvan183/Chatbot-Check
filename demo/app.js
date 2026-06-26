const state = {
  versions: [],
  datasets: [],
  runs: [],
  selectedRunId: null,
  comparisonBaselineId: null,
  comparisonCandidateId: null,
  selectedDocumentId: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("visible");
  window.setTimeout(() => toast.classList.remove("visible"), 3200);
}

function formatPercent(value) {
  if (value === null || value === undefined) return "n/a";
  return `${Math.round(value * 100)}%`;
}

function formatSignedPercent(value) {
  if (value === null || value === undefined) return "n/a";
  const rounded = Math.round(value * 100);
  return `${rounded > 0 ? "+" : ""}${rounded}%`;
}

function formatSignedNumber(value) {
  return `${value > 0 ? "+" : ""}${value}`;
}

function truncate(text, size = 220) {
  if (!text) return "";
  return text.length > size ? `${text.slice(0, size).trim()}...` : text;
}

function selectedVersionId() {
  return Number($("#versionSelect").value);
}

function renderVersions() {
  const select = $("#versionSelect");
  select.innerHTML = state.versions
    .map((version) => `<option value="${version.id}">${version.name}</option>`)
    .join("");
  renderVersionDetails();
}

function renderVersionDetails() {
  const version = state.versions.find((item) => item.id === selectedVersionId());
  $("#versionDetails").innerHTML = version
    ? `
      <div><span>Model</span> ${version.model_name}</div>
      <div><span>Top-k</span> ${version.top_k}</div>
      <div><span>Temperature</span> ${version.temperature}</div>
      <div><span>Chunking</span> ${version.chunk_size}/${version.chunk_overlap}</div>
    `
    : "<div>No versions found. Run the demo seed script first.</div>";
  populateTuningForm(version);
}

function populateTuningForm(version) {
  if (!version) return;
  $("#tunedNameInput").value = `${version.name}_tuned`;
  $("#tunedTopKInput").value = version.top_k;
  $("#tunedTemperatureInput").value = version.temperature;
  $("#tunedModelInput").value = version.model_name;
  $("#tunedPromptInput").value = version.prompt_template || "";
}

function renderDatasets() {
  $("#datasetSelect").innerHTML = state.datasets
    .map((dataset) => (
      `<option value="${dataset.id}">${dataset.name} (${dataset.case_count})</option>`
    ))
    .join("");
}

function renderEvidence(chunks) {
  $("#citationBadge").textContent = `${chunks.length} chunks`;
  $("#evidenceList").innerHTML = chunks.map((chunk) => `
    <article class="evidence-card">
      <div class="card-title">
        <span>[${chunk.chunk_id}] ${chunk.filename}</span>
        <span class="score">${chunk.score.toFixed(3)}</span>
      </div>
      <div class="card-meta">Document ${chunk.document_id}</div>
      <div class="card-body">${truncate(chunk.chunk_text, 420)}</div>
    </article>
  `).join("");
}

async function askQuestion() {
  const question = $("#questionInput").value.trim();
  if (!question) {
    showToast("Enter a question first.");
    return;
  }

  $("#askButton").disabled = true;
  $("#answerOutput").textContent = "Thinking...";
  $("#latencyBadge").textContent = "Running";
  $("#evidenceList").innerHTML = "";

  try {
    const payload = await api("/chat", {
      method: "POST",
      body: JSON.stringify({
        question,
        chatbot_version_id: selectedVersionId(),
      }),
    });
    $("#answerOutput").textContent = payload.answer;
    $("#latencyBadge").textContent = `${payload.latency_ms} ms`;
    $("#citationBadge").textContent = payload.citations.length
      ? `Citations ${payload.citations.join(", ")}`
      : "No citations";
    renderEvidence(payload.retrieved_chunks);
  } catch (error) {
    $("#answerOutput").textContent = error.message;
    $("#latencyBadge").textContent = "Error";
    showToast(error.message);
  } finally {
    $("#askButton").disabled = false;
  }
}

function renderRunSummary() {
  const latest = state.runs[0];
  $("#runSummary").innerHTML = latest ? `
    <div class="metric"><strong>${formatPercent(latest.overall_score)}</strong><span>Latest score</span></div>
    <div class="metric"><strong>${latest.passed_cases}/${latest.total_cases}</strong><span>Passed cases</span></div>
    <div class="metric"><strong>${latest.failed_cases}</strong><span>Failures</span></div>
  ` : `
    <div class="metric"><strong>0</strong><span>No evaluation runs yet</span></div>
  `;
}

function runOptionLabel(run) {
  return `${run.chatbot_version_name} - ${run.run_name}`;
}

function renderComparisonSelectors() {
  const baselineSelect = $("#baselineRunSelect");
  const candidateSelect = $("#candidateRunSelect");
  const runIds = new Set(state.runs.map((run) => run.id));
  const options = state.runs
    .map((run) => `<option value="${run.id}">${runOptionLabel(run)}</option>`)
    .join("");

  baselineSelect.innerHTML = options;
  candidateSelect.innerHTML = options;

  if (!runIds.has(state.comparisonCandidateId)) {
    state.comparisonCandidateId = null;
  }
  if (!runIds.has(state.comparisonBaselineId)) {
    state.comparisonBaselineId = null;
  }

  if (!state.comparisonCandidateId && state.runs.length) {
    state.comparisonCandidateId = state.runs[0].id;
  }
  if (!state.comparisonBaselineId && state.runs.length > 1) {
    state.comparisonBaselineId = state.runs[1].id;
  }
  if (!state.comparisonBaselineId && state.runs.length) {
    state.comparisonBaselineId = state.runs[0].id;
  }

  baselineSelect.value = state.comparisonBaselineId || "";
  candidateSelect.value = state.comparisonCandidateId || "";
  $("#comparisonBadge").textContent = state.runs.length > 1
    ? "Ready"
    : "Need two runs";
}

function renderRuns() {
  $("#runCountBadge").textContent = `${state.runs.length} runs`;
  renderRunSummary();
  renderComparisonSelectors();
  $("#runsList").innerHTML = state.runs.map((run) => `
    <button class="run-card ${run.id === state.selectedRunId ? "selected" : ""}" data-run-id="${run.id}" type="button">
      <div class="card-title">
        <span>${run.chatbot_version_name}</span>
        <span class="score">${formatPercent(run.overall_score)}</span>
      </div>
      <div class="card-meta">${run.eval_dataset_name} - ${run.status}</div>
      <div class="card-body">${run.passed_cases} passed, ${run.failed_cases} failed</div>
    </button>
  `).join("");

  $$(".run-card").forEach((button) => {
    button.addEventListener("click", () => loadRunResults(Number(button.dataset.runId)));
  });
}

function renderComparison(comparison) {
  $("#comparisonBadge").textContent = `${comparison.baseline_run.chatbot_version_name} vs ${comparison.candidate_run.chatbot_version_name}`;
  $("#comparisonSummary").innerHTML = `
    <div class="metric"><strong>${formatSignedPercent(comparison.overall_score_delta)}</strong><span>Score delta</span></div>
    <div class="metric"><strong>${formatSignedNumber(comparison.passed_cases_delta)}</strong><span>Passed cases</span></div>
    <div class="metric"><strong>${comparison.fixed_cases}</strong><span>Fixed cases</span></div>
    <div class="metric"><strong>${comparison.new_failures}</strong><span>New failures</span></div>
  `;

  $("#comparisonBreakdown").innerHTML = comparison.failure_breakdown.length
    ? comparison.failure_breakdown.map((item) => `
      <article class="breakdown-card">
        <div class="card-title">
          <span>${item.failure_type}</span>
          <span class="score">${formatSignedNumber(item.delta)}</span>
        </div>
        <div class="card-meta">Baseline ${item.baseline_count} - Candidate ${item.candidate_count}</div>
      </article>
    `).join("")
    : `
      <article class="breakdown-card">
        <div class="card-title"><span>No failures</span><span class="score">0</span></div>
        <div class="card-meta">Both runs passed all comparable cases.</div>
      </article>
    `;

  $("#comparisonCases").innerHTML = comparison.case_comparisons.map((item) => `
    <article class="result-card">
      <div class="card-title">
        <span class="${item.status === "new_failure" || item.status === "regressed" ? "failed" : "passed"}">${item.status}</span>
        <span class="score">${formatSignedPercent(item.score_delta)}</span>
      </div>
      <div class="card-meta">Case ${item.eval_case_id} - ${formatPercent(item.baseline_score)} to ${formatPercent(item.candidate_score)}</div>
      <div class="card-body"><strong>Q:</strong> ${item.question}</div>
      <div class="card-body"><strong>Failure:</strong> ${item.baseline_failure_type || "none"} to ${item.candidate_failure_type || "none"}</div>
    </article>
  `).join("");
}

function renderScorecard(scorecard) {
  $("#scorecardBadge").textContent = `${scorecard.run.chatbot_version_name} - ${scorecard.run.status}`;
  $("#scorecardMetrics").innerHTML = `
    <div class="metric"><strong>${formatPercent(scorecard.run.overall_score)}</strong><span>Overall score</span></div>
    <div class="metric"><strong>${scorecard.run.passed_cases}/${scorecard.run.total_cases}</strong><span>Passed cases</span></div>
    ${scorecard.metric_scores.map((metric) => `
      <div class="metric">
        <strong>${formatPercent(metric.score)}</strong>
        <span>${metric.label} (${metric.measured_cases})</span>
      </div>
    `).join("")}
  `;

  $("#failureSummary").innerHTML = scorecard.failure_summary.length
    ? scorecard.failure_summary.map((failure) => `
      <article class="breakdown-card">
        <div class="card-title">
          <span>${failure.failure_type}</span>
          <span class="score">${failure.count}</span>
        </div>
      </article>
    `).join("")
    : `
      <article class="breakdown-card">
        <div class="card-title"><span>No failures</span><span class="score">0</span></div>
        <div class="card-meta">This run passed every evaluated case.</div>
      </article>
    `;

  $("#recommendationsList").innerHTML = scorecard.recommendations.map((item) => `
    <article class="result-card">
      <div class="card-title">
        <span>${item.parameter}</span>
        <span class="score">${item.suggested_value}</span>
      </div>
      <div class="card-meta">${item.current_value} to ${item.suggested_value}</div>
      <div class="card-body">${item.reason}</div>
    </article>
  `).join("");
}

async function loadScorecard(runId) {
  try {
    const scorecard = await api(`/evaluations/runs/${runId}/scorecard`);
    renderScorecard(scorecard);
  } catch (error) {
    $("#scorecardBadge").textContent = "Unavailable";
    $("#scorecardMetrics").innerHTML = "";
    $("#failureSummary").innerHTML = "";
    $("#recommendationsList").innerHTML = "";
    showToast(error.message);
  }
}

async function compareRuns() {
  const baselineId = Number($("#baselineRunSelect").value);
  const candidateId = Number($("#candidateRunSelect").value);
  if (!baselineId || !candidateId) {
    showToast("Choose two evaluation runs first.");
    return;
  }
  if (baselineId === candidateId) {
    showToast("Choose two different runs to compare.");
    return;
  }

  state.comparisonBaselineId = baselineId;
  state.comparisonCandidateId = candidateId;
  $("#compareRunsButton").disabled = true;
  $("#comparisonBadge").textContent = "Comparing";

  try {
    const comparison = await api(`/evaluations/runs/${baselineId}/compare/${candidateId}`);
    renderComparison(comparison);
  } catch (error) {
    showToast(error.message);
    $("#comparisonBadge").textContent = "Error";
  } finally {
    $("#compareRunsButton").disabled = false;
  }
}

async function createTunedVersion() {
  const name = $("#tunedNameInput").value.trim();
  const topK = Number($("#tunedTopKInput").value);
  const temperature = Number($("#tunedTemperatureInput").value);
  const modelName = $("#tunedModelInput").value.trim();
  const promptTemplate = $("#tunedPromptInput").value.trim();
  const baseVersion = state.versions.find((item) => item.id === selectedVersionId());

  if (!name || !topK || !modelName || Number.isNaN(temperature)) {
    showToast("Enter a name, top-k, temperature, and model.");
    return;
  }

  $("#createVersionButton").disabled = true;
  try {
    const created = await api("/chatbot-versions", {
      method: "POST",
      body: JSON.stringify({
        name,
        description: `Tuned from ${baseVersion ? baseVersion.name : "UI settings"}`,
        model_name: modelName,
        embedding_model: baseVersion ? baseVersion.embedding_model : undefined,
        chunk_size: baseVersion ? baseVersion.chunk_size : undefined,
        chunk_overlap: baseVersion ? baseVersion.chunk_overlap : undefined,
        top_k: topK,
        temperature,
        prompt_template: promptTemplate || null,
      }),
    });
    state.versions = await api("/chatbot-versions");
    renderVersions();
    $("#versionSelect").value = created.id;
    renderVersionDetails();
    showToast("Tuned version created. Run an evaluation to score it.");
  } catch (error) {
    showToast(error.message);
  } finally {
    $("#createVersionButton").disabled = false;
  }
}

function openComparisonReport() {
  const baselineId = Number($("#baselineRunSelect").value);
  const candidateId = Number($("#candidateRunSelect").value);
  if (!baselineId || !candidateId) {
    showToast("Choose two evaluation runs first.");
    return;
  }
  if (baselineId === candidateId) {
    showToast("Choose two different runs to report.");
    return;
  }

  state.comparisonBaselineId = baselineId;
  state.comparisonCandidateId = candidateId;
  window.open(
    `/evaluations/runs/${baselineId}/compare/${candidateId}/report`,
    "_blank",
    "noopener",
  );
}

function renderResults(results) {
  $("#resultCountBadge").textContent = `${results.length} cases`;
  $("#resultsList").innerHTML = results.map((result) => `
    <article class="result-card">
      <div class="card-title">
        <span class="${result.passed ? "passed" : "failed"}">${result.passed ? "Passed" : "Failed"}</span>
        <span class="score">${formatPercent(result.overall_case_score)}</span>
      </div>
      <div class="card-meta">${result.failure_type || "passed"} - case ${result.eval_case_id}</div>
      <div class="card-body"><strong>Q:</strong> ${result.question}</div>
      <div class="card-body"><strong>A:</strong> ${truncate(result.generated_answer, 280)}</div>
    </article>
  `).join("");
}

async function loadRuns() {
  state.runs = await api("/evaluations/runs");
  if (!state.selectedRunId && state.runs.length) {
    state.selectedRunId = state.runs[0].id;
  }
  renderRuns();
  if (state.selectedRunId) {
    await loadRunResults(state.selectedRunId);
  }
}

async function loadRunResults(runId) {
  state.selectedRunId = runId;
  renderRuns();
  const [results] = await Promise.all([
    api(`/evaluations/runs/${runId}/results`),
    loadScorecard(runId),
  ]);
  renderResults(results);
}

async function runEvaluation() {
  const datasetId = Number($("#datasetSelect").value);
  const versionId = selectedVersionId();
  if (!datasetId || !versionId) {
    showToast("Seed a dataset and chatbot version first.");
    return;
  }

  $("#runEvaluationButton").disabled = true;
  showToast("Evaluation queued. Results will update shortly.");
  try {
    const summary = await api("/evaluations/runs", {
      method: "POST",
      body: JSON.stringify({
        eval_dataset_id: datasetId,
        chatbot_version_id: versionId,
      }),
    });
    state.selectedRunId = summary.eval_run_id;
    await loadRuns();
    showToast("Evaluation run started.");
    window.setTimeout(loadRuns, 1500);
  } catch (error) {
    showToast(error.message);
  } finally {
    $("#runEvaluationButton").disabled = false;
  }
}

async function loadDocuments() {
  const documents = await api("/documents");
  $("#documentsList").innerHTML = documents.map((document) => `
    <button class="document-card ${document.id === state.selectedDocumentId ? "selected" : ""}" data-document-id="${document.id}" type="button">
      <div class="card-title">
        <span>${document.filename}</span>
        <span class="score">${document.num_chunks}</span>
      </div>
      <div class="card-meta">${document.status} - ${document.document_type}</div>
    </button>
  `).join("");

  $$(".document-card").forEach((button) => {
    button.addEventListener("click", () => loadChunks(Number(button.dataset.documentId)));
  });
}

async function loadChunks(documentId) {
  state.selectedDocumentId = documentId;
  await loadDocuments();
  const chunks = await api(`/documents/${documentId}/chunks`);
  $("#chunkCountBadge").textContent = `${chunks.length} chunks`;
  $("#chunksList").innerHTML = chunks.map((chunk) => `
    <article class="chunk-card">
      <div class="card-title">
        <span>Chunk ${chunk.chunk_index}</span>
        <span class="score">#${chunk.id}</span>
      </div>
      <div class="card-meta">${chunk.section_title || "Untitled section"}</div>
      <div class="card-body">${chunk.chunk_text}</div>
    </article>
  `).join("");
}

function wireEvents() {
  $$(".nav-tab").forEach((button) => {
    button.addEventListener("click", () => {
      $$(".nav-tab").forEach((tab) => tab.classList.remove("active"));
      $$(".view").forEach((view) => view.classList.remove("active"));
      button.classList.add("active");
      $(`#${button.dataset.view}`).classList.add("active");
    });
  });

  $("#versionSelect").addEventListener("change", renderVersionDetails);
  $("#askButton").addEventListener("click", askQuestion);
  $("#questionInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) askQuestion();
  });
  $$(".quick-prompts button").forEach((button) => {
    button.addEventListener("click", () => {
      $("#questionInput").value = button.textContent;
      askQuestion();
    });
  });
  $("#runEvaluationButton").addEventListener("click", runEvaluation);
  $("#createVersionButton").addEventListener("click", createTunedVersion);
  $("#refreshRunsButton").addEventListener("click", loadRuns);
  $("#compareRunsButton").addEventListener("click", compareRuns);
  $("#openReportButton").addEventListener("click", openComparisonReport);
  $("#baselineRunSelect").addEventListener("change", (event) => {
    state.comparisonBaselineId = Number(event.target.value);
  });
  $("#candidateRunSelect").addEventListener("change", (event) => {
    state.comparisonCandidateId = Number(event.target.value);
  });
  $("#refreshDocsButton").addEventListener("click", loadDocuments);
}

async function init() {
  wireEvents();
  try {
    const health = await api("/health");
    $("#systemStatus").textContent = health.status;
    $("#systemStatus").classList.add("ok");
  } catch {
    $("#systemStatus").textContent = "offline";
    $("#systemStatus").classList.add("error");
  }

  try {
    const [versions, datasets] = await Promise.all([
      api("/chatbot-versions"),
      api("/evaluations/datasets"),
    ]);
    state.versions = versions;
    state.datasets = datasets;
    renderVersions();
    renderDatasets();
    await Promise.all([loadRuns(), loadDocuments()]);
  } catch (error) {
    showToast(error.message);
  }
}

init();
