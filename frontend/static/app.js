const state = {
  config: null,
  selectedVariableIndex: null,
  hasFactoryDefaults: false,
  inclusionManagerOpen: false,
  jobsPanelOpen: false,
  latestJobs: [],
  artifactsPayload: null,
  artifactGroupFilter: "all",
  artifactSearch: "",
};

const TOOLTIP_DELAY_MS = 550;

const GLOBAL_PARAMETER_LABELS = {
  simulation_population: "Simulation Population",
  global_bad_rate_pct: "Global Bad Rate (%)",
  train_set_pct: "Training Set Size (%)",
  test_set_pct: "Test Set Size (%)",
};

const ARTIFACT_GROUP_ORDER = [
  "Data Dictionary",
  "Simulated Data",
  "Datasets",
  "Feature Selection",
  "Preprocessing",
  "Trained Models",
  "Model Comparison",
  "Visualizations",
  "Other",
];

const STATIC_HELP_TEXTS = {
  "help-button": "Open the full UI user guide in a new browser tab.",
  "reload-config": "Reload settings from disk and discard unsaved screen edits.",
  "validate-config": "Check your current settings for errors before saving.",
  "save-config": "Save your current settings to the configuration files.",
  "add-variable": "Create a new variable template you can configure.",
  "delete-variable": "Delete the currently selected variable from the setup.",
  "factory-reset-variables": "Restore variable settings back to your factory defaults file.",
  "opt-use-woe": "Use WOE-transformed datasets when running model training.",
  "opt-dry-run": "Preview archive actions without moving files.",
  "toggle-jobs": "Show or hide background job history and logs.",
  "clear-activity": "Clear messages shown in the activity feed.",
  "refresh-jobs": "Refresh job statuses and logs right now.",
  "refresh-artifacts": "Reload the generated artifacts list from disk.",
  "clear-artefacts": "Move current generated artifacts into archive storage.",
  "artifact-group-filter": "Filter artifacts by category.",
  "artifact-search": "Search artifacts by name, path, or category.",
  "variable-select": "Pick which variable to edit in the form below.",
};

const tooltipState = {
  timerId: null,
  target: null,
  element: null,
};

function showMessage(text, type = "info") {
  const box = document.getElementById("messages");
  const item = document.createElement("div");
  item.className = `message ${type}`;
  item.textContent = text;
  box.prepend(item);
}

function clearMessages() {
  document.getElementById("messages").innerHTML = "";
}

function parseTypedValue(value, valueType) {
  if (valueType === "number") {
    const parsed = Number(value);
    return Number.isNaN(parsed) ? value : parsed;
  }
  if (valueType === "boolean") {
    return value === "true";
  }
  return value;
}

function humanizeKey(key) {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function globalParameterLabel(key) {
  return GLOBAL_PARAMETER_LABELS[key] || humanizeKey(key);
}

function pipelineDisplayName(value) {
  const raw = String(value || "");
  if (!raw) {
    return "Pipeline";
  }
  return raw
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function artifactDownloadUrl(path) {
  return `/api/artifacts/download?path=${encodeURIComponent(path)}`;
}

function formatBytes(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value < 1024) {
    return `${Math.max(0, Math.round(value || 0))} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  if (value < 1024 * 1024 * 1024) {
    return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(value / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

function formatTimestamp(isoTimestamp) {
  if (!isoTimestamp) {
    return "n/a";
  }
  const date = new Date(isoTimestamp);
  if (Number.isNaN(date.getTime())) {
    return isoTimestamp;
  }
  return date.toLocaleString();
}

function normalizeText(value) {
  return String(value || "").trim().toLowerCase();
}

function ensureTooltipElement() {
  if (tooltipState.element) {
    return tooltipState.element;
  }
  const element = document.createElement("div");
  element.className = "help-tooltip";
  element.hidden = true;
  document.body.appendChild(element);
  tooltipState.element = element;
  return element;
}

function clearTooltipTimer() {
  if (tooltipState.timerId) {
    window.clearTimeout(tooltipState.timerId);
    tooltipState.timerId = null;
  }
}

function hideTooltip() {
  clearTooltipTimer();
  tooltipState.target = null;
  if (!tooltipState.element) {
    return;
  }
  tooltipState.element.hidden = true;
  tooltipState.element.textContent = "";
}

function placeTooltip(target, text) {
  const element = ensureTooltipElement();
  element.textContent = text;
  element.hidden = false;

  const rect = target.getBoundingClientRect();
  const tipRect = element.getBoundingClientRect();
  const margin = 10;
  let top = rect.bottom + margin;
  let left = rect.left;

  const maxLeft = window.innerWidth - tipRect.width - margin;
  if (left > maxLeft) {
    left = Math.max(margin, maxLeft);
  }

  if (top + tipRect.height > window.innerHeight - margin) {
    top = rect.top - tipRect.height - margin;
  }
  top = Math.max(margin, top);

  element.style.left = `${Math.round(left)}px`;
  element.style.top = `${Math.round(top)}px`;
}

function resolveHelpText(target) {
  if (!target) {
    return "";
  }
  const explicit = String(target.dataset.help || "").trim();
  if (explicit) {
    return explicit;
  }
  if (target.matches("input[type='checkbox']")) {
    return "Toggle this option on or off.";
  }
  if (target.matches("button")) {
    return "Click to run this action.";
  }
  if (target.matches(".artifact-pill")) {
    return "Filter the artifact list by this category.";
  }
  return "";
}

function scheduleTooltip(target, delayMs = TOOLTIP_DELAY_MS) {
  const helpText = resolveHelpText(target);
  if (!helpText) {
    hideTooltip();
    return;
  }
  clearTooltipTimer();
  tooltipState.target = target;
  tooltipState.timerId = window.setTimeout(() => {
    if (!tooltipState.target || tooltipState.target !== target) {
      return;
    }
    placeTooltip(target, helpText);
  }, delayMs);
}

function setElementHelpText(element, text) {
  if (!element || !text) {
    return;
  }
  element.dataset.help = text;
}

function applyStaticHelpTexts() {
  Object.entries(STATIC_HELP_TEXTS).forEach(([id, help]) => {
    const element = document.getElementById(id);
    if (element) {
      setElementHelpText(element, help);
    }
  });

  document.querySelectorAll(".run-btn").forEach((button) => {
    const pipeline = button.dataset.pipeline || "pipeline";
    if (pipeline === "archive") {
      setElementHelpText(button, "Archive current generated artifacts to clear the active output folders.");
      return;
    }
    const pipelineLabel = pipelineDisplayName(pipeline);
    setElementHelpText(button, `Start the ${pipelineLabel} process in the background.`);
  });
}

function installHelpTooltips() {
  const selector = "button, input[type='checkbox'], .artifact-pill";

  document.addEventListener("mouseover", (event) => {
    const target = event.target.closest(selector);
    if (!target) {
      hideTooltip();
      return;
    }
    if (target === tooltipState.target) {
      return;
    }
    scheduleTooltip(target);
  });

  document.addEventListener("mouseout", (event) => {
    if (!tooltipState.target) {
      return;
    }
    const from = event.target.closest(selector);
    if (!from || from !== tooltipState.target) {
      return;
    }
    const related = event.relatedTarget;
    if (related && from.contains(related)) {
      return;
    }
    hideTooltip();
  });

  document.addEventListener("focusin", (event) => {
    const target = event.target.closest(selector);
    if (!target) {
      return;
    }
    scheduleTooltip(target, 300);
  });

  document.addEventListener("focusout", () => {
    hideTooltip();
  });

  document.addEventListener("click", () => {
    hideTooltip();
  });

  window.addEventListener("scroll", hideTooltip, true);
  window.addEventListener("resize", hideTooltip);
}

function latestMeaningfulLog(job) {
  const logs = Array.isArray(job?.logs) ? job.logs : [];
  for (let idx = logs.length - 1; idx >= 0; idx -= 1) {
    const line = String(logs[idx] || "").trim();
    if (!line || line.startsWith("$")) {
      continue;
    }
    return line;
  }
  return "";
}

function updateGeneratorStatus(jobs) {
  const status = document.getElementById("generator-status");
  status.className = "generator-status";

  if (!Array.isArray(jobs) || !jobs.length) {
    status.classList.add("status-idle");
    status.textContent = "No data generation jobs started yet.";
    return;
  }

  const runningJobs = jobs.filter((job) => job.status === "running" || job.status === "queued");
  if (runningJobs.length) {
    const primary = runningJobs[0];
    const label = pipelineDisplayName(primary.pipeline);
    const step = latestMeaningfulLog(primary);
    const runningText = step
      ? `Running ${label} in background. Current step: ${step}`
      : `Running ${label} in background. Please wait for completion.`;
    status.classList.add("status-running");
    status.textContent = runningText;
    return;
  }

  const latest = jobs[0];
  const latestLabel = pipelineDisplayName(latest.pipeline);
  const completedAt = formatTimestamp(latest.completed_at || latest.created_at);
  if (latest.status === "completed") {
    status.classList.add("status-complete");
    status.textContent = `${latestLabel} completed successfully at ${completedAt}.`;
    return;
  }

  if (latest.status === "failed") {
    status.classList.add("status-failed");
    status.textContent = `${latestLabel} failed at ${completedAt}. Check Activity or Job History for details.`;
    return;
  }

  status.classList.add("status-idle");
  status.textContent = `${latestLabel} is currently ${latest.status}.`;
}

function setJobsPanelVisibility(open) {
  state.jobsPanelOpen = Boolean(open);
  const panel = document.getElementById("jobs-panel");
  const toggle = document.getElementById("toggle-jobs");
  panel.hidden = !state.jobsPanelOpen;
  toggle.textContent = state.jobsPanelOpen ? "Hide Job History" : "Show Job History";
  toggle.setAttribute("aria-expanded", state.jobsPanelOpen ? "true" : "false");
}

function getVariables() {
  return state.config?.variables || [];
}

function selectedVariable() {
  const variables = getVariables();
  if (
    state.selectedVariableIndex === null ||
    state.selectedVariableIndex < 0 ||
    state.selectedVariableIndex >= variables.length
  ) {
    return null;
  }
  return variables[state.selectedVariableIndex];
}

function normalizeConfig(config) {
  const normalized = {
    global_parameters: config.global_parameters || {},
    variables: Array.isArray(config.variables) ? config.variables : [],
  };

  normalized.variables = normalized.variables.map((variable) => {
    const copy = { ...variable };
    copy.name = String(copy.name || "");
    copy.type = String(copy.type || "categorical");
    copy.include_in_data_creation = Boolean(
      Object.prototype.hasOwnProperty.call(copy, "include_in_data_creation")
        ? copy.include_in_data_creation
        : true,
    );
    copy.bands = Array.isArray(copy.bands)
      ? copy.bands.map((band) => ({
          ...band,
          band: String(band.band || ""),
          distribution_pct: Number(band.distribution_pct),
          bad_rate_ratio: Number(band.bad_rate_ratio),
        }))
      : [];
    return copy;
  });

  return normalized;
}

function renderGlobalParameters() {
  const globalParams = state.config.global_parameters;
  const container = document.getElementById("global-params");
  container.innerHTML = "";

  Object.entries(globalParams).forEach(([key, value]) => {
    const wrap = document.createElement("label");
    wrap.className = "field";

    const label = document.createElement("span");
    label.textContent = globalParameterLabel(key);
    label.title = key;
    wrap.appendChild(label);

    const input = document.createElement("input");
    const valueType = typeof value;
    input.dataset.key = key;
    input.dataset.valueType = valueType;

    if (valueType === "number") {
      input.type = "number";
      input.step = "any";
      input.value = String(value);
    } else {
      input.type = "text";
      input.value = String(value);
    }

    input.addEventListener("input", () => {
      state.config.global_parameters[key] = parseTypedValue(input.value, valueType);
    });

    wrap.appendChild(input);
    container.appendChild(wrap);
  });
}

function buildVariableLabel(variable, index) {
  const enabled = variable.include_in_data_creation ? "on" : "off";
  const displayName = variable.name || `variable_${index + 1}`;
  return `${displayName} [${enabled}]`;
}

function countIncludedVariables() {
  return getVariables().reduce(
    (acc, variable) => acc + (variable.include_in_data_creation ? 1 : 0),
    0,
  );
}

function refreshInclusionViews() {
  renderVariableSelect();
  renderInclusionControls();
  renderInclusionManager();
}

function renderVariableSelect() {
  const select = document.getElementById("variable-select");
  const variables = getVariables();
  select.innerHTML = "";

  if (!variables.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No variables";
    select.appendChild(option);
    state.selectedVariableIndex = null;
    select.disabled = true;
    return;
  }

  select.disabled = false;
  variables.forEach((variable, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = buildVariableLabel(variable, index);
    select.appendChild(option);
  });

  if (
    state.selectedVariableIndex === null ||
    state.selectedVariableIndex < 0 ||
    state.selectedVariableIndex >= variables.length
  ) {
    state.selectedVariableIndex = 0;
  }

  select.value = String(state.selectedVariableIndex);
}

function renderInclusionControls() {
  const controls = document.getElementById("inclusion-controls");
  controls.innerHTML = "";

  const variables = getVariables();
  if (!variables.length) {
    const muted = document.createElement("p");
    muted.className = "muted";
    muted.textContent = "No variables configured.";
    controls.appendChild(muted);
    return;
  }

  const chip = document.createElement("span");
  chip.className = "inclusion-chip";
  chip.textContent = `${countIncludedVariables()} of ${variables.length} included`;
  controls.appendChild(chip);

  const toggleBtn = document.createElement("button");
  toggleBtn.type = "button";
  toggleBtn.className = "btn btn-secondary";
  toggleBtn.textContent = state.inclusionManagerOpen ? "Hide Include Manager" : "Manage Included Variables";
  setElementHelpText(toggleBtn, "Open or close the list used to choose which variables are included in data creation.");
  toggleBtn.addEventListener("click", () => {
    state.inclusionManagerOpen = !state.inclusionManagerOpen;
    renderInclusionControls();
    renderInclusionManager();
  });
  controls.appendChild(toggleBtn);
}

function renderInclusionManager() {
  const container = document.getElementById("variable-inclusion-manager");
  container.innerHTML = "";

  const variables = getVariables();
  if (!variables.length || !state.inclusionManagerOpen) {
    container.hidden = true;
    return;
  }

  container.hidden = false;

  const header = document.createElement("div");
  header.className = "inclusion-manager-head";

  const title = document.createElement("h3");
  title.textContent = "Include In Data Creation";
  header.appendChild(title);

  const closeBtn = document.createElement("button");
  closeBtn.type = "button";
  closeBtn.className = "btn btn-secondary";
  closeBtn.textContent = "Close";
  setElementHelpText(closeBtn, "Close this inclusion manager and return to variable editing.");
  closeBtn.addEventListener("click", () => {
    state.inclusionManagerOpen = false;
    renderInclusionControls();
    renderInclusionManager();
  });
  header.appendChild(closeBtn);

  container.appendChild(header);

  const table = document.createElement("table");
  table.className = "inclusion-table";
  table.innerHTML = `
    <thead>
      <tr>
        <th>Variable</th>
        <th>Include</th>
      </tr>
    </thead>
    <tbody></tbody>
  `;

  const tbody = table.querySelector("tbody");
  variables.forEach((variable, index) => {
    const row = document.createElement("tr");

    const nameCell = document.createElement("td");
    const nameSpan = document.createElement("span");
    nameSpan.className = "inclusion-name";
    nameSpan.title = variable.name || `variable_${index + 1}`;
    nameSpan.textContent = variable.name || `variable_${index + 1}`;
    nameCell.appendChild(nameSpan);

    const includeCell = document.createElement("td");
    includeCell.className = "inclusion-toggle-cell";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(variable.include_in_data_creation);
    setElementHelpText(
      input,
      `Include or exclude '${variable.name || `variable_${index + 1}`}' from data creation.`,
    );
    input.addEventListener("change", () => {
      variable.include_in_data_creation = input.checked;
      refreshInclusionViews();
      if (state.selectedVariableIndex === index) {
        renderVariableEditor();
      }
    });
    includeCell.appendChild(input);

    row.appendChild(nameCell);
    row.appendChild(includeCell);
    tbody.appendChild(row);
  });

  container.appendChild(table);
}

function createBandRow(rowIndex, band, onChange, onDelete, onReorder) {
  const row = document.createElement("tr");
  row.dataset.bandIndex = String(rowIndex);
  row.classList.add("band-row");
  row.draggable = true;

  row.addEventListener("dragstart", (event) => {
    row.classList.add("dragging");
    if (event.dataTransfer) {
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", String(rowIndex));
    }
  });

  row.addEventListener("dragend", () => {
    row.classList.remove("dragging");
    row.classList.remove("drop-target");
  });

  row.addEventListener("dragover", (event) => {
    event.preventDefault();
    row.classList.add("drop-target");
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = "move";
    }
  });

  row.addEventListener("dragleave", () => {
    row.classList.remove("drop-target");
  });

  row.addEventListener("drop", (event) => {
    event.preventDefault();
    row.classList.remove("drop-target");
    const fromRaw = event.dataTransfer ? event.dataTransfer.getData("text/plain") : "";
    const fromIndex = Number(fromRaw);
    if (Number.isNaN(fromIndex)) {
      return;
    }
    onReorder(fromIndex, rowIndex);
  });

  const dragCell = document.createElement("td");
  dragCell.className = "drag-cell";
  const dragHandle = document.createElement("button");
  dragHandle.type = "button";
  dragHandle.className = "drag-handle";
  dragHandle.textContent = "|||";
  dragHandle.setAttribute("aria-label", "Drag row");
  setElementHelpText(dragHandle, "Click and drag this row to move it up or down.");
  dragCell.appendChild(dragHandle);

  const bandCell = document.createElement("td");
  const bandInput = document.createElement("input");
  bandInput.type = "text";
  bandInput.value = String(band.band || "");
  bandInput.addEventListener("input", () => onChange(rowIndex, "band", bandInput.value));
  bandCell.appendChild(bandInput);

  const distCell = document.createElement("td");
  const distInput = document.createElement("input");
  distInput.type = "number";
  distInput.step = "any";
  distInput.value = String(band.distribution_pct ?? "");
  distInput.addEventListener("input", () => onChange(rowIndex, "distribution_pct", Number(distInput.value)));
  distCell.appendChild(distInput);

  const ratioCell = document.createElement("td");
  const ratioInput = document.createElement("input");
  ratioInput.type = "number";
  ratioInput.step = "any";
  ratioInput.value = String(band.bad_rate_ratio ?? "");
  ratioInput.addEventListener("input", () => onChange(rowIndex, "bad_rate_ratio", Number(ratioInput.value)));
  ratioCell.appendChild(ratioInput);

  const removeCell = document.createElement("td");
  const removeBtn = document.createElement("button");
  removeBtn.type = "button";
  removeBtn.className = "btn btn-danger btn-mini";
  removeBtn.textContent = "Remove";
  setElementHelpText(removeBtn, "Remove this band row from the selected variable.");
  removeBtn.addEventListener("click", () => onDelete(rowIndex));
  removeCell.appendChild(removeBtn);

  row.appendChild(dragCell);
  row.appendChild(bandCell);
  row.appendChild(distCell);
  row.appendChild(ratioCell);
  row.appendChild(removeCell);

  return row;
}

function renderExpectedRange(editor, variable) {
  const wrap = document.createElement("div");
  wrap.className = "expected-range";

  const head = document.createElement("div");
  head.className = "range-head";
  const title = document.createElement("h3");
  title.textContent = "Expected Range";
  head.appendChild(title);

  const toggleBtn = document.createElement("button");
  toggleBtn.type = "button";
  toggleBtn.className = "btn btn-secondary";

  if (variable.expected_range && typeof variable.expected_range === "object") {
    toggleBtn.textContent = "Remove Range";
    setElementHelpText(toggleBtn, "Remove the expected min and max range for this variable.");
    toggleBtn.addEventListener("click", () => {
      delete variable.expected_range;
      renderVariableEditor();
    });
  } else {
    toggleBtn.textContent = "Add Range";
    setElementHelpText(toggleBtn, "Add an expected min and max range for this variable.");
    toggleBtn.addEventListener("click", () => {
      variable.expected_range = { min: 0, max: 1 };
      renderVariableEditor();
    });
  }

  head.appendChild(toggleBtn);
  wrap.appendChild(head);

  if (variable.expected_range && typeof variable.expected_range === "object") {
    const fields = document.createElement("div");
    fields.className = "range-fields";

    const minField = document.createElement("label");
    minField.className = "field compact";
    const minLabel = document.createElement("span");
    minLabel.textContent = "min";
    const minInput = document.createElement("input");
    minInput.type = "number";
    minInput.step = "any";
    minInput.value = String(variable.expected_range.min ?? "");
    minInput.addEventListener("input", () => {
      variable.expected_range.min = Number(minInput.value);
    });
    minField.appendChild(minLabel);
    minField.appendChild(minInput);

    const maxField = document.createElement("label");
    maxField.className = "field compact";
    const maxLabel = document.createElement("span");
    maxLabel.textContent = "max";
    const maxInput = document.createElement("input");
    maxInput.type = "number";
    maxInput.step = "any";
    maxInput.value = String(variable.expected_range.max ?? "");
    maxInput.addEventListener("input", () => {
      variable.expected_range.max = Number(maxInput.value);
    });
    maxField.appendChild(maxLabel);
    maxField.appendChild(maxInput);

    fields.appendChild(minField);
    fields.appendChild(maxField);
    wrap.appendChild(fields);
  }

  editor.appendChild(wrap);
}

function renderVariableEditor() {
  const editor = document.getElementById("variable-editor");
  editor.innerHTML = "";

  const variable = selectedVariable();
  if (!variable) {
    editor.innerHTML = '<p class="muted">Select or add a variable to begin editing.</p>';
    return;
  }

  const card = document.createElement("article");
  card.className = "variable-card";

  const top = document.createElement("div");
  top.className = "variable-head";

  const nameField = document.createElement("label");
  nameField.className = "field compact";
  const nameLabel = document.createElement("span");
  nameLabel.textContent = "Name";
  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.value = variable.name;
  nameInput.addEventListener("input", () => {
    variable.name = nameInput.value;
    refreshInclusionViews();
  });
  nameField.appendChild(nameLabel);
  nameField.appendChild(nameInput);

  const typeField = document.createElement("label");
  typeField.className = "field compact";
  const typeLabel = document.createElement("span");
  typeLabel.textContent = "Type";
  const typeInput = document.createElement("input");
  typeInput.type = "text";
  typeInput.value = variable.type;
  typeInput.addEventListener("input", () => {
    variable.type = typeInput.value;
  });
  typeField.appendChild(typeLabel);
  typeField.appendChild(typeInput);

  const includeField = document.createElement("label");
  includeField.className = "tick include-toggle";
  const includeInput = document.createElement("input");
  includeInput.type = "checkbox";
  includeInput.checked = Boolean(variable.include_in_data_creation);
  setElementHelpText(includeInput, "Include this selected variable when generating new data.");
  includeInput.addEventListener("change", () => {
    variable.include_in_data_creation = includeInput.checked;
    refreshInclusionViews();
  });
  const includeText = document.createElement("span");
  includeText.textContent = "Include in data creation";
  includeField.appendChild(includeInput);
  includeField.appendChild(includeText);

  top.appendChild(nameField);
  top.appendChild(typeField);
  top.appendChild(includeField);

  const table = document.createElement("table");
  table.innerHTML = `
    <thead>
      <tr>
        <th>Move</th>
        <th>Band</th>
        <th>distribution_pct</th>
        <th>bad_rate_ratio</th>
        <th>Actions</th>
      </tr>
    </thead>
    <tbody></tbody>
  `;

  const tbody = table.querySelector("tbody");
  const onBandChange = (rowIndex, key, value) => {
    if (!Array.isArray(variable.bands) || !variable.bands[rowIndex]) {
      return;
    }
    variable.bands[rowIndex][key] = value;
    if (key === "distribution_pct") {
      updateDistributionSummary(distributionSummary, variable);
    }
  };

  const onBandDelete = (rowIndex) => {
    variable.bands.splice(rowIndex, 1);
    renderVariableEditor();
  };

  const onBandReorder = (fromIndex, toIndex) => {
    if (toIndex < 0 || toIndex >= variable.bands.length || fromIndex === toIndex) {
      return;
    }
    const [row] = variable.bands.splice(fromIndex, 1);
    variable.bands.splice(toIndex, 0, row);
    renderVariableEditor();
  };

  variable.bands.forEach((band, rowIndex) => {
    const row = createBandRow(
      rowIndex,
      band,
      onBandChange,
      onBandDelete,
      onBandReorder,
    );
    tbody.appendChild(row);
  });

  const bandActions = document.createElement("div");
  bandActions.className = "actions";

  const addBandBtn = document.createElement("button");
  addBandBtn.type = "button";
  addBandBtn.className = "btn btn-secondary";
  addBandBtn.textContent = "Add Row";
  setElementHelpText(addBandBtn, "Add a new band row to this variable.");
  addBandBtn.addEventListener("click", () => {
    variable.bands.push({ band: `band_${variable.bands.length + 1}`, distribution_pct: 0, bad_rate_ratio: 1 });
    renderVariableEditor();
  });

  const distributionSummary = document.createElement("p");
  distributionSummary.className = "distribution-summary";
  updateDistributionSummary(distributionSummary, variable);

  bandActions.appendChild(addBandBtn);

  card.appendChild(top);
  renderExpectedRange(card, variable);
  card.appendChild(table);
  card.appendChild(bandActions);
  card.appendChild(distributionSummary);

  editor.appendChild(card);
}

function sumDistribution(variable) {
  if (!Array.isArray(variable.bands)) {
    return 0;
  }
  return variable.bands.reduce((acc, band) => acc + Number(band.distribution_pct || 0), 0);
}

function updateDistributionSummary(element, variable) {
  const total = sumDistribution(variable);
  element.textContent = `Distribution total: ${total.toFixed(2)}%`;
  const isTarget = Math.abs(total - 100) < 1e-9;
  element.classList.toggle("status-ok", isTarget);
  element.classList.toggle("status-error", !isTarget);
}

function renderVariablesSection() {
  renderVariableSelect();
  renderInclusionControls();
  renderInclusionManager();
  renderVariableEditor();
}

function uniqueVariableName(base = "new_variable") {
  const names = new Set(getVariables().map((variable) => variable.name));
  if (!names.has(base)) {
    return base;
  }

  let suffix = 1;
  let candidate = `${base}_${suffix}`;
  while (names.has(candidate)) {
    suffix += 1;
    candidate = `${base}_${suffix}`;
  }
  return candidate;
}

function collectConfigFromState() {
  return {
    global_parameters: state.config.global_parameters,
    variables: state.config.variables,
  };
}

async function loadConfig() {
  const response = await fetch("/api/config");
  if (!response.ok) {
    throw new Error("Failed to load configuration.");
  }

  const data = await response.json();
  state.config = normalizeConfig(data);
  state.hasFactoryDefaults = Boolean(data.has_factory_defaults);
  document.getElementById("factory-reset-variables").disabled = !state.hasFactoryDefaults;

  if (getVariables().length && (state.selectedVariableIndex === null || state.selectedVariableIndex >= getVariables().length)) {
    state.selectedVariableIndex = 0;
  }

  renderGlobalParameters();
  renderVariablesSection();
}

async function validateConfig() {
  clearMessages();
  const response = await fetch("/api/config/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(collectConfigFromState()),
  });
  const data = await response.json();
  if (data.valid) {
    showMessage("Validation passed.", "success");
  } else {
    data.errors.forEach((error) => showMessage(error, "error"));
  }
}

async function saveConfig() {
  clearMessages();
  const response = await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(collectConfigFromState()),
  });

  if (!response.ok) {
    const errorPayload = await response.json();
    const errors = errorPayload?.detail?.errors || ["Save failed."];
    errors.forEach((error) => showMessage(error, "error"));
    return;
  }

  showMessage("Configuration saved with backup files.", "success");
}

async function factoryResetVariables() {
  clearMessages();
  if (!state.hasFactoryDefaults) {
    showMessage("No factory defaults file is available.", "error");
    return;
  }

  const confirmed = window.confirm("Reset variables to factory defaults? This overwrites current variable setup.");
  if (!confirmed) {
    return;
  }

  const response = await fetch("/api/config/variables/factory-reset", {
    method: "POST",
  });

  if (!response.ok) {
    const data = await response.json();
    showMessage(data.detail || "Factory reset failed.", "error");
    return;
  }

  state.selectedVariableIndex = 0;
  await loadConfig();
  showMessage("Variables reset to factory defaults.", "success");
}

async function runPipeline(pipeline) {
  clearMessages();
  const payload = {
    pipeline,
    use_woe: document.getElementById("opt-use-woe").checked,
    dry_run: document.getElementById("opt-dry-run").checked,
  };
  const response = await fetch("/api/pipeline/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const data = await response.json();
    const errorMessage = typeof data.detail === "string" ? data.detail : "Failed to start pipeline.";
    showMessage(errorMessage, "error");
    return;
  }

  const data = await response.json();
  showMessage(`Started ${pipeline} job: ${data.job_id}`, "info");
  await refreshJobs(true);
  await refreshArtifacts();
}

function renderJobs(jobs) {
  const container = document.getElementById("jobs");
  container.innerHTML = "";

  if (!jobs.length) {
    container.innerHTML = '<p class="muted">No jobs yet.</p>';
    return;
  }

  jobs.forEach((job) => {
    const card = document.createElement("article");
    card.className = `job-card ${job.status}`;
    const logs = Array.isArray(job.logs) ? job.logs.slice(-80).join("\n") : "";
    card.innerHTML = `
      <header>
        <h3>${job.pipeline}</h3>
        <span class="badge">${job.status}</span>
      </header>
      <p>ID: ${job.id}</p>
      <p>Created: ${formatTimestamp(job.created_at)}</p>
      <pre>${logs || "No logs yet."}</pre>
    `;

    const artifactsPanel = renderJobArtifacts(job);
    if (artifactsPanel) {
      card.appendChild(artifactsPanel);
    }

    container.appendChild(card);
  });
}

function renderJobArtifacts(job) {
  const items = Array.isArray(job.artifacts) ? job.artifacts : [];
  if (!items.length) {
    return null;
  }

  const panel = document.createElement("div");
  panel.className = "job-artifacts";

  const heading = document.createElement("p");
  heading.className = "job-artifacts-title";
  heading.textContent = `Useful artefacts (${items.length})`;
  panel.appendChild(heading);

  const list = document.createElement("ul");
  list.className = "job-artifacts-list";

  items.slice(0, 10).forEach((item) => {
    const row = document.createElement("li");

    const link = document.createElement("a");
    link.href = artifactDownloadUrl(item.path);
    link.textContent = item.name || "unnamed";
    link.setAttribute("download", "");
    link.target = "_blank";
    link.rel = "noopener noreferrer";

    const pathToggle = document.createElement("button");
    pathToggle.type = "button";
    pathToggle.className = "artifact-path-toggle";
    pathToggle.textContent = "Show Path";
    setElementHelpText(pathToggle, "Show or hide the full storage path for this artifact.");

    const path = document.createElement("span");
    path.className = "artifact-inline-path";
    path.textContent = item.path || "";
    path.hidden = true;
    pathToggle.addEventListener("click", () => {
      const nextHidden = !path.hidden;
      path.hidden = nextHidden;
      pathToggle.textContent = nextHidden ? "Show Path" : "Hide Path";
    });

    const topRow = document.createElement("div");
    topRow.className = "job-artifact-top";
    topRow.appendChild(link);
    topRow.appendChild(pathToggle);

    const meta = document.createElement("span");
    const group = item.group || "Other";
    const size = formatBytes(item.size_bytes);
    const status = item.status || "existing";
    meta.textContent = `${group} | ${size} | ${status}`;

    row.appendChild(topRow);
    row.appendChild(path);
    row.appendChild(meta);
    list.appendChild(row);
  });

  if (items.length > 10) {
    const more = document.createElement("p");
    more.className = "muted";
    more.textContent = `Showing 10 of ${items.length} artefacts. See the full list below.`;
    panel.appendChild(more);
  }

  panel.appendChild(list);
  return panel;
}

async function refreshJobs(forceRender = false) {
  const response = await fetch("/api/pipeline/jobs");
  if (!response.ok) {
    return;
  }
  const jobs = await response.json();
  state.latestJobs = jobs;
  updateGeneratorStatus(jobs);
  if (forceRender || state.jobsPanelOpen) {
    renderJobs(jobs);
  }
}

function ensureArtifactFilters(payload) {
  const select = document.getElementById("artifact-group-filter");
  const searchInput = document.getElementById("artifact-search");
  const groups = sortArtifactGroups(Array.isArray(payload?.groups) ? payload.groups : []);

  const previous = state.artifactGroupFilter || "all";
  select.innerHTML = "";

  const allOption = document.createElement("option");
  allOption.value = "all";
  allOption.textContent = "All Categories";
  select.appendChild(allOption);

  groups.forEach((group) => {
    const option = document.createElement("option");
    option.value = group.name;
    option.textContent = `${group.name} (${group.count})`;
    select.appendChild(option);
  });

  const available = new Set(["all", ...groups.map((group) => group.name)]);
  state.artifactGroupFilter = available.has(previous) ? previous : "all";
  select.value = state.artifactGroupFilter;
  searchInput.value = state.artifactSearch;
  renderArtifactQuickFilters(payload);
}

function filteredArtifacts(payload) {
  const items = Array.isArray(payload?.items) ? payload.items : [];
  const selectedGroup = state.artifactGroupFilter || "all";
  const searchText = normalizeText(state.artifactSearch);

  return items.filter((item) => {
    const groupOk = selectedGroup === "all" || item.group === selectedGroup;
    const searchTarget = `${item.name || ""} ${item.path || ""} ${item.group || ""}`;
    const searchOk = !searchText || normalizeText(searchTarget).includes(searchText);
    return groupOk && searchOk;
  });
}

function sortArtifactGroups(groups) {
  const orderMap = new Map(ARTIFACT_GROUP_ORDER.map((name, idx) => [name, idx]));
  return [...groups].sort((a, b) => {
    const ai = orderMap.has(a.name) ? orderMap.get(a.name) : Number.MAX_SAFE_INTEGER;
    const bi = orderMap.has(b.name) ? orderMap.get(b.name) : Number.MAX_SAFE_INTEGER;
    if (ai !== bi) {
      return ai - bi;
    }
    return String(a.name).localeCompare(String(b.name));
  });
}

function setArtifactGroupFilter(groupValue) {
  state.artifactGroupFilter = groupValue || "all";
  const select = document.getElementById("artifact-group-filter");
  if (select) {
    select.value = state.artifactGroupFilter;
  }
  renderArtifactQuickFilters(state.artifactsPayload);
  renderArtifacts(state.artifactsPayload);
}

function renderArtifactQuickFilters(payload) {
  const container = document.getElementById("artifact-quick-filters");
  container.innerHTML = "";

  const groups = Array.isArray(payload?.groups) ? payload.groups : [];
  if (!groups.length) {
    return;
  }

  const sorted = sortArtifactGroups(groups);
  const allCount = sorted.reduce((acc, group) => acc + Number(group.count || 0), 0);

  const createPill = (label, value, count) => {
    const pill = document.createElement("button");
    pill.type = "button";
    pill.className = `artifact-pill${state.artifactGroupFilter === value ? " active" : ""}`;
    pill.textContent = `${label} (${count})`;
    setElementHelpText(pill, `Quickly filter artifacts to ${label}.`);
    pill.addEventListener("click", () => setArtifactGroupFilter(value));
    return pill;
  };

  container.appendChild(createPill("All", "all", allCount));
  sorted.forEach((group) => {
    container.appendChild(createPill(group.name, group.name, group.count));
  });
}

function artifactNameWithPathToggle(item) {
  const wrapper = document.createElement("div");
  wrapper.className = "artifact-name-wrap";

  const name = document.createElement("span");
  name.className = "artifact-name";
  name.textContent = item.name || "unnamed";
  wrapper.appendChild(name);

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "artifact-path-toggle";
  toggle.textContent = "Show Path";
  setElementHelpText(toggle, "Show or hide the full storage path for this artifact.");
  wrapper.appendChild(toggle);

  const path = document.createElement("div");
  path.className = "artifact-path";
  path.textContent = item.path || "";
  path.hidden = true;
  wrapper.appendChild(path);

  toggle.addEventListener("click", () => {
    const nextHidden = !path.hidden;
    path.hidden = nextHidden;
    toggle.textContent = nextHidden ? "Show Path" : "Hide Path";
  });

  return wrapper;
}

function renderArtifacts(payload) {
  const container = document.getElementById("artifacts");
  container.innerHTML = "";

  const allItems = Array.isArray(payload?.items) ? payload.items : [];
  const items = filteredArtifacts(payload);
  if (!items.length) {
    const message = allItems.length
      ? "No artefacts match the current filter."
      : "No artefacts available yet.";
    container.innerHTML = `<p class="muted">${message}</p>`;
    return;
  }

  const summary = document.createElement("p");
  summary.className = "muted";
  summary.textContent = `Showing ${items.length} artefacts.`;
  container.appendChild(summary);

  const table = document.createElement("table");
  table.className = "artifacts-table";
  table.innerHTML = `
    <thead>
      <tr>
        <th>File</th>
        <th>Category</th>
        <th>Modified</th>
        <th>Size</th>
        <th>Download</th>
      </tr>
    </thead>
    <tbody></tbody>
  `;

  const tbody = table.querySelector("tbody");
  items.forEach((item) => {
    const row = document.createElement("tr");

    const fileCell = document.createElement("td");
    fileCell.appendChild(artifactNameWithPathToggle(item));

    const categoryCell = document.createElement("td");
    categoryCell.textContent = item.group || "Other";

    const modifiedCell = document.createElement("td");
    modifiedCell.textContent = formatTimestamp(item.modified_at);

    const sizeCell = document.createElement("td");
    sizeCell.textContent = formatBytes(item.size_bytes);

    const downloadCell = document.createElement("td");
    const link = document.createElement("a");
    link.href = artifactDownloadUrl(item.path);
    link.textContent = "Download";
    link.className = "artifact-download";
    link.setAttribute("download", "");
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    downloadCell.appendChild(link);

    row.appendChild(fileCell);
    row.appendChild(categoryCell);
    row.appendChild(modifiedCell);
    row.appendChild(sizeCell);
    row.appendChild(downloadCell);
    tbody.appendChild(row);
  });

  container.appendChild(table);
}

async function refreshArtifacts() {
  const response = await fetch("/api/artifacts?limit=250");
  if (!response.ok) {
    return;
  }
  const payload = await response.json();
  state.artifactsPayload = payload;
  ensureArtifactFilters(payload);
  renderArtifacts(payload);
}

function addVariable() {
  const variables = getVariables();
  const variable = {
    name: uniqueVariableName(),
    type: "categorical",
    include_in_data_creation: true,
    bands: [
      { band: "band_1", distribution_pct: 50, bad_rate_ratio: 1 },
      { band: "band_2", distribution_pct: 50, bad_rate_ratio: 1 },
    ],
  };

  variables.push(variable);
  state.selectedVariableIndex = variables.length - 1;
  renderVariablesSection();
}

function deleteSelectedVariable() {
  const variables = getVariables();
  const idx = state.selectedVariableIndex;
  if (idx === null || idx < 0 || idx >= variables.length) {
    return;
  }

  const name = variables[idx].name || `variable_${idx + 1}`;
  const confirmed = window.confirm(`Delete variable '${name}'?`);
  if (!confirmed) {
    return;
  }

  variables.splice(idx, 1);
  if (!variables.length) {
    state.selectedVariableIndex = null;
  } else if (idx >= variables.length) {
    state.selectedVariableIndex = variables.length - 1;
  }
  renderVariablesSection();
}

function attachHandlers() {
  document.getElementById("reload-config").addEventListener("click", () => {
    loadConfig()
      .then(() => showMessage("Configuration reloaded.", "info"))
      .catch((error) => showMessage(error.message, "error"));
  });

  document.getElementById("validate-config").addEventListener("click", () => {
    validateConfig().catch((error) => showMessage(error.message, "error"));
  });

  document.getElementById("save-config").addEventListener("click", () => {
    saveConfig().catch((error) => showMessage(error.message, "error"));
  });

  document.getElementById("add-variable").addEventListener("click", addVariable);
  document.getElementById("delete-variable").addEventListener("click", deleteSelectedVariable);
  document.getElementById("factory-reset-variables").addEventListener("click", () => {
    factoryResetVariables().catch((error) => showMessage(error.message, "error"));
  });
  document.getElementById("toggle-jobs").addEventListener("click", () => {
    const next = !state.jobsPanelOpen;
    setJobsPanelVisibility(next);
    if (next) {
      refreshJobs(true).catch((error) => showMessage(error.message, "error"));
    }
  });
  document.getElementById("refresh-jobs").addEventListener("click", () => {
    refreshJobs(true).catch((error) => showMessage(error.message, "error"));
  });
  document.getElementById("clear-activity").addEventListener("click", clearMessages);
  document.getElementById("refresh-artifacts").addEventListener("click", () => {
    refreshArtifacts().catch((error) => showMessage(error.message, "error"));
  });
  document.getElementById("artifact-group-filter").addEventListener("change", (event) => {
    setArtifactGroupFilter(event.target.value || "all");
  });
  document.getElementById("artifact-search").addEventListener("input", (event) => {
    state.artifactSearch = event.target.value || "";
    renderArtifacts(state.artifactsPayload);
  });

  document.getElementById("variable-select").addEventListener("change", (event) => {
    const value = Number(event.target.value);
    state.selectedVariableIndex = Number.isNaN(value) ? null : value;
    renderVariableEditor();
  });

  document.querySelectorAll(".run-btn").forEach((button) => {
    button.addEventListener("click", () => {
      runPipeline(button.dataset.pipeline).catch((error) => showMessage(error.message, "error"));
    });
  });
}

async function init() {
  installHelpTooltips();
  applyStaticHelpTexts();
  attachHandlers();
  setJobsPanelVisibility(false);
  await loadConfig();
  await refreshJobs(true);
  await refreshArtifacts();
  window.setInterval(() => {
    refreshJobs();
    refreshArtifacts();
  }, 3000);
}

init().catch((error) => showMessage(error.message, "error"));

