const state = {
  config: null,
  selectedVariableIndex: null,
  hasFactoryDefaults: false,
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
    label.textContent = key;
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

function renderInclusionList() {
  const container = document.getElementById("variable-inclusion");
  container.innerHTML = "";

  const variables = getVariables();
  if (!variables.length) {
    container.innerHTML = '<p class="muted">No variables configured.</p>';
    return;
  }

  const header = document.createElement("p");
  header.className = "muted";
  header.textContent = "Include in data creation:";
  container.appendChild(header);

  const list = document.createElement("div");
  list.className = "inclusion-grid";

  variables.forEach((variable, index) => {
    const label = document.createElement("label");
    label.className = "tick";

    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(variable.include_in_data_creation);
    input.addEventListener("change", () => {
      variable.include_in_data_creation = input.checked;
      renderVariableSelect();
      if (state.selectedVariableIndex === index) {
        renderVariableEditor();
      }
    });

    const text = document.createElement("span");
    text.textContent = variable.name || `variable_${index + 1}`;

    label.appendChild(input);
    label.appendChild(text);
    list.appendChild(label);
  });

  container.appendChild(list);
}

function createBandRow(rowIndex, band, onChange, onDelete) {
  const row = document.createElement("tr");
  row.dataset.bandIndex = String(rowIndex);

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
  removeBtn.className = "btn btn-danger";
  removeBtn.textContent = "Remove";
  removeBtn.addEventListener("click", () => onDelete(rowIndex));
  removeCell.appendChild(removeBtn);

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
    toggleBtn.addEventListener("click", () => {
      delete variable.expected_range;
      renderVariableEditor();
    });
  } else {
    toggleBtn.textContent = "Add Range";
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
    renderVariableSelect();
    renderInclusionList();
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
  includeInput.addEventListener("change", () => {
    variable.include_in_data_creation = includeInput.checked;
    renderVariableSelect();
    renderInclusionList();
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
      distributionSummary.textContent = `distribution_pct total: ${sumDistribution(variable).toFixed(6)}`;
    }
  };

  const onBandDelete = (rowIndex) => {
    variable.bands.splice(rowIndex, 1);
    renderVariableEditor();
  };

  variable.bands.forEach((band, rowIndex) => {
    const row = createBandRow(rowIndex, band, onBandChange, onBandDelete);
    tbody.appendChild(row);
  });

  const bandActions = document.createElement("div");
  bandActions.className = "actions";

  const addBandBtn = document.createElement("button");
  addBandBtn.type = "button";
  addBandBtn.className = "btn btn-secondary";
  addBandBtn.textContent = "Add Row";
  addBandBtn.addEventListener("click", () => {
    variable.bands.push({ band: `band_${variable.bands.length + 1}`, distribution_pct: 0, bad_rate_ratio: 1 });
    renderVariableEditor();
  });

  const distributionSummary = document.createElement("p");
  distributionSummary.className = "muted";
  distributionSummary.textContent = `distribution_pct total: ${sumDistribution(variable).toFixed(6)}`;

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

function renderVariablesSection() {
  renderVariableSelect();
  renderInclusionList();
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
  await refreshJobs();
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
      <p>Created: ${job.created_at}</p>
      <pre>${logs || "No logs yet."}</pre>
    `;
    container.appendChild(card);
  });
}

async function refreshJobs() {
  const response = await fetch("/api/pipeline/jobs");
  if (!response.ok) {
    return;
  }
  const jobs = await response.json();
  renderJobs(jobs);
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
  attachHandlers();
  await loadConfig();
  await refreshJobs();
  window.setInterval(refreshJobs, 2500);
}

init().catch((error) => showMessage(error.message, "error"));

