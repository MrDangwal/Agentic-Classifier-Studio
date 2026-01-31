const el = (id) => document.getElementById(id);

let currentDatasetId = null;
let currentAgentId = null;
let labelBuffer = new Map();

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

async function loadDatasets() {
  const datasets = await fetchJSON('/datasets');
  const select = el('dataset-select');
  select.innerHTML = '';
  datasets.forEach(d => {
    const opt = document.createElement('option');
    opt.value = d.id;
    opt.textContent = `${d.id}: ${d.name}`;
    select.appendChild(opt);
  });
  if (datasets.length) {
    currentDatasetId = datasets[0].id;
    select.value = currentDatasetId;
  }
}

async function uploadCSV(evt) {
  evt.preventDefault();
  const formData = new FormData(evt.target);
  const status = el('upload-status');
  status.textContent = 'Uploading...';
  try {
    const res = await fetchJSON('/datasets/upload', { method: 'POST', body: formData });
    status.textContent = `Uploaded dataset ${res.dataset_id} with ${res.rows} rows.`;
    await loadDatasets();
  } catch (err) {
    status.textContent = `Upload failed: ${err.message}`;
  }
}

async function loadRows(filter) {
  if (!currentDatasetId) return;
  labelBuffer.clear();
  const rows = await fetchJSON(`/datasets/${currentDatasetId}/rows?filter=${filter}&limit=50`);
  renderTable(rows);
}

function renderTable(rows) {
  const table = document.createElement('table');
  table.innerHTML = `
    <thead>
      <tr><th>Text</th><th>Label</th><th>Pred</th><th>Conf</th></tr>
    </thead>
    <tbody></tbody>
  `;
  const tbody = table.querySelector('tbody');
  rows.forEach(r => {
    const tr = document.createElement('tr');
    const labelInput = document.createElement('input');
    labelInput.value = r.classification || '';
    labelInput.placeholder = 'label';
    labelInput.addEventListener('input', () => {
      labelBuffer.set(r.id, labelInput.value.trim());
    });

    tr.innerHTML = `
      <td>${r.text}</td>
      <td></td>
      <td>${r.predicted_label || ''}</td>
      <td>${r.confidence ? r.confidence.toFixed(2) : ''}</td>
    `;
    tr.children[1].appendChild(labelInput);
    tbody.appendChild(tr);
  });
  const container = el('label-table');
  container.innerHTML = '';
  container.appendChild(table);
}

async function saveLabels() {
  if (!currentDatasetId || labelBuffer.size === 0) return;
  const items = Array.from(labelBuffer.entries()).map(([row_id, classification]) => ({ row_id, classification }));
  await fetchJSON('/labels/bulk', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_id: currentDatasetId, items })
  });
  labelBuffer.clear();
  await loadRows('unlabeled');
}

async function createAgent() {
  if (!currentDatasetId) return;
  const name = el('agent-name').value || 'default-agent';
  const res = await fetchJSON('/agents', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_id: currentDatasetId, name })
  });
  currentAgentId = res.agent_id;
  el('agent-status').textContent = `Agent created: ${currentAgentId}`;
}

async function trainAgent() {
  if (!currentAgentId) return;
  const label_budget = parseInt(el('label-budget').value, 10);
  const target_f1 = parseFloat(el('target-f1').value);
  const res = await fetchJSON(`/agents/${currentAgentId}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label_budget, target_f1 })
  });
  el('train-metrics').textContent = `Metrics: ${JSON.stringify(res.metrics)}`;
}

async function runAgent() {
  if (!currentAgentId || !currentDatasetId) return;
  const res = await fetchJSON(`/agents/${currentAgentId}/run?dataset_id=${currentDatasetId}`, { method: 'POST' });
  el('run-status').textContent = `Updated ${res.updated} rows.`;
  await loadRows('uncertain');
}

function init() {
  el('upload-form').addEventListener('submit', uploadCSV);
  el('refresh-datasets').addEventListener('click', loadDatasets);
  el('dataset-select').addEventListener('change', (e) => { currentDatasetId = e.target.value; });
  el('load-unlabeled').addEventListener('click', () => loadRows('unlabeled'));
  el('load-uncertain').addEventListener('click', () => loadRows('uncertain'));
  el('save-labels').addEventListener('click', saveLabels);
  el('create-agent').addEventListener('click', createAgent);
  el('train-agent').addEventListener('click', trainAgent);
  el('run-agent').addEventListener('click', runAgent);
  loadDatasets();
}

window.addEventListener('DOMContentLoaded', init);
