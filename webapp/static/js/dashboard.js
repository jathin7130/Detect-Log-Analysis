const COLORS = {
  isolation_forest: "#6b6d72",
  random_forest: "#15803d",
  lstm: "#0f766e",
  transformer: "#6d4fb0",
  lstm_combined: "#a15c07",
};
const LABELS = {
  isolation_forest: "Isolation Forest",
  random_forest: "Random Forest",
  lstm: "LSTM",
  transformer: "Transformer",
  lstm_combined: "LSTM + semantic",
};

function clockTick() {
  const el = document.getElementById("clock");
  if (el) el.textContent = new Date().toISOString().replace("T", " ").slice(0, 19) + " UTC";
}
setInterval(clockTick, 1000);
clockTick();

function barRow(label, value, max, color) {
  const pct = Math.max(2, (value / max) * 100);
  return `
    <div class="bar-row">
      <div class="bar-label">${label}</div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${pct}%; background:${color};">${value.toFixed(3)}</div>
      </div>
    </div>`;
}

async function renderDetectorCharts() {
  const res = await fetch("/api/detector_metrics");
  const data = await res.json();
  const container = document.getElementById("detector-charts");

  for (const [dataset, models] of Object.entries(data)) {
    const panel = document.createElement("div");
    panel.className = "panel";
    let html = `<div class="panel-header"><h3>${dataset.toUpperCase()}</h3><span class="hint">F1 score</span></div>`;

    const order = ["isolation_forest", "random_forest", "lstm", "transformer", "lstm_combined"];
    for (const model of order) {
      if (!(model in models)) continue;
      const entry = models[model];
      const f1 = entry.metrics?.f1;
      const f1Value = typeof f1 === "object" ? f1.mean : f1;
      if (f1Value === undefined) continue;
      html += barRow(LABELS[model] || model, f1Value, 1.0, COLORS[model] || "#58a6ff");
    }
    panel.innerHTML = html;
    container.appendChild(panel);
  }
}

function severityBadge(relIncrease) {
  if (relIncrease >= 20) return '<span class="badge badge-bad">catastrophic</span>';
  if (relIncrease >= 5) return '<span class="badge badge-warn">severe</span>';
  return '<span class="badge badge-info">moderate</span>';
}

async function renderDriftTable() {
  const res = await fetch("/api/drift");
  const data = await res.json();
  const tbody = document.querySelector("#drift-table tbody");
  let rows = "";
  for (const [dataset, d] of Object.entries(data)) {
    if (!d.false_positive_rate_clean === undefined) continue;
    rows += `<tr>
      <td>${dataset.toUpperCase()}</td>
      <td class="mono-num">${(d.false_positive_rate_clean * 100).toFixed(1)}%</td>
      <td class="mono-num">${(d.false_positive_rate_drifted * 100).toFixed(1)}%</td>
      <td class="mono-num">${d.fpr_increase_relative.toFixed(1)}x</td>
      <td>${severityBadge(d.fpr_increase_relative)}</td>
    </tr>`;
  }
  tbody.innerHTML = rows;
}

async function renderCrossDomainTable() {
  const res = await fetch("/api/cross_domain");
  const data = await res.json();
  const tbody = document.querySelector("#cross-domain-table tbody");
  let rows = "";
  for (const [pair, d] of Object.entries(data)) {
    const label = pair.replace("_to_", " → ");
    const before = d.before_adaptation?.mean_raw_score;
    const after = d.after_adaptation?.mean_raw_score;
    rows += `<tr>
      <td>${label}</td>
      <td><span class="badge badge-bad">100%</span></td>
      <td class="mono-num">${before !== undefined ? before.toFixed(3) : "—"}</td>
      <td class="mono-num">${after !== undefined ? after.toFixed(3) : "—"}</td>
      <td class="hint">model's prediction flips from "always anomalous" to "always normal" — no real signal transfers</td>
    </tr>`;
  }
  tbody.innerHTML = rows;
}

async function renderTestbedTable() {
  const res = await fetch("/api/testbed_incidents");
  const data = await res.json();
  const tbody = document.querySelector("#testbed-table tbody");
  let rows = "";
  for (const inc of data.incidents || []) {
    rows += `<tr>
      <td class="mono-num">${inc.injected_at.slice(11, 19)}</td>
      <td>${inc.service}</td>
      <td><span class="badge badge-warn">${inc.fault_type}</span></td>
      <td class="mono-num">${inc.fault_duration_s}s</td>
    </tr>`;
  }
  tbody.innerHTML = rows || "<tr><td colspan='4' class='hint'>no incidents recorded yet</td></tr>";
}

renderDetectorCharts();
renderDriftTable();
async function renderRcaCard() {
  try {
    const res = await fetch("/api/rca_summary");
    if (!res.ok) return;
    const d = await res.json();
    document.getElementById("rca-correct").textContent = `${d.hand_check_n_correct}/${d.n_cases}`;
    document.getElementById("rca-sub").textContent =
      `gate overrode LLM ${d.n_gate_overrode_llm}x · ${d.hand_check_n_abstain} honest abstains`;
  } catch (e) {
    // rca_results.json not generated yet -- leave the placeholder dash.
  }
}

renderCrossDomainTable();
renderTestbedTable();
renderRcaCard();
