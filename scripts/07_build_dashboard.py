#!/usr/bin/env python
"""Aggregate all MiniJudge experiment metrics into a local HTML dashboard.

Scans outputs/**/summary.json and *_metrics.json, writes:
  outputs/dashboard_data.json
  outputs/dashboard.html

Open dashboard.html in a browser after each experiment run.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Canonical experiment order for the research table
EXPERIMENT_ORDER = [
    "baseline_0.6b",
    "baseline_1.7b",
    "smoke_qlora_1.7b",
    "qlora_1.7b",
    "eval_qlora",
    "eval_qlora_reliability",
    "eval_bias",
]

DISPLAY_NAMES = {
    "baseline_0.6b": "0.6B prompt-only",
    "baseline_1.7b": "1.7B prompt-only",
    "smoke_qlora_1.7b": "1.7B smoke QLoRA",
    "qlora_1.7b": "1.7B QLoRA (train only)",
    "eval_qlora": "1.7B QLoRA + swap",
    "eval_qlora_reliability": "1.7B QLoRA + swap + vote",
    "eval_bias": "Bias: QLoRA fine-tuned",
    "eval_bias_suite": "Bias: QLoRA fine-tuned",
    "eval_bias_base": "Bias: 1.7B prompt-only",
    "eval_bias_suite_base": "Bias: 1.7B prompt-only",
}

# Rows shown in the main research comparison table
PRIMARY_IDS = {
    "baseline_0.6b",
    "baseline_1.7b",
    "eval_qlora",
    "eval_qlora_reliability",
}
BIAS_IDS = {
    "eval_bias",
    "eval_bias_suite",
    "eval_bias_base",
    "eval_bias_suite_base",
}
SKIP_IDS = {
    "qlora_1.7b",  # train-only; VRAM shown in notes
    "smoke_qlora_1.7b",
    "base",  # nested folder alias; use eval_bias_base instead
    "Bias suite",
}


def _experiment_id_from_path(path: Path, outputs_dir: Path) -> str:
    """Map outputs/... paths to stable experiment ids."""
    rel = path.relative_to(outputs_dir)
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "eval_bias" and parts[1] == "base":
        return "eval_bias_base"
    return parts[0]


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _pick_metrics(obj: dict) -> dict:
    """Normalize a metrics blob to flat keys we chart."""
    keys = [
        "accuracy",
        "macro_f1",
        "position_consistency",
        "conflict_rate",
        "invalid_output_rate",
        "latency_mean_sec",
        "peak_vram_gb",
        "n",
        "n_valid",
        "n_comparable",
    ]
    out = {k: obj.get(k) for k in keys if k in obj}
    if "confusion" in obj:
        out["confusion"] = obj["confusion"]
    if "pred_distribution" in obj:
        out["pred_distribution"] = obj["pred_distribution"]
    if "label_distribution" in obj:
        out["label_distribution"] = obj["label_distribution"]
    if "by_bias_type" in obj:
        out["by_bias_type"] = obj["by_bias_type"]
    return out


def collect_experiments(outputs_dir: Path) -> list[dict]:
    found: dict[str, dict] = {}

    if not outputs_dir.exists():
        return []

    for summary in outputs_dir.rglob("summary.json"):
        data = _load_json(summary)
        if not data:
            continue
        name = _experiment_id_from_path(summary, outputs_dir)
        # Prefer folder-derived id over noisy experiment_name duplicates
        entry = {
            "id": name,
            "display_name": DISPLAY_NAMES.get(name, name),
            "model_name": data.get("model_name"),
            "adapter_path": data.get("adapter_path"),
            "config": data.get("config") or {},
            "path": str(summary.relative_to(ROOT)).replace("\\", "/"),
            "datasets": {},
            "peak_vram_gb": data.get("peak_vram_gb"),
        }
        for split, metrics in (data.get("datasets") or {}).items():
            entry["datasets"][split] = _pick_metrics(metrics)
        found[name] = entry

    for metrics_path in outputs_dir.rglob("*_metrics.json"):
        name = _experiment_id_from_path(metrics_path, outputs_dir)
        data = _load_json(metrics_path)
        if not data:
            continue

        if metrics_path.name == "train_metrics.json":
            entry = found.get(name) or {
                "id": name,
                "display_name": DISPLAY_NAMES.get(name, name),
                "model_name": data.get("model_name"),
                "adapter_path": None,
                "config": {},
                "path": str(metrics_path.relative_to(ROOT)).replace("\\", "/"),
                "datasets": {},
            }
            entry["train"] = {
                "train_runtime_sec": data.get("train_runtime_sec"),
                "peak_vram_gb": data.get("peak_vram_gb"),
                "n_train": data.get("n_train"),
                "backend": data.get("backend"),
                "train_loss": data.get("train_loss"),
            }
            found[name] = entry
            continue

        split = metrics_path.name.replace("_metrics.json", "")
        if name not in found:
            found[name] = {
                "id": name,
                "display_name": DISPLAY_NAMES.get(name, name),
                "model_name": None,
                "adapter_path": None,
                "config": {},
                "path": str(metrics_path.relative_to(ROOT)).replace("\\", "/"),
                "datasets": {},
            }
        if split not in found[name]["datasets"]:
            found[name]["datasets"][split] = _pick_metrics(data)

    ordered = []
    seen = set()
    for key in EXPERIMENT_ORDER + ["eval_bias_base"]:
        if key in found:
            ordered.append(found[key])
            seen.add(key)
    for key, val in sorted(found.items()):
        if key not in seen:
            ordered.append(val)
    return ordered


def build_comparison_rows(experiments: list[dict]) -> list[dict]:
    """Primary research rows only — skips train-only and duplicate bias aliases."""
    by_id = {ex["id"]: ex for ex in experiments}
    rows = []

    def add_row(ex: dict, *, force_bias: bool = False) -> None:
        arena = ex["datasets"].get("arena_test") or {}
        jb = ex["datasets"].get("judgebench") or {}
        bias = ex["datasets"].get("bias_suite") or {}
        if force_bias and not bias:
            return
        if not arena and not jb and not bias and not ex.get("train"):
            return
        jb_n = jb.get("n")
        rows.append(
            {
                "experiment": ex["display_name"],
                "id": ex["id"],
                "n": arena.get("n") or bias.get("n") or jb_n,
                "arena_acc": arena.get("accuracy"),
                "arena_f1": arena.get("macro_f1"),
                "pos_consistency": arena.get("position_consistency"),
                "conflict_rate": arena.get("conflict_rate"),
                "invalid_rate": arena.get("invalid_output_rate"),
                "judgebench_acc": jb.get("accuracy"),
                "judgebench_n": jb_n,
                "judgebench_note": (
                    f"n={jb_n} seed only — not real JudgeBench"
                    if jb_n is not None and jb_n < 50
                    else None
                ),
                "bias_acc": bias.get("accuracy"),
                "latency_sec": arena.get("latency_mean_sec")
                or bias.get("latency_mean_sec")
                or jb.get("latency_mean_sec"),
                "peak_vram_gb": arena.get("peak_vram_gb")
                or bias.get("peak_vram_gb")
                or ex.get("peak_vram_gb")
                or (ex.get("train") or {}).get("peak_vram_gb"),
            }
        )

    for pid in ["baseline_0.6b", "baseline_1.7b", "eval_qlora", "eval_qlora_reliability"]:
        if pid in by_id:
            add_row(by_id[pid])

    # One clean bias pair
    for bid in ["eval_bias_base", "eval_bias"]:
        if bid in by_id:
            add_row(by_id[bid], force_bias=True)
        elif bid == "eval_bias_base" and "eval_bias_suite_base" in by_id:
            add_row(by_id["eval_bias_suite_base"], force_bias=True)
        elif bid == "eval_bias" and "eval_bias_suite" in by_id:
            add_row(by_id["eval_bias_suite"], force_bias=True)

    return rows


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MiniJudge Results Dashboard</title>
<style>
  :root {
    --bg: #0f1115;
    --panel: #171a21;
    --line: #2a303c;
    --text: #e8eaed;
    --muted: #9aa3b2;
    --accent: #6ea8fe;
    --good: #3dd68c;
    --bad: #f07178;
    --warn: #e6c07b;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: "Segoe UI", system-ui, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.45;
  }
  header {
    padding: 28px 32px 12px; border-bottom: 1px solid var(--line);
  }
  header h1 { margin: 0 0 6px; font-size: 22px; font-weight: 600; }
  header p { margin: 0; color: var(--muted); font-size: 13px; }
  main { padding: 24px 32px 48px; max-width: 1200px; margin: 0 auto; }
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0 28px; }
  .stat {
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px;
  }
  .stat .label { color: var(--muted); font-size: 12px; }
  .stat .value { font-size: 26px; font-weight: 600; margin-top: 4px; }
  .stat .sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
  h2 { font-size: 16px; margin: 28px 0 12px; font-weight: 600; }
  table {
    width: 100%; border-collapse: collapse; background: var(--panel);
    border: 1px solid var(--line); border-radius: 10px; overflow: hidden; font-size: 13px;
  }
  th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 500; background: #1c212b; }
  tr:last-child td { border-bottom: none; }
  td.num { font-variant-numeric: tabular-nums; }
  .good { color: var(--good); }
  .bad { color: var(--bad); }
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .chart-card {
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px;
  }
  .chart-card h3 { margin: 0 0 12px; font-size: 14px; font-weight: 600; }
  canvas { width: 100% !important; max-height: 280px; }
  .bars { display: flex; flex-direction: column; gap: 10px; }
  .bar-row { display: grid; grid-template-columns: 160px 1fr 52px; gap: 10px; align-items: center; }
  .bar-label { font-size: 12px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .bar-track { height: 10px; background: #222833; border-radius: 999px; overflow: hidden; }
  .bar-fill { height: 100%; background: var(--accent); border-radius: 999px; }
  .bar-val { font-size: 12px; font-variant-numeric: tabular-nums; text-align: right; }
  .empty { color: var(--muted); font-size: 13px; padding: 20px; border: 1px dashed var(--line); border-radius: 10px; }
  details { margin-top: 10px; }
  summary { cursor: pointer; color: var(--accent); font-size: 13px; }
  pre {
    background: #12151b; border: 1px solid var(--line); border-radius: 8px;
    padding: 12px; overflow: auto; font-size: 11px; color: var(--muted);
  }
  @media (max-width: 900px) {
    .stats, .charts { grid-template-columns: 1fr 1fr; }
    main { padding: 16px; }
  }
  @media (max-width: 600px) {
    .stats, .charts { grid-template-columns: 1fr; }
    .bar-row { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<header>
  <h1>MiniJudge Results Dashboard</h1>
  <p>Source: outputs/** · Generated __GENERATED__ · Research Q: QLoRA + swap + vote for a 1.7B judge</p>
</header>
<main>
  <div class="stats" id="stats"></div>
  <h2>Experiment comparison</h2>
  <div id="table-wrap"></div>
  <h2>Metric charts</h2>
  <div class="charts">
    <div class="chart-card"><h3>Arena accuracy by experiment</h3><div id="acc-bars" class="bars"></div></div>
    <div class="chart-card"><h3>Position consistency vs conflict rate</h3><div id="rel-bars" class="bars"></div></div>
    <div class="chart-card"><h3>Latency (sec / comparison)</h3><div id="lat-bars" class="bars"></div></div>
    <div class="chart-card"><h3>Peak VRAM (GB)</h3><div id="vram-bars" class="bars"></div></div>
  </div>
  <h2>Per-experiment detail</h2>
  <div id="details"></div>
</main>
<script>
const DATA = __DATA__;

function pct(x) {
  if (x === null || x === undefined) return "—";
  return (100 * x).toFixed(1) + "%";
}
function num(x, d=3) {
  if (x === null || x === undefined) return "—";
  return Number(x).toFixed(d);
}
function toneAcc(x) {
  if (x === null || x === undefined) return "";
  if (x >= 0.65) return "good";
  if (x < 0.55) return "bad";
  return "";
}

const rows = DATA.comparison || [];
const exps = DATA.experiments || [];

// Stats
  const best = rows.filter(r => r.arena_acc != null).sort((a,b) => b.arena_acc - a.arena_acc)[0];
const primaryCount = rows.filter(r => r.arena_acc != null).length;
const stats = [
  { label: "Primary rows shown", value: String(primaryCount), sub: "baselines + QLoRA evals + bias" },
  { label: "Best Arena accuracy", value: best ? pct(best.arena_acc) : "—", sub: best ? best.experiment : "run baselines first" },
  { label: "Bias: base vs FT", value: (() => {
      const b = rows.find(r => (r.id || "").includes("bias") && (r.id || "").includes("base"));
      const f = rows.find(r => r.id === "eval_bias" || r.experiment.includes("fine-tuned"));
      if (!b || !f || b.bias_acc == null || f.bias_acc == null) return "—";
      return pct(b.bias_acc) + " → " + pct(f.bias_acc);
    })(), sub: "prompt-only → QLoRA" },
  { label: "JudgeBench status", value: (() => {
      const j = rows.find(r => r.judgebench_n != null);
      if (!j) return "missing";
      return j.judgebench_n < 50 ? "SEED ONLY (n=" + j.judgebench_n + ")" : "n=" + j.judgebench_n;
    })(), sub: "do not report 75% as real yet" },
];
document.getElementById("stats").innerHTML = stats.map(s =>
  `<div class="stat"><div class="label">${s.label}</div><div class="value">${s.value}</div><div class="sub">${s.sub}</div></div>`
).join("");

// Table
if (!rows.length) {
  document.getElementById("table-wrap").innerHTML = `<div class="empty">No metrics yet. Run baselines, then re-run this dashboard script.</div>`;
} else {
  const head = ["Experiment","N","Arena Acc","Macro F1","Pos. cons.","Conflict","Invalid","JudgeBench","Bias","Latency","VRAM"];
  const body = rows.map(r => {
    const jb = r.judgebench_acc == null ? "—" :
      (r.judgebench_n != null && r.judgebench_n < 50
        ? pct(r.judgebench_acc) + " *"
        : pct(r.judgebench_acc));
    return `<tr>
    <td>${r.experiment}</td>
    <td class="num">${r.n ?? "—"}</td>
    <td class="num ${toneAcc(r.arena_acc)}">${pct(r.arena_acc)}</td>
    <td class="num">${pct(r.arena_f1)}</td>
    <td class="num">${pct(r.pos_consistency)}</td>
    <td class="num ${r.conflict_rate > 0.5 ? "bad" : ""}">${pct(r.conflict_rate)}</td>
    <td class="num">${pct(r.invalid_rate)}</td>
    <td class="num">${jb}</td>
    <td class="num">${pct(r.bias_acc)}</td>
    <td class="num">${num(r.latency_sec, 2)}</td>
    <td class="num">${num(r.peak_vram_gb, 2)}</td>
  </tr>`;
  }).join("");
  document.getElementById("table-wrap").innerHTML =
    `<table><thead><tr>${head.map(h=>`<th>${h}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table>
     <p style="color:var(--muted);font-size:12px;margin-top:8px">* JudgeBench star = builtin seed (n&lt;50), not a real benchmark score. Dashes mean that metric was not measured for that row. Train-only / duplicate bias rows are hidden.</p>`;
}

function renderBars(elId, items, key, color) {
  const el = document.getElementById(elId);
  const vals = items.filter(r => r[key] != null);
  if (!vals.length) { el.innerHTML = `<div class="empty">No data yet</div>`; return; }
  const max = Math.max(...vals.map(r => Number(r[key])), 0.0001);
  el.innerHTML = vals.map(r => {
    const v = Number(r[key]);
    const isRate = key.includes("acc") || key.includes("rate") || key.includes("consistency") || key.includes("f1");
    const label = isRate ? pct(v) : num(v, 2);
    const width = Math.max(2, 100 * (v / (isRate ? 1 : max)));
    return `<div class="bar-row">
      <div class="bar-label" title="${r.experiment}">${r.experiment}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${width}%;background:${color}"></div></div>
      <div class="bar-val">${label}</div>
    </div>`;
  }).join("");
}

renderBars("acc-bars", rows, "arena_acc", "var(--accent)");
// dual metric: show consistency bars
(() => {
  const el = document.getElementById("rel-bars");
  const vals = rows.filter(r => r.pos_consistency != null);
  if (!vals.length) { el.innerHTML = `<div class="empty">No data yet</div>`; return; }
  el.innerHTML = vals.map(r => {
    const c = Number(r.pos_consistency);
    const f = Number(r.conflict_rate || 0);
    return `<div style="margin-bottom:12px">
      <div class="bar-label" style="margin-bottom:4px">${r.experiment}</div>
      <div class="bar-row"><div class="bar-label">consistency</div>
        <div class="bar-track"><div class="bar-fill" style="width:${100*c}%;background:var(--good)"></div></div>
        <div class="bar-val">${pct(c)}</div></div>
      <div class="bar-row"><div class="bar-label">conflict</div>
        <div class="bar-track"><div class="bar-fill" style="width:${100*f}%;background:var(--bad)"></div></div>
        <div class="bar-val">${pct(f)}</div></div>
    </div>`;
  }).join("");
})();
renderBars("lat-bars", rows, "latency_sec", "var(--warn)");
renderBars("vram-bars", rows, "peak_vram_gb", "var(--accent)");

// Details
document.getElementById("details").innerHTML = exps.map(ex => {
  const arena = ex.datasets.arena_test;
  let conf = "";
  if (arena && arena.confusion) {
    const m = arena.confusion.matrix;
    conf = `<p style="color:var(--muted);font-size:12px;margin:8px 0 0">Confusion [[AA,AB],[BA,BB]] = ${JSON.stringify(m)} · preds ${JSON.stringify(arena.pred_distribution)}</p>`;
  }
  return `<details>
    <summary>${ex.display_name} <span style="color:var(--muted)">(${ex.id})</span></summary>
    <pre>${JSON.stringify(ex, null, 2)}</pre>${conf}
  </details>`;
}).join("") || `<div class="empty">No experiments found under outputs/</div>`;
</script>
</body>
</html>
"""


def write_dashboard(outputs_dir: Path) -> Path:
    experiments = collect_experiments(outputs_dir)
    comparison = build_comparison_rows(experiments)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiments": experiments,
        "comparison": comparison,
    }
    outputs_dir.mkdir(parents=True, exist_ok=True)
    data_path = outputs_dir / "dashboard_data.json"
    data_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    html = HTML_TEMPLATE.replace("__GENERATED__", payload["generated_at"]).replace(
        "__DATA__", json.dumps(payload)
    )
    html_path = outputs_dir / "dashboard.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build MiniJudge HTML dashboard")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--open", action="store_true", help="Open dashboard in browser")
    args = ap.parse_args()

    out = Path(args.outputs)
    if not out.is_absolute():
        out = ROOT / out
    path = write_dashboard(out)
    data = _load_json(out / "dashboard_data.json") or {}
    n = len(data.get("experiments") or [])
    print(f"Dashboard written: {path}")
    print(f"Experiments found: {n}")
    print(f"Data JSON: {out / 'dashboard_data.json'}")

    if args.open:
        import webbrowser

        webbrowser.open(path.resolve().as_uri())


if __name__ == "__main__":
    main()
