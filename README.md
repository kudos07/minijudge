# MiniJudge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Research question:** How much do QLoRA fine-tuning, position swapping, and majority voting improve the accuracy and reliability of a 1.7B LLM-as-a-Judge?

A fully local, $0 LLM-as-a-Judge experiment on Chatbot Arena preferences + JudgeBench — no paid APIs.

| Version | Description |
|---|---|
| Qwen3-0.6B baseline | Prompt only |
| Qwen3-1.7B baseline | Prompt only |
| Qwen3-1.7B + QLoRA | Fine-tuned judge |
| Qwen3-1.7B + QLoRA + reliability | Position swap + majority vote |

> **Your hardware note:** Detected **RTX 4050 Laptop (~6 GB VRAM)**. Configs in `configs/` are tuned for 6 GB (shorter sequences, batch size 1, 4-bit QLoRA). The original plan assumed 8 GB; everything still works, just keep `max_seq_length ≤ 768` and avoid the optional 4B stretch until you verify VRAM headroom.

---

## Step-by-step roadmap

### Step 0 — Environment (once)

```powershell
cd c:\Users\saran\Videos\LLM-as-a-judge
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

If Unsloth fails on Windows, use the PEFT fallback path (`configs/train_qlora_peft.yaml`) — training still works without Unsloth.

### Step 1 — Prepare data (Week 1)

First get Hugging Face access to the gated Arena dataset:

1. Create a Read token: https://huggingface.co/settings/tokens  
2. Accept dataset terms: https://huggingface.co/datasets/lmsys/chatbot_arena_conversations  
3. Login:

```powershell
huggingface-cli login
# or:  $env:HF_TOKEN = "hf_xxxxxxxx"
```

Check access, then prepare splits:

```powershell
python scripts/01a_check_arena_access.py
python scripts/01_prepare_data.py --config configs/data.yaml
```

What this does:
- Loads `lmsys/chatbot_arena_conversations` via `datasets.load_dataset`
- Keeps English, single-turn, clear winners, no ties, length-capped
- Writes train / val / test JSONL under `data/processed/`
- Optionally fetches JudgeBench for the external challenge set

### Step 2 — Baseline judges (Week 1)

```powershell
python scripts/02_run_baseline.py --config configs/baseline_0.6b.yaml
python scripts/02_run_baseline.py --config configs/baseline_1.7b.yaml
```

Measures accuracy, macro-F1, position consistency, conflict rate, invalid-output rate, latency, peak VRAM.

### Step 3 — Smoke-train QLoRA (Week 2 start)

```powershell
python scripts/03_train_qlora.py --config configs/train_smoke.yaml
```

500 examples, 1 epoch, seq 512 — proves the pipeline fits in 6 GB VRAM.

### Step 4 — Full QLoRA train (Week 2)

```powershell
python scripts/03_train_qlora.py --config configs/train_qlora.yaml
```

3,000 examples, 2 epochs. Adapter saved to `outputs/qlora_1.7b/`.

### Step 5 — Evaluate reliability (Week 3)

```powershell
python scripts/04_evaluate.py --config configs/eval_qlora.yaml
python scripts/04_evaluate.py --config configs/eval_qlora_reliability.yaml
python scripts/05_bias_suite_eval.py --config configs/eval_bias.yaml
```

### Step 6 — Tests

```powershell
pytest -q
```

### Step 7 — Dashboard

After any experiment, rebuild the local HTML dashboard (aggregates all `outputs/`):

```powershell
python scripts/07_build_dashboard.py --open
```

Opens `outputs/dashboard.html` with comparison table + accuracy / consistency / latency / VRAM charts. Also open the Cursor canvas [minijudge-results](canvases) beside chat for the live summary.

### One-shot (after Step 0)

```powershell
python scripts/run_pipeline.py --stage all
```

Stages: `data` → `baseline` → `smoke` → `train` → `eval` → `bias`.

---

## Metrics reported

| Metric | Meaning |
|---|---|
| Accuracy | Correct A/B decisions |
| Macro F1 | Balanced classification quality |
| Position consistency | Same winner after swapping A/B |
| Conflict rate | Judgment reverses after swap |
| Bias-suite accuracy | Resistance to manipulated responses |
| Invalid-output rate | Anything other than A or B |
| Latency | Seconds per comparison |
| Peak VRAM | Max GPU memory used |
| Training time | Fine-tuning wall time |

---

## Project layout

```text
configs/          YAML experiment configs
src/minijudge/    Library code (data, judge, train, eval)
scripts/          Runnable entry points
tests/            Parser / swap / metrics unit tests
data/             Raw + processed datasets (gitignored)
outputs/          Checkpoints + metric JSON (gitignored)
```

---

## What we deliberately skip (v1)

7B models · paid teacher labels · A/B/TIE · rationale training · RL · Gradio UI · ten benchmarks.

Prove A-vs-B classification first; everything else is a later ablation.

---

## Docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — system design
- [STEPS.md](STEPS.md) — week-by-week runbook
- [RESULTS.md](RESULTS.md) — metrics table
- [results/final/SUMMARY.md](results/final/SUMMARY.md) — **saved Arena + Bias final snapshot**
- `outputs/dashboard.html` — live local dashboard (regenerated)
- `results/final/dashboard.html` — frozen dashboard copy

---

## License

This project is released under the [MIT License](LICENSE).

**Note:** Third-party models and datasets keep their own licenses (e.g. Qwen3 Apache-2.0; LMSYS Chatbot Arena / JudgeBench terms on their Hugging Face or GitHub pages). Review those before redistribution of derived weights or data.

