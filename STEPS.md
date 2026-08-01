# MiniJudge — detailed walkthrough

This document explains **what each step does, why it matters, and what you should look at**.

Your GPU: **RTX 4050 Laptop ~6 GB VRAM**. All configs already use 4-bit loading, batch size 1, and short sequences.

---

## Big picture

You are not just fine-tuning a model. You are answering:

> Does QLoRA + position swapping + majority voting make a 1.7B judge more accurate and more reliable?

Four configurations, one research question:

1. **0.6B prompt-only** — tiny ceiling check
2. **1.7B prompt-only** — main baseline
3. **1.7B + QLoRA** — did fine-tuning help?
4. **1.7B + QLoRA + swap + vote** — did reliability tricks help?

Two test worlds:

- **Chatbot Arena held-out** → human preference alignment
- **JudgeBench** → objective correctness (harder, different distribution)

---

## Step 0 — Environment

```powershell
cd c:\Users\saran\Videos\LLM-as-a-judge
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip wheel
pip install -r requirements.txt
pip install -e .
pytest -q
```

**Why:** Isolates dependencies. `pytest` should pass without a GPU (parser/metrics only).

**Windows note:** Default training uses **PEFT + TRL + bitsandbytes** (`backend: peft`). Unsloth is optional (`requirements-unsloth.txt`) and nicer on Linux/WSL; not required.

**Hugging Face:** First model/dataset download needs network. If a model is gated, run `huggingface-cli login`.

---

## Step 1 — Prepare data

The Arena dataset is gated. Authenticate first (same as HF docs):

```python
from datasets import load_dataset
ds = load_dataset("lmsys/chatbot_arena_conversations")
```

```powershell
# One-time HF setup
# 1) https://huggingface.co/settings/tokens  -> create Read token
# 2) https://huggingface.co/datasets/lmsys/chatbot_arena_conversations  -> Accept
huggingface-cli login

python scripts/01a_check_arena_access.py
python scripts/01_prepare_data.py --config configs/data.yaml
```

Or with an env token:

```powershell
$env:HF_TOKEN = "hf_xxxxxxxx"
python scripts/01_prepare_data.py
```

**What happens:**

1. Logs into Hugging Face, then `load_dataset("lmsys/chatbot_arena_conversations")`
2. Keeps only: English, single-turn, clear winner, no ties, length-capped
3. Writes:
   - `data/processed/arena_train.jsonl` (3,000)
   - `data/processed/arena_val.jsonl` (400)
   - `data/processed/arena_test.jsonl` (600)
   - `data/processed/arena_smoke_train.jsonl` (500)
   - `data/processed/arena_week1.jsonl` (1,000)
4. Tries JudgeBench; if unavailable, writes a tiny seed set so the pipeline still runs
5. Builds **150 controlled bias examples** (6 types x 25)

**Each JSONL row looks like:**

```json
{
  "id": "...",
  "question": "...",
  "response_a": "...",
  "response_b": "...",
  "label": "A"
}
```

**Why filter hard:** A 1.7B model cannot learn from messy multi-turn ties. Clean A/B pairs teach a clear preference boundary.

---

## Step 2 — Baseline (Week 1)

```powershell
python scripts/02_run_baseline.py --config configs/baseline_0.6b.yaml
python scripts/02_run_baseline.py --config configs/baseline_1.7b.yaml
```

Quick GPU sanity check first:

```powershell
python scripts/02_run_baseline.py --config configs/baseline_1.7b.yaml --max-examples 20
```

**What is measured:**

| Metric | Meaning |
|---|---|
| Accuracy | Matches human/gold label |
| Macro F1 | Balanced A/B quality |
| Position consistency | Same winner after swapping A↔B |
| Conflict rate | Preference flips after swap |
| Invalid-output rate | Did not produce A or B |
| Latency / Peak VRAM | Practicality on your laptop |

**Why position swap already at baseline:** Tiny models often have strong “always pick A” bias. Catching that early is part of the story.

Results land in `outputs/baseline_*/`.

---

## Step 3 — Smoke QLoRA (Week 2 start)

```powershell
python scripts/03_train_qlora.py --config configs/train_smoke.yaml
```

**Settings:** 500 examples · 1 epoch · seq 512 · LoRA rank 8 · 4-bit base

**Goal:** Prove training fits in 6 GB and the loss moves — not publishable numbers yet.

If you OOM:

1. Lower `max_seq_length` to `384` in the YAML
2. Raise `gradient_accumulation_steps` (same effective batch, same memory)
3. Stay on 1.7B; do not jump to 4B yet

Adapter → `outputs/smoke_qlora_1.7b/`.

---

## Step 4 — Full QLoRA (Week 2)

```powershell
python scripts/03_train_qlora.py --config configs/train_qlora.yaml
```

**Settings:** 3,000 examples · 2 epochs · seq 768 · LoRA rank 16

**Training format:** prompt → target `A` or `B` only (no explanations).

**Why no rationales in v1:** Small models copy explanation style instead of learning the preference boundary. Classification first; explanations later as an ablation.

Adapter → `outputs/qlora_1.7b/`. Check `train_metrics.json` for time and peak VRAM.

---

## Step 5 — Reliability evaluation (Week 3)

### 5a — Fine-tuned, deterministic + swap

```powershell
python scripts/04_evaluate.py --config configs/eval_qlora.yaml
```

### 5b — Fine-tuned + swap + majority-at-3

```powershell
python scripts/04_evaluate.py --config configs/eval_qlora_reliability.yaml
```

Runs each ordering 3 times with light temperature, majority-votes, maps swap predictions back to original A/B identity.

### 5c — Bias suite

```powershell
python scripts/05_bias_suite_eval.py --config configs/eval_bias.yaml
```

Six attacks (position, length, fake citations, formatting, majority opinion, irrelevant padding). Per-type accuracy is written under `outputs/eval_bias/`.

---

## Final table you should fill

Copy into your README / paper notes after runs finish:

```text
Config                         Arena Acc  Arena PosCons  JudgeBench Acc  Bias Acc  Invalid%  VRAM
0.6B prompt-only
1.7B prompt-only
1.7B QLoRA
1.7B QLoRA + swap
1.7B QLoRA + swap + vote
```

That table **is** the project deliverable.

---

## Suggested weekly cadence

| Week | Focus | Commands |
|---|---|---|
| 1 | Data + baselines | `01_prepare_data` → both `02_run_baseline` |
| 2 | QLoRA | `train_smoke` → `train_qlora` → compare vs baseline |
| 3 | Reliability | `eval_qlora*` + bias suite + error analysis |

---

## What not to do yet

- 7B fine-tuning on 6 GB
- Paid teacher labels (GPT-4/Claude)
- A/B/TIE three-class training
- Rationale / CoT targets
- RL / DPO as the first training method
- Gradio UI
- Ten benchmarks

Ship one clean A-vs-B story first.
