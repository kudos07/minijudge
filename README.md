# MiniJudge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Research question:** How much do QLoRA fine-tuning, position swapping, and majority voting improve the accuracy and reliability of a 1.7B LLM-as-a-Judge?

A fully local, $0 LLM-as-a-Judge experiment on Chatbot Arena preferences + a synthetic bias suite (and optional JudgeBench) — no paid APIs.

| Version | Description |
|---|---|
| Qwen3-0.6B baseline | Prompt only |
| Qwen3-1.7B baseline | Prompt only |
| Qwen3-1.7B + QLoRA | Fine-tuned judge |
| Qwen3-1.7B + QLoRA + reliability | Position swap + majority vote |

## Hardware

Designed for **consumer NVIDIA GPUs (~6–8 GB VRAM)** using 4-bit QLoRA:

- `per_device_train_batch_size: 1`
- `gradient_accumulation_steps: 8`
- `max_seq_length: 512–768`
- Prefer **bf16** on RTX 40-series when available

Larger GPUs can raise sequence length / batch size in `configs/`. Start with the smoke config before a full train.

---

## Quick start

### 0. Environment

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Install a CUDA build of PyTorch that matches your system (see [pytorch.org](https://pytorch.org)). On Windows you can also use `scripts/00_install_windows.ps1`.

Default training uses **PEFT + TRL**. Unsloth is optional (`requirements-unsloth.txt`).

### 1. Prepare data

The Arena dataset is gated. Authenticate first:

1. Create a Read token: https://huggingface.co/settings/tokens  
2. Accept terms: https://huggingface.co/datasets/lmsys/chatbot_arena_conversations  
3. Login (do **not** commit tokens):

```bash
huggingface-cli login
# or set HF_TOKEN in your environment for the session only
```

```bash
python scripts/01a_check_arena_access.py
python scripts/01_prepare_data.py --config configs/data.yaml
```

This writes filtered train/val/test JSONL under `data/processed/`, builds the bias suite, and attempts JudgeBench (falls back to a tiny seed if unavailable).

### 2. Baselines

```bash
python scripts/02_run_baseline.py --config configs/baseline_0.6b.yaml
python scripts/02_run_baseline.py --config configs/baseline_1.7b.yaml
```

### 3. Smoke QLoRA, then full train

```bash
python scripts/03_train_qlora.py --config configs/train_smoke.yaml
python scripts/03_train_qlora.py --config configs/train_qlora.yaml
```

### 4. Evaluate

```bash
python scripts/04_evaluate.py --config configs/eval_qlora.yaml
python scripts/04_evaluate.py --config configs/eval_qlora_reliability.yaml
python scripts/05_bias_suite_eval.py --config configs/eval_bias.yaml
```

### 5. Tests + dashboard

```bash
pytest -q
python scripts/07_build_dashboard.py --open
```

### One-shot

```bash
python scripts/run_pipeline.py --stage all
```

Stages: `data` → `baseline` → `smoke` → `train` → `eval` → `bias`.

---

## Metrics

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

Saved snapshots: [`results/final/SUMMARY.md`](results/final/SUMMARY.md).

---

## Project layout

```text
configs/          YAML experiment configs
src/minijudge/    Library code (data, judge, train, eval)
scripts/          Runnable entry points
tests/            Parser / swap / metrics unit tests
data/             Raw + processed datasets (gitignored)
outputs/          Checkpoints + metric JSON (gitignored)
results/final/    Committed Arena + bias result snapshots
```

---

## Security notes

- Never commit Hugging Face tokens, API keys, or `.env` files with secrets.
- `.env` is gitignored; `.env.example` contains placeholders only.
- Model weights under `outputs/` are gitignored.

---

## Out of scope (v1)

7B models · paid teacher labels · A/B/TIE · rationale training · RL · Gradio UI · large multi-benchmark suites.

Prove A-vs-B classification first; everything else is a later ablation.

---

## Docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — system design
- [STEPS.md](STEPS.md) — detailed runbook
- [RESULTS.md](RESULTS.md) — metrics table
- [results/final/SUMMARY.md](results/final/SUMMARY.md) — Arena + bias snapshot

---

## License

This project is released under the [MIT License](LICENSE).

Third-party models and datasets keep their own licenses (e.g. Qwen3 Apache-2.0; LMSYS Chatbot Arena / JudgeBench terms on their Hugging Face or GitHub pages). Review those before redistributing derived weights or data.
