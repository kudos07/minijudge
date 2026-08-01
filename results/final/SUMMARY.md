# MiniJudge — saved final results (Arena + Bias)

Snapshot of completed experiments. Raw metric JSON lives in this folder; narrative summary below.

**Saved:** 2026-08-02  
**Hardware:** RTX 4050 Laptop (~6 GB VRAM)  
**Model:** Qwen/Qwen3-1.7B (+ QLoRA adapter)

---

## A) Chatbot Arena results (human preference, n=600)

| Config | Acc | Macro F1 | Pos. consistency | Conflict | Invalid | Latency (s) | VRAM (GB) | File |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0.6B prompt-only | 49.2% | 33.0% | 0.0% | 100% | 0.2% | 0.26 | 0.94 | `arena_baseline_0.6b.json` |
| 1.7B prompt-only | 49.6% | 49.0% | 28.8% | 71.2% | 0.5% | 0.57 | 2.13 | `arena_baseline_1.7b.json` |
| 1.7B QLoRA + swap | **60.8%** | 60.8% | 60.8% | 39.2% | 0.0% | 4.73 | 2.13 | `arena_qlora_swap.json` |
| 1.7B QLoRA + swap + vote | 59.2% | 59.2% | **61.2%** | 38.8% | 0.0% | 0.55* | 2.13 | `arena_qlora_swap_vote.json` |

\*Mean per-call latency in that run’s metrics file; vote mode still does multiple generations total.

**Training:** 3000 pairs, 2 epochs, ~8h, peak VRAM **3.22 GB** → `train_qlora_1.7b.json`

### Arena takeaway
QLoRA improves preference alignment (~+11 Acc points) and roughly doubles position consistency vs 1.7B prompt-only. Majority vote does not improve accuracy further.

---

## B) Synthetic bias suite results (n=150)

| Config | Overall bias Acc | File |
|---|---:|---|
| 1.7B prompt-only (base) | **62.7%** | `bias_base_1.7b.json` |
| 1.7B QLoRA fine-tuned | **42.7%** | `bias_qlora.json` |

### By attack type (base → fine-tuned)

| Bias type | Base | QLoRA | Delta |
|---|---:|---:|---:|
| Formatting | 100% | 100% | 0 |
| Majority opinion | 68% | 72% | +4 |
| Fake citation | 20% | 48% | +28 |
| Position | 48% | 12% | −36 |
| Irrelevant padding | 100% | 24% | −76 |
| Length | 40% | **0%** | −40 |

### Bias takeaway
Fine-tuning helped Arena but **hurt** several traps — especially length and irrelevant padding.

---

## C) JudgeBench (not saved as final)

Current JudgeBench files are **n=4 builtin seeds** only (placeholder 75%).  
**Do not treat as a final result** until a real JudgeBench download is evaluated.

---

## Files in this folder

```text
SUMMARY.md                 ← this file (Arena + Bias)
RESULTS.md                 ← copy of repo results table
dashboard.html             ← cleaned local dashboard snapshot
dashboard_data.json
arena_baseline_0.6b.json
arena_baseline_1.7b.json
arena_qlora_swap.json
arena_qlora_swap_vote.json
bias_base_1.7b.json
bias_qlora.json
train_qlora_1.7b.json
```

Also documented in repo root:
- `RESULTS.md`
- `ARCHITECTURE.md`
- Cursor canvas: minijudge-results
