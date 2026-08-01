# Results

Filled from `outputs/**/*_metrics.json` (Arena n=600 unless noted).

## Primary table

| Config | Arena Acc | Arena Macro-F1 | Pos. consistency | Conflict rate | Bias Acc | Invalid % | Latency (s) | Peak VRAM (GB) | Train time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6B prompt-only | 49.2% | 33.0% | 0.0% | 100% | — | 0.2% | 0.26 | 0.94 | — |
| 1.7B prompt-only | 49.6% | 49.0% | 28.8% | 71.2% | 62.7% | 0.5% | 0.57 | 2.13 | — |
| 1.7B QLoRA + swap | **60.8%** | 60.8% | 60.8% | 39.2% | **42.7%** | 0.0% | 4.73 | 2.13 | ~8.0 h |
| 1.7B QLoRA + swap + vote | 60.7% | 60.7% | **62.7%** | 37.3% | — | 0.0% | 1.31* | 2.13 | — |

\*Mean latency per generation call in reliability mode; total wall time is higher because of multiple votes.

Training (`outputs/qlora_1.7b/train_metrics.json`): 3000 examples, 2 epochs, peak VRAM **3.22 GB**, train loss ≈ 0.999.

## Bias suite by type (n=25 each)

| Bias type | Base 1.7B | QLoRA FT | Delta |
|---|---:|---:|---:|
| Formatting | 100% | 100% | 0 |
| Majority opinion | 68% | 72% | +4 |
| Fake citation | 20% | 48% | +28 |
| Position | 48% | 12% | −36 |
| Irrelevant padding | 100% | 24% | −76 |
| Length | 40% | **0%** | −40 |
| **Overall** | **62.7%** | **42.7%** | **−20** |

## JudgeBench

Reported 75% on **4 builtin seed examples only**. Do not treat as a real JudgeBench score until the full dataset is loaded.

## Takeaways

1. **QLoRA helps Arena preference alignment** (~+11 Acc points vs 1.7B prompt-only).
2. **Position consistency roughly doubles** with QLoRA; vote adds a small extra bump.
3. **0.6B prompt-only collapses to always-A** (0% consistency).
4. **Fine-tuning hurts several bias traps**, especially length and irrelevant padding.
5. Next fixes: mix bias pairs into training, length-aware prompts, real JudgeBench download.

## Metric file locations

- `outputs/baseline_0.6b/arena_test_metrics.json`
- `outputs/baseline_1.7b/arena_test_metrics.json`
- `outputs/eval_qlora/arena_test_metrics.json`
- `outputs/eval_qlora_reliability/arena_test_metrics.json`
- `outputs/eval_bias/bias_suite_metrics.json`
- `outputs/eval_bias/base/bias_suite_metrics.json`
- `outputs/qlora_1.7b/train_metrics.json`
- `outputs/dashboard.html`
