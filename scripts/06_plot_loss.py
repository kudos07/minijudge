"""Plot training loss from Trainer state / metrics if available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/qlora_1.7b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run_dir)
    state_path = run / "trainer_state.json"
    if not state_path.exists():
        print(f"No trainer_state.json in {run} — train first.")
        return

    state = json.loads(state_path.read_text(encoding="utf-8"))
    history = state.get("log_history", [])
    steps, losses, eval_losses = [], [], []
    for row in history:
        if "loss" in row and "step" in row:
            steps.append(row["step"])
            losses.append(row["loss"])
        if "eval_loss" in row:
            eval_losses.append((row.get("step"), row["eval_loss"]))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    if steps:
        ax.plot(steps, losses, label="train_loss")
    if eval_losses:
        ax.plot([s for s, _ in eval_losses], [v for _, v in eval_losses], label="eval_loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"Training curve — {run.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = Path(args.out) if args.out else run / "loss_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
