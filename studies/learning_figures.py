"""Render the recorded learning-signal demonstrations."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from studies.data import ROOT
from studies.figures import BLUE, INK, ORANGE, PURPLE, TEAL, finish, style


def render(report, output):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2))
    history = report["td"]["history"]
    trials = [row["trial"] + 1 for row in history]
    axes[0].plot(trials, [row["cue_error"] for row in history], color=TEAL, linewidth=2.2, label="Error at cue onset")
    axes[0].plot(trials, [row["reward_error"] for row in history], color=ORANGE, linewidth=2.2, label="Error at reward delivery")
    omission = report["td_omission"]
    axes[0].scatter([omission["trial"] + 1], [omission["reward_error"]], color=PURPLE, s=60, zorder=3, label="Reward omitted after learning")
    axes[0].axhline(0, color=INK, linewidth=.8)
    axes[0].set(xlabel="Trial", ylabel="Temporal-difference error δ", ylim=(-1.15, 1.15), title="01 / The error moves to the earliest reliable cue")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    colors = {"prediction_error": ORANGE, "belief_change": TEAL, "learning_progress": BLUE}
    labels = {"prediction_error": "Surprise (prediction error)", "belief_change": "Belief change", "learning_progress": "Learning progress"}
    for signal, trials_by_seed in report["curiosity"]["trials"].items():
        blocks = np.array([trial["screen_fraction_by_block"] for trial in trials_by_seed])
        x = np.arange(blocks.shape[1]) * 100 + 50
        axes[1].plot(x, blocks.mean(axis=0), color=colors[signal], linewidth=2.2, label=labels[signal])
        axes[1].fill_between(x, blocks.min(axis=0), blocks.max(axis=0), color=colors[signal], alpha=.12)
    axes[1].set(xlabel="Step", ylabel="Fraction of choices at the random screen", ylim=(-.03, 1.03), title="02 / Irreducible noise can look endlessly novel")
    axes[1].legend(frameon=False, fontsize=8, loc="center right")
    rows = report["proxy"]["rows"]
    counts = [row["candidates"] for row in rows]
    axes[2].semilogx(counts, [row["mean_proxy"] for row in rows], color=PURPLE, linewidth=2.2, marker="o", base=2, label="Proxy score of selected candidate")
    true = np.array([row["mean_true"] for row in rows])
    error = 1.96 * np.array([row["true_standard_error"] for row in rows])
    axes[2].errorbar(counts, true, yerr=error, color=TEAL, linewidth=2.2, marker="o", capsize=3, label="True utility (95% interval)")
    axes[2].axvline(report["proxy"]["true_optimum_candidates"], color=INK, linestyle=":", linewidth=1)
    axes[2].set(xlabel="Optimization pressure: candidates compared (log₂)", ylabel="Mean score of selected candidate", title="03 / A rising proxy can hide a falling goal")
    axes[2].legend(frameon=False, fontsize=8, loc="upper left")
    for axis in axes:
        axis.grid(axis="y")
    seeds = len(report["curiosity"]["seeds"])
    finish(fig, Path(output) / "learning_signals", "Optimize the signal you mean, not the one that is easiest to raise.",
           f"Tabular TD(0) · four-option exploration task over {seeds} seeds · best-of-n selection with 2,000 repeats per pressure level",
           "Seeded computational demonstrations, not neural recordings or a deployed-model evaluation. Shaded bands: min–max across seeds.\nBelief change is the KL divergence between successive predictive distributions; learning progress is the change in windowed surprise.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "studies/results/learning_signals.json")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/figures")
    args = parser.parse_args()
    render(json.loads(args.report.read_text()), args.output)
