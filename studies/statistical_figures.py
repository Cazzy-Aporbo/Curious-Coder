"""Render the statistical comparison and its evidence flow, without invented results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import font_manager
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from studies.data import ROOT
from studies.figures import BLUE, INK, ORANGE, PAPER, PURPLE, TEAL, finish, style


def paired_figure(report, output):
    comparisons = report["paired_test_comparisons"]["comparisons"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    colors = [ORANGE, PURPLE]
    for ax, metric, label in zip(axes, ("brier", "log_loss"), ("Brier-score difference (dimensionless)", "Log-loss difference (nats / record)")):
        ax.axvline(0, color=INK, linestyle="--", linewidth=1.2)
        for row, (name, results) in enumerate(comparisons.items()):
            result = results[metric]
            lo, hi = result["pointwise_percentile_95"]
            point = result["candidate_minus_reference"]
            ax.plot([lo, hi], [row, row], color=colors[row], linewidth=4, solid_capstyle="round")
            ax.scatter(point, row, color=colors[row], s=90, edgecolor=PAPER, zorder=3)
            ax.annotate(f"{point:+.4f}  [{lo:+.4f}, {hi:+.4f}]", (point, row), xytext=(0, 17), textcoords="offset points", ha="center", fontsize=9)
        ax.set(yticks=range(len(comparisons)), yticklabels=list(comparisons), xlabel=label, ylim=(-.7, 1.7))
        ax.invert_yaxis()
        ax.grid(axis="x")
        ax.locator_params(axis="x", nbins=5)
        ax.set_title("01 / Probability error" if metric == "brier" else "02 / Confident errors receive more weight")
    intervals = [result["pointwise_percentile_95"] for comparison in comparisons.values() for result in comparison.values()]
    include_zero = sum(lo <= 0 <= hi for lo, hi in intervals)
    repeats = report["paired_test_comparisons"]["bootstrap_repeats"]
    finish(fig, output / "paired_comparison", "Compare the same records before comparing the models.",
           f"Candidate minus logistic regression · negative favors candidate · {len(report['excluded_test_rows'])} fixed held-out predictions",
           f"Source: pinned UCI WDBC predictions. Bars: pointwise 95% percentile intervals from {repeats:,} paired, class-stratified resamples.\n{include_zero}/{len(intervals)} intervals include zero. This is not proof of equivalence, multiplicity-adjusted inference, or uncertainty from retraining.")


def nested_figure(report, output):
    nested = report["nested_selection"]
    audits = nested["fold_audits"]
    matrix = np.zeros((len(audits), len(report["development_rows"]) + len(report["excluded_test_rows"])))
    for row, audit in enumerate(audits):
        matrix[row, audit["outer_training_rows"]] = 1
        matrix[row, audit["outer_evaluation_rows"]] = 2
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    display_order = report["excluded_test_rows"] + report["development_rows"]
    excluded = len(report["excluded_test_rows"])
    image = axes[0].imshow(matrix[:, display_order], aspect="auto", interpolation="nearest", cmap=ListedColormap(["#d8dad5", "#91c6ba", ORANGE]), vmin=0, vmax=2)
    axes[0].axvline(excluded - .5, color=INK, linewidth=1)
    axes[0].set(xlabel="Display columns ordered by partition, then source row", ylabel="Repeat / outer fold",
                xticks=[(excluded - 1) / 2, excluded + (len(report['development_rows']) - 1) / 2],
                xticklabels=[f"Test: {excluded}", f"Development: {len(report['development_rows'])}"],
                yticks=range(len(audits)), yticklabels=[f"{a['repeat'] + 1} / {a['fold'] + 1}" for a in audits],
                title="01 / The original test records stay out")
    colorbar = fig.colorbar(image, ax=axes[0], orientation="horizontal", pad=.23, ticks=[0, 1, 2], shrink=.95)
    colorbar.ax.set_xticklabels(["Excluded test", "Outer fitting", "Outer evaluation"], fontsize=8)
    for repeat, color in enumerate((TEAL, PURPLE)):
        rows = [audit for audit in audits if audit["repeat"] == repeat]
        losses = [audit["selected_outer_scores"]["log_loss"] for audit in rows]
        pooled = nested["repeat_metrics"][repeat]["selected_procedure"]["log_loss"]
        axes[1].scatter(np.arange(1, len(rows) + 1) + (repeat - .5) * .1, losses, s=65, color=color, label=f"Seed {nested['seeds'][repeat]}; pooled {pooled:.4f}")
        axes[1].axhline(pooled, color=color, linestyle=":", linewidth=1.3)
    axes[1].set(xlabel="Outer fold (an index, not time)", ylabel="Outer-fold log loss (nats / record)",
                xticks=range(1, nested["outer_folds"] + 1), title="02 / Changing the split changes the estimate")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y")
    selections = '; '.join(f'{name}: {count}' for name, count in nested['selection_counts'].items())
    finish(fig, output / "nested_validation", "Evaluate the selection process, not its best inner score.",
           f"{len(report['development_rows'])} development records · {nested['outer_folds']} outer folds × {nested['inner_folds']} inner folds × {len(nested['seeds'])} repeats · {excluded} test records excluded",
           f"Source: pinned UCI WDBC; every scaler refit within its fitting partition. Selection counts: {selections}.\nDots are fold outcomes, not independent studies. Dotted lines are pooled out-of-fold loss within each repeat, not confidence limits.")


def decision_figure(report, output):
    frame = pd.DataFrame(report["decision_curves"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for repeat, color in enumerate((TEAL, PURPLE)):
        rows = frame[(frame["repeat"] == repeat) & (frame.threshold <= .5)]
        axes[0].plot(rows.threshold, rows.selected, color=color, linewidth=2, label=f"OOF selection, seed {report['nested_selection']['seeds'][repeat]}")
    rows = frame[(frame["repeat"] == 0) & (frame.threshold <= .5)]
    axes[0].plot(rows.threshold, rows.all_positive, color=ORANGE, linestyle="--", label="Classify all positive")
    axes[0].axhline(0, color=INK, linestyle=":", label="Classify all negative")
    axes[0].set(xlabel="Hypothetical decision threshold pₜ", ylabel="Net benefit (true-positive-equivalent / record)",
                title="01 / Utility needs an explicit exchange rate")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[0].grid(axis="y")
    scenario = report["prevalence_sensitivity"]
    axes[1].semilogx(scenario["prevalence"], scenario["ppv"], color=BLUE, linewidth=2.5)
    sensitivity, specificity = scenario["assumed_sensitivity"], scenario["assumed_specificity"]
    ppv = .01 * sensitivity / (.01 * sensitivity + .99 * (1 - specificity))
    axes[1].scatter(.01, ppv, color=ORANGE, s=70, zorder=3)
    axes[1].annotate(f"At an assumed 1% prevalence:\nPPV ≈ {100 * ppv:.1f}%", (.01, ppv), xytext=(12, -45), textcoords="offset points", fontsize=9)
    axes[1].set(xlabel="Hypothetical target prevalence (log scale)", ylabel="Implied positive predictive value", ylim=(0, 1.03),
                title="02 / A different base rate changes a positive result")
    axes[1].grid(axis="both")
    finish(fig, output / "decision_context", "A prediction needs a decision context before it has clinical value.",
           "Mathematical scenarios, not clinical operating policies · no threshold is recommended",
           "Left: development out-of-fold predictions; shown thresholds 0.05–0.50, full 0.05–0.80 data retained. No uncertainty bands or threshold tuning.\nRight: Bayes calculation holding test-estimated sensitivity/specificity fixed. No evidence that either transports to another population.")


def workflow_motion(report, output):
    width, height = 1360, 530
    font_path = font_manager.findfont("DejaVu Sans")
    label_font = ImageFont.truetype(font_path, 23)
    small_font = ImageFont.truetype(font_path, 19)
    title_font = ImageFont.truetype(font_path, 31)
    boxes = [(35, 205, 240, 315), (315, 100, 575, 210), (650, 100, 915, 210), (990, 100, 1325, 210),
             (315, 335, 575, 445), (650, 335, 915, 445), (990, 335, 1325, 445)]
    n_development, n_test = len(report["development_rows"]), len(report["excluded_test_rows"])
    nested = report["nested_selection"]
    labels = [f"Published cohort\n{n_development + n_test} records", f"Development only\n{n_development} records",
              f"Inner selection\n{nested['inner_folds']} folds / outer fit", f"Outer evaluation\n{nested['outer_folds']} folds × {len(nested['seeds'])} repeats",
              f"Archived test\n{n_test} fixed predictions", f"Paired resampling\n{report['paired_test_comparisons']['bootstrap_repeats']:,} draws",
              "Conditional comparison\nNot retraining uncertainty"]
    colors = [TEAL, TEAL, PURPLE, PURPLE, ORANGE, ORANGE, BLUE]
    edges = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)]
    frames = []
    for active in range(8):
        image = Image.new("RGB", (width, height), PAPER)
        draw = ImageDraw.Draw(image)
        draw.text((35, 22), "Two questions. Two separate paths through the evidence.", font=title_font, fill=INK)
        for left, right in edges:
            x1, y1, x2, y2 = boxes[left]
            a1, b1, a2, b2 = boxes[right]
            start, end = (x2 + 5, (y1 + y2) // 2), (a1 - 12, (b1 + b2) // 2)
            draw.line([start, end], fill="#aabbb4", width=4)
            draw.polygon([(end[0], end[1]), (end[0] - 10, end[1] - 6), (end[0] - 10, end[1] + 6)], fill="#aabbb4")
        for index, (box, label, color) in enumerate(zip(boxes, labels, colors)):
            selected = index == active or active == 7
            draw.rounded_rectangle(box, radius=14, fill=color if selected else "#e6eee6", outline=color, width=3)
            x1, y1, x2, y2 = box
            draw.multiline_text(((x1 + x2) / 2, (y1 + y2) / 2), label, font=label_font if index != 6 else small_font,
                                fill="white" if selected else INK, anchor="mm", align="center", spacing=8)
        draw.text((35, 486), "Protocol diagram, not a live run. The test records never enter inner or outer model fitting.", font=small_font, fill=INK)
        frames.append(image)
    output.mkdir(parents=True, exist_ok=True)
    frames[-1].save(output / "statistical_flow.png")
    frames[0].save(output / "statistical_flow.gif", save_all=True, append_images=frames[1:], duration=[500] * 7 + [900], optimize=True)


def render_statistics(report, output):
    style()
    output = Path(output)
    paired_figure(report, output)
    nested_figure(report, output)
    decision_figure(report, output)
    workflow_motion(report, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "studies/results/statistical_validation.json")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/figures")
    args = parser.parse_args()
    render_statistics(json.loads(args.report.read_text()), args.output)
