"""Show the retrieval trade-offs and the selection process that produced them."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from studies.data import ROOT
from studies.figures import BLUE, INK, ORANGE, PAPER, PURPLE, TEAL, finish, style


COLORS = {"TF-IDF": BLUE, "MiniLM FP32": TEAL, "MiniLM INT8-linear": PURPLE, "Hybrid RRF": ORANGE}
SHORT = {"TF-IDF": "TF-IDF", "MiniLM FP32": "FP32 encoder", "MiniLM INT8-linear": "INT8-linear", "Hybrid RRF": "Hybrid RRF"}
QUERY_LABELS = {"reporting": "Reporting guidance", "selection_bias": "Selection bias", "decision_utility": "Decision utility",
                "protein_model": "Protein structure", "read_encoding": "Read encoding", "assay_window": "Assay controls"}


def tradeoff_figure(report, output):
    methods = list(report["methods"])
    queries = list(dict.fromkeys(row["query_key"] for row in report["query_results"]))
    ranks = np.array([[next(row["target_rank"] for row in report["query_results"] if row["method"] == method and row["query_key"] == query)
                       for method in methods] for query in queries])
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    cmap = LinearSegmentedColormap.from_list("rank-quality", ["#e4f0e7", "#b9a6ca", "#793d68"])
    image = axes[0, 0].imshow(ranks, cmap=cmap, vmin=1, vmax=ranks.max(), aspect="auto")
    axes[0, 0].set(xticks=range(len(methods)), xticklabels=[SHORT[name] for name in methods],
                   yticks=range(len(queries)), yticklabels=[QUERY_LABELS[key] for key in queries], title="01 / Which known paper reaches the reader?")
    axes[0, 0].tick_params(axis="x", rotation=20)
    for row in range(len(queries)):
        for column in range(len(methods)):
            axes[0, 0].text(column, row, str(ranks[row, column]), ha="center", va="center", weight="bold",
                            color="white" if ranks[row, column] > ranks.max() / 2 else INK)
    fig.colorbar(image, ax=axes[0, 0], label="Known-target rank (lower is better)", shrink=.85)
    for name in methods:
        values = report["methods"][name]
        samples = np.sort(values["raw_warm_query_ms"])
        axes[0, 1].semilogx(samples, np.arange(1, len(samples) + 1) / len(samples), color=COLORS[name], linewidth=2,
                            label=f"{SHORT[name]}: p95 {values['warm_query_p95_ms']:.2f} ms")
    axes[0, 1].set(xlabel="Warm query wall time (ms, log scale)", ylabel="Fraction of measured requests ≤ time",
                   ylim=(0, 1.04), title="02 / Look beyond the median")
    axes[0, 1].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0, 1].grid(axis="both")
    for index, name in enumerate(methods):
        mib = report["methods"][name]["serialized_model_state_bytes"] / 2 ** 20
        axes[1, 0].bar(index, mib, color=COLORS[name], width=.6)
        axes[1, 0].text(index, mib + 2, f"{mib:.1f}" if mib else "No neural\nweights", ha="center", fontsize=9)
        result = report["methods"][name]
        axes[1, 1].scatter(result["warm_query_p50_ms"], result["mean_anchor_reciprocal_rank"], color=COLORS[name], s=100, edgecolor=PAPER, zorder=3)
        offsets = {"TF-IDF": (6, -20), "MiniLM FP32": (-95, 16), "MiniLM INT8-linear": (7, -16), "Hybrid RRF": (7, -20)}
        axes[1, 1].annotate(SHORT[name], (result["warm_query_p50_ms"], result["mean_anchor_reciprocal_rank"]),
                            xytext=offsets[name], textcoords="offset points", fontsize=9)
    axes[1, 0].set(xticks=range(len(methods)), xticklabels=[SHORT[name] for name in methods],
                   ylabel="Serialized model-state size (MiB)", ylim=(0, 105), title="03 / Smaller weights are not total memory")
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 1].set(xscale="log", xlabel="Median warm query time (ms, log scale)", ylabel="Mean known-target reciprocal rank", ylim=(0, 1.04),
                   title="04 / The trade-off depends on the task")
    axes[1, 1].grid(axis="both")
    environment = report["environment"]
    count = report["methods"][methods[0]]["timed_requests"]
    finish(fig, output / "retrieval_tradeoffs", "A more expensive representation has to improve the retrieval task.",
           f"{report['source_audit']['records']} retained citations · {len(queries)} authored known-item questions · {count} timed requests per method · {environment['system']} {environment['machine']}, one PyTorch thread",
           "Source: recorded Europe PMC snapshot and pinned MiniLM checkpoint. Anchors were deliberately included; these are not complete relevance labels.\nWarm timing excludes network, cold start, concurrency, and review. Model-state size excludes index, metadata, allocator, and process memory.")


def source_figure(report, output):
    audit = report["source_audit"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.7))
    counts = [audit["abstract_included"], audit["abstract_withheld_by_reuse_policy"], audit["abstract_missing_upstream"]]
    labels = ["Abstract retained\nCC BY / CC0 metadata", "Abstract withheld\nReuse policy", "Abstract absent\nUpstream metadata"]
    for position, (count, label, color) in enumerate(zip(counts, labels, (TEAL, PURPLE, "#929d97"))):
        axes[0].barh(position, count, color=color)
        axes[0].text(count + 2, position, f"{count} / {audit['records']}", va="center", fontsize=10)
    axes[0].set(yticks=range(3), yticklabels=labels, xlabel="Records (count)", xlim=(0, max(counts) * 1.35),
                title="01 / Available text is not uniformly reusable")
    axes[0].invert_yaxis()
    years, values = list(audit["years"]), list(audit["years"].values())
    axes[1].bar(range(len(years)), values, color=[ORANGE if year == "2026" else BLUE for year in years])
    for index, count in enumerate(values):
        axes[1].text(index, count + 2, str(count), ha="center", fontsize=9)
    axes[1].set(xticks=range(len(years)), xticklabels=years, ylabel="Retained records (count)", xlabel="Europe PMC pubYear field",
                ylim=(0, max(values) * 1.18), title="02 / The acquisition window shapes the evidence")
    axes[1].grid(axis="y")
    finish(fig, output / "retrieval_source_audit", "Selection happens before the model sees a document.",
           f"{audit['records']} records · languages: {', '.join(f'{key}={value}' for key, value in audit['languages'].items())} · {report['input_audit']['records_truncated_at_256_wordpieces']} neural inputs exceed 256 wordpieces",
           "Source: three capped Europe PMC API queries plus six explicit anchors. Missing abstracts and withheld abstracts are different states.\nThe query filters FIRST_PDATE; the chart displays pubYear. Online-first and issue years can differ. The corpus is not a systematic review.")


def render(report, output):
    style()
    output = Path(output)
    tradeoff_figure(report, output)
    source_figure(report, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "studies/results/evidence_retrieval.json")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/figures")
    args = parser.parse_args()
    render(json.loads(args.report.read_text()), args.output)
