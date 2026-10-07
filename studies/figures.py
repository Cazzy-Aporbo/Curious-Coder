"""Consistent, labeled figures generated only from recorded measurements/results."""

from pathlib import Path

import h3
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import networkx as nx
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_curve

from studies.data import clinical_data, climate_data
from studies.systems import heat_scenario, synthetic_landscape


PAPER, INK, TEAL, ORANGE, BLUE, PURPLE = "#faf8f2", "#203a3b", "#237b70", "#bc5939", "#387b9c", "#75668b"
COLORS = {"Prior baseline": "#8d9590", "Logistic regression": TEAL, "Gradient boosting": ORANGE, "Residual MLP": PURPLE}


def style():
    plt.rcParams.update({"figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
                         "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": "#c1ccc5", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlepad": 14,
                         "axes.labelsize": 10, "grid.color": "#dfe5dc", "grid.alpha": .7,
                         "svg.fonttype": "none", "svg.hashsalt": "curious-coder-evidence-v1", "figure.dpi": 140})


def finish(fig, destination, title, subtitle, source):
    fig.suptitle(title, x=.07, y=.98, ha="left", fontsize=23, fontfamily="DejaVu Serif")
    subtitle_y = .975 - 44 / (fig.get_figheight() * 72)
    fig.text(.07, subtitle_y, subtitle, fontsize=10, ha="left")
    fig.text(.07, .025, source, fontsize=8, color="#526867", ha="left")
    fig.subplots_adjust(left=.12, right=.96, top=subtitle_y - 48 / (fig.get_figheight() * 72), bottom=.17, hspace=.6, wspace=.6)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight", pad_inches=.25, metadata={"Date": None, "Title": title, "Description": subtitle + " " + source})
    svg = destination.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    fig.savefig(destination.with_suffix(".png"), dpi=160, bbox_inches="tight", pad_inches=.25, metadata={"Title": title, "Description": subtitle})
    plt.close(fig)


def cohort_figure(report, output):
    X, y = clinical_data()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    splits = report["split_indices"]
    labels = ["Training", "Validation", "Held-out test"]
    for position, name in enumerate(("train", "validation", "test")):
        rows = splits[name]
        negative, positive = int((y[rows] == 0).sum()), int(y[rows].sum())
        axes[0].barh(position, negative, color=TEAL, label="Benign" if position == 0 else None)
        axes[0].barh(position, positive, left=negative, color=ORANGE, label="Malignant" if position == 0 else None)
        axes[0].text(negative / 2, position, str(negative), color="white", ha="center", va="center", weight="bold")
        axes[0].text(negative + positive / 2, position, str(positive), color="white", ha="center", va="center", weight="bold")
        axes[0].text(negative + positive + 5, position, f"n = {len(rows)}", va="center", fontsize=9)
    axes[0].set(yticks=range(3), yticklabels=labels, xlabel="Records (count)", xlim=(0, 395), title="01 / Reserve evidence before fitting")
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right", frameon=False)
    features = ["mean radius", "mean perimeter", "mean area", "mean concavity", "mean texture"]
    matrix = X.iloc[splits["train"]][features].corr(method="spearman").to_numpy()
    image = axes[1].imshow(matrix, cmap="BrBG", vmin=-1, vmax=1)
    short = ["Radius", "Perimeter", "Area", "Concavity", "Texture"]
    axes[1].set(xticks=range(5), yticks=range(5), xticklabels=short, yticklabels=short,
                title="02 / Correlated measurements")
    axes[1].tick_params(axis="x", rotation=35)
    for row in range(5):
        for col in range(5):
            axes[1].text(col, row, f"{matrix[row, col]:.3f}", ha="center", va="center", fontsize=9,
                         color="white" if abs(matrix[row, col]) > .65 else INK)
    fig.colorbar(image, ax=axes[1], shrink=.75, label="Training Spearman correlation (ρ)")
    finish(fig, output / "cohort", "Before the model, examine the measurement.",
           "569 digitized fine-needle-aspirate records · 30 numerical features · positive label: malignant · split seed 42",
           "Source: UCI WDBC, DOI 10.24432/C5DW2B (CC BY 4.0). Counts and correlations computed from the pinned snapshot.\nNo missing feature values; no site, date, or demographic fields for external-validity auditing.")


def diagnostic_figure(report, predictions, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    names = list(report["test_evaluation"])
    for column, (metric, axis_label) in enumerate((("roc_auc", "Test ROC AUC (higher is better)"), ("brier", "Test Brier score (lower is better)"))):
        ax = axes[0, column]
        for row, name in enumerate(names):
            entry = report["test_evaluation"][name]
            score, interval = entry["test"][metric], entry["bootstrap_95"][metric]
            ax.plot(interval, [row, row], color=COLORS[name], linewidth=3, solid_capstyle="round")
            ax.scatter(score, row, color=COLORS[name], s=65, zorder=3)
            ax.annotate(f"{score:.3f}", (score, row), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
        ax.set(yticks=range(len(names)), yticklabels=names, xlabel=axis_label, ylim=(-.7, len(names) - .2))
        ax.invert_yaxis()
        ax.grid(axis="x")
        ax.set_title("01 / Discrimination" if column == 0 else "02 / Probability error")
    y = predictions.malignant.to_numpy()
    for name in names:
        false_positive, true_positive, _ = roc_curve(y, predictions[name])
        axes[1, 0].plot(false_positive, true_positive, color=COLORS[name], label=name, linewidth=1.8)
    axes[1, 0].plot([0, 1], [0, 1], "--", color="#8d9590", linewidth=1)
    axes[1, 0].set(xlabel="False-positive rate", ylabel="True-positive rate (sensitivity)", xlim=(0, 1), ylim=(0, 1.03), title="03 / Thresholds change the error trade-off")
    axes[1, 0].legend(frameon=False, fontsize=8, loc="lower right")
    name = report["selected_by_validation_log_loss"]
    p = predictions[name].to_numpy()
    ax = axes[1, 1]
    ax.plot([0, 1], [0, 1], linestyle="--", color="#8d9590", label="Perfect calibration")
    assignments = np.minimum((p * 5).astype(int), 4)
    z = norm.ppf(.975)
    for bucket in range(5):
        mask = assignments == bucket
        n = int(mask.sum())
        if n == 0:
            continue
        observed, predicted = y[mask].mean(), p[mask].mean()
        center = (observed + z * z / (2 * n)) / (1 + z * z / n)
        radius = z * np.sqrt(observed * (1 - observed) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        ax.plot([predicted, predicted], [center - radius, center + radius], color=TEAL, linewidth=2)
        ax.scatter(predicted, observed, s=35 + n, color=TEAL, edgecolor=PAPER, zorder=3)
        ax.annotate(f"n={n}", (predicted, observed), xytext=(5, 8 if observed < .9 else -16), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Mean predicted P(malignant) per bin", ylabel="Observed malignant fraction", xlim=(-.03, 1.03), ylim=(-.05, 1.05), title="04 / Calibration and sample support")
    finish(fig, output / "diagnostics", "Ranking well is not the same as knowing the risk.",
           f"Held-out test: n={len(y)} · selected by validation log loss: {name} · no test-set tuning",
           f"Source: pinned UCI WDBC snapshot. Top intervals: {report['bootstrap_repeats']} stratified bootstrap resamples of fixed-model test predictions.\nCalibration bars: 95% Wilson intervals in 5 fixed-width bins. Neither interval captures retraining or hospital-shift uncertainty.")


def training_figure(report, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    history = report["training_history"]
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_log_loss"] for row in history], color=PURPLE, label="Training (dropout active)", linewidth=2)
    axes[0].plot(epochs, [row["validation_log_loss"] for row in history], color=ORANGE, label="Validation (dropout disabled)", linewidth=2)
    axes[0].axvline(report["mlp_best_epoch"], color=INK, linestyle="--", linewidth=1, label=f"Restored epoch {report['mlp_best_epoch']}")
    axes[0].set(xlabel="Training epoch", ylabel="Binary cross-entropy (nats / record)", title="01 / Restore the validation checkpoint")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y")
    importance = report["validation_permutation_importance"]
    selected = np.argsort(importance["mean"])[-8:]
    axes[1].barh(range(8), np.array(importance["mean"])[selected], xerr=np.array(importance["std"])[selected],
                 color=TEAL, alpha=.9, error_kw={"ecolor": INK, "capsize": 2, "linewidth": 1})
    axes[1].set(yticks=range(8), yticklabels=np.array(importance["features"])[selected],
                xlabel="Increase in validation log loss (nats / record)", title="02 / Baseline feature reliance")
    axes[1].axvline(0, color=INK, linewidth=.8)
    finish(fig, output / "training", "Complexity has to earn its place.",
           f"Residual MLP: {report['mlp_parameter_count']:,} parameters · LayerNorm · AdamW · gradient clipping · cosine schedule · early stopping",
           "Source: UCI WDBC. All transforms fit on training rows. Importance: logistic regression, validation split, 15 repeats; bars ±1 SD.\nCorrelated features can substitute for one another: permutation importance is model reliance, not biological causation.")


def climate_figure(output):
    frame, station = climate_data()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.7), gridspec_kw={"height_ratios": [1, 1.3]})
    axes[0].plot(frame.index, frame.TMAX, color=ORANGE, linewidth=.7, alpha=.75, label="Daily maximum air temperature")
    axes[0].plot(frame.index, frame.TMIN, color=BLUE, linewidth=.7, alpha=.75, label="Daily minimum air temperature")
    axes[0].set(ylabel="Air temperature (°C)", xlabel="Observation date", title="01 / Two years of observations, not a street-level heat map")
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    axes[0].grid(axis="y")
    calendar_matrix = np.full((12, 31), np.nan)
    for timestamp, row in frame.loc["2024"].iterrows():
        calendar_matrix[timestamp.month - 1, timestamp.day - 1] = row.TMAX
    cmap = LinearSegmentedColormap.from_list("air-temperature", ["#deecea", "#efd79b", ORANGE, "#743a35"])
    cmap.set_bad("#e6e5df")
    image = axes[1].imshow(calendar_matrix, aspect="auto", cmap=cmap, vmin=10, vmax=50)
    axes[1].set(yticks=range(12), yticklabels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                xticks=[0, 4, 9, 14, 19, 24, 30], xticklabels=[1, 5, 10, 15, 20, 25, 31],
                xlabel="Day of month in 2024", ylabel="Calendar month", title="02 / Daily maximum temperature, with the calendar left visible")
    fig.colorbar(image, ax=axes[1], label="Daily maximum air temperature (°C)", pad=.02)
    missing = frame.isna().sum().to_dict()
    finish(fig, output / "climate", "A measured boundary condition, not a modeled city.",
           f"NOAA/NCEI · {station['name']} · {station['id']} · {station['latitude']:.4f}° N, {abs(station['longitude']):.4f}° W",
           f"Source: GHCN-Daily, DOI 10.7289/V5D21VHZ; 2023–2024 snapshot. Rejected/missing daily values: TMAX={missing['TMAX']}, TMIN={missing['TMIN']}.\nNonblank quality flags are excluded, not imputed. Gray calendar cells are absent dates or missing values. No homogenization or causal canopy inference.")


def spatial_figure(output):
    climate, station = climate_data()
    latitude, longitude = station["latitude"], station["longitude"]
    landscape = synthetic_landscape(latitude, longitude)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4))
    polygons = []
    for record in landscape:
        boundary = h3.cell_to_boundary(record["cell"])
        xy = [((lon - longitude) * 111320 * np.cos(np.deg2rad(latitude)), (lat - latitude) * 111320) for lat, lon in boundary]
        polygons.append(Polygon(xy))
    collection = PatchCollection(polygons, cmap="YlGn", edgecolor=PAPER, linewidth=1, clim=(0, 1))
    collection.set_array(np.array([record["canopy_fraction"] for record in landscape]))
    axes[0].add_collection(collection)
    axes[0].autoscale_view()
    axes[0].set_aspect("equal")
    axes[0].scatter(0, 0, marker="+", s=90, color=INK)
    axes[0].annotate("Station", (0, 0), xytext=(6, 6), textcoords="offset points", fontsize=8)
    axes[0].set(xlabel="Approximate east offset (m)", ylabel="Approximate north offset (m)", title="01 / H3 resolution 9; scenario canopy")
    fig.colorbar(collection, ax=axes[0], label="Scenario canopy fraction (0–1)", shrink=.7)
    air = float(climate.loc["2024-06-01":"2024-08-31", "TMAX"].mean())
    canopy, albedo = np.linspace(0, 1, 51), np.linspace(.1, .7, 51)
    values = np.array([[heat_scenario(air, c, a) for c in canopy] for a in albedo])
    image = axes[1].imshow(values, origin="lower", extent=(0, 1, .1, .7), aspect="auto", cmap="YlOrRd")
    axes[1].set(xlabel="Assumed canopy fraction", ylabel="Assumed surface albedo", title="02 / Assumed surface heat balance")
    fig.colorbar(image, ax=axes[1], label="Scenario surface temperature (°C)", shrink=.7)
    axes[2].add_patch(Rectangle((0, -2), 1, 1, facecolor=TEAL, alpha=.8, label="Proposed asset (synthetic)"))
    axes[2].add_patch(Rectangle((1.2, -2), .8, 1, facecolor=ORANGE, alpha=.8, label="Nominal utility envelope"))
    axes[2].add_patch(Rectangle((.9, -2.3), 1.4, 1.6, facecolor="none", edgecolor=ORANGE, hatch="///", label="±0.3 m assumed uncertainty"))
    axes[2].axhline(0, color=INK, linewidth=1)
    axes[2].set(xlim=(-.4, 2.7), ylim=(-2.8, .3), xlabel="Local horizontal coordinate x (m)", ylabel="Local elevation z (m; ground = 0)", title="03 / Possible utility conflict")
    axes[2].legend(loc="upper right", frameon=True, facecolor=PAPER, edgecolor="none", fontsize=7)
    finish(fig, output / "spatial", "Index the place. Preserve the uncertainty.",
           f"Measured summer-2024 station mean TMAX: {air:.1f} °C · all canopy, albedo, material, and utility geometry below are synthetic",
           "H3 coordinates from the pinned NOAA station location. East/north offsets use a local approximation, not a survey CRS.\nHeat scenario assumes 700 W/m² solar forcing and 25 W/(m²·K) heat transfer; no evapotranspiration, radiation exchange, or observed street-temperature validation.")


def architecture_figure(output):
    graph = nx.DiGraph()
    positions = {"Client\nrequest key": (0, .55), "Gateway\nidentity + tenant policy": (1.4, .55),
                 "Mutation service\nvalidate intent": (3, .55), "Atomic transaction\nasset + event + key": (4.8, .55),
                 "Durable journal\nreplay + anchored hash": (4.8, -.25), "Async projection\nH3 tiles + cache": (3, -.25),
                 "Read API / CDN\nversioned, stale-tolerant": (1.4, -.25)}
    edges = [(list(positions)[i], list(positions)[i + 1]) for i in range(6)]
    graph.add_edges_from(edges)
    fig, ax = plt.subplots(figsize=(14, 5.6))
    labels = {}
    for node, (x, y) in positions.items():
        labels[node] = ax.text(x, y, node, ha="center", va="center", fontsize=10, linespacing=1.7,
                               bbox={"boxstyle": "round,pad=.65", "facecolor": "#e5eee5", "edgecolor": "#8aaa9b"})
    for source, target in graph.edges:
        ax.annotate("", xy=positions[target], xytext=positions[source],
                    arrowprops={"arrowstyle": "-|>", "color": "#728c83", "lw": 1.8,
                                "patchA": labels[source].get_bbox_patch(), "patchB": labels[target].get_bbox_patch(),
                                "shrinkA": 6, "shrinkB": 6, "mutation_scale": 18})
    ax.set(xlim=(-.7, 5.65), ylim=(-.75, 1.05))
    ax.axis("off")
    finish(fig, output / "architecture", "A retry is ordinary. A duplicate mutation is a bug.",
           "Reference architecture · solid arrows show intended flow · only the local SQLite transaction and replay contracts are implemented here",
           "The authorization boundary precedes tenant-scoped idempotency. The journal is not a queue acknowledgement and does not make external side effects atomic.\nDeployment, identity provider, broker, outbox relay, replicas, CDN, and regional recovery remain explicit design work—not claimed infrastructure.")


def render_all(report, predictions, output):
    style()
    output = Path(output)
    cohort_figure(report, output)
    diagnostic_figure(report, predictions, output)
    training_figure(report, output)
    climate_figure(output)
    spatial_figure(output)
    architecture_figure(output)
