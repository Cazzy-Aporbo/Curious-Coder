"""Trace the synthetic facility's data decisions and conserved particle inventory."""

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
import numpy as np

from studies.data import ROOT
from studies.figures import BLUE, ORANGE, PURPLE, TEAL, finish, style


def render(report, output):
    style()
    rows = [row for row in report["observations"] if row["declared_sensor"] == "PH-01" and row["status"] == "accepted"]
    values = np.array([row["scaled_value"] / row["value_scale"] for row in rows])
    drift = report["drift"]
    windows = drift["windows"]
    ends = np.array([window["ending_observation"] for window in windows])
    baseline = float(np.median(values[:drift["reference_count"]]))
    medians = baseline + np.array([window["median_shift"] for window in windows])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.8))
    axes[0, 0].axvspan(0, drift["reference_count"] - 1, color="#dcece4", label="Frozen baseline")
    axes[0, 0].plot(values, color=BLUE, linewidth=1, alpha=.65, label="Accepted pH observation")
    axes[0, 0].plot(ends, medians, color=TEAL, linewidth=2, label=f"{drift['window_size']}-observation median")
    limit = baseline + 4 * windows[0]["reference_mad_scale"]
    axes[0, 0].axhline(limit, color=ORANGE, linestyle="--", label="Example +4 MAD-scale review bound")
    axes[0, 0].set(xlabel="Synthetic observation index (one minute apart)", ylabel="pH", title="01 / Valid input can still require process review")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].plot(ends, [window["shannon_entropy_bits"] for window in windows], color=PURPLE, linewidth=2)
    axes[0, 1].set(xlabel="Ending observation index", ylabel="Shannon entropy of fixed bins (bits)", ylim=(-.05, 2.1),
                   title="02 / Entropy is a description, not a diagnosis")
    axes[0, 1].text(.03, .07, "Chi-square p-values withheld:\nserial-window independence is not established.", transform=axes[0, 1].transAxes, fontsize=9)
    air = report["airflow"]
    available = float(sum(air["initial_total_by_bin"]) + sum(air["generated_total_by_bin"]))
    removed, retained = float(sum(air["removed_total_by_bin"])), float(sum(air["final_total_by_bin"]))
    Sankey(ax=axes[1, 0], scale=1 / available, unit=" particles", format="%.0f", gap=.6).add(
        flows=[available, -removed, -retained], labels=["Available", "Removed", "Remaining"],
        orientations=[0, 0, -1], pathlengths=[.25, .25, .4], facecolor="#8dc6b6", edgecolor=TEAL).finish()
    axes[1, 0].set_aspect("equal", adjustable="box")
    axes[1, 0].set_title("03 / Close the particle inventory")
    axes[1, 0].axis("off")
    reasons = Counter(reason for row in report["observations"] for reason in json.loads(row["reasons"]))
    labels = {"VALUE_OR_UNIT_OUTSIDE_CONTRACT": "Unit/value contract", "UNKNOWN_SENSOR": "Unknown sensor",
              "NO_UNAMBIGUOUS_VALID_CALIBRATION": "Calibration not valid", "OBSERVATION_AFTER_RECEIPT": "Clock ordering"}
    axes[1, 1].barh([labels[key] for key in reasons], list(reasons.values()), color=ORANGE)
    axes[1, 1].set(xlabel="Quarantined records (count)", xlim=(0, max(reasons.values()) + 1), title="04 / Preserve the reason, not just the flag")
    axes[1, 1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    for axis in (axes[0, 0], axes[0, 1]):
        axis.grid(axis="y")
    finish(fig, Path(output) / "facility_evidence", "Follow the observation all the way to the review decision.",
           f"Synthetic fixture · {report['counts']['accepted']} accepted readings · {report['counts']['quarantined']} quarantined · {report['audit_verification']['events']} signed events",
           "No real bioreactor or cleanroom data. Accepted means the input contract passed, not that the process is in control.\nParticle accounting combines two disjoint size bins over 30 simulated minutes; no viable-organism or resolved-airflow inference is made.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "studies/results/facility.json")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/figures")
    args = parser.parse_args()
    render(json.loads(args.report.read_text()), args.output)
