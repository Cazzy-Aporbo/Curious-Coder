"""Make the resource constraint visible beside the synthetic schedule."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from studies.data import ROOT
from studies.figures import BLUE, ORANGE, PURPLE, TEAL, finish, style


def render(report, output):
    style()
    cases = {case["case_id"]: case for case in report["cases"]}
    colors = dict(zip(cases, (TEAL, PURPLE, ORANGE, BLUE)))
    minutes = report["slot_minutes"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    for row, scenario in enumerate(report["scenarios"].values()):
        if scenario["status"] != "optimal_for_declared_model":
            raise ValueError("Only verified optimal fixture schedules belong in this comparison.")
        occupancy = np.zeros(scenario["horizon"] + 1)
        for assignment in scenario["schedule"]:
            case = cases[assignment["case_id"]]
            start, end = assignment["start"], assignment["end"]
            axes[row, 0].barh(assignment["room"], (end - start) * minutes, left=start * minutes,
                              color=colors[case["case_id"]], height=.55)
            axes[row, 0].text((start + end) * minutes / 2, assignment["room"], case["case_id"].removeprefix("case-"),
                              ha="center", va="center", color="white", weight="bold")
            axes[row, 0].barh(assignment["room"], case["room_turnover"] * minutes, left=end * minutes,
                              color="#ece3e9", edgecolor="#7d6579", hatch="///", height=.55)
            occupancy[end:assignment["recovery_end"]] += 1
        axes[row, 0].set(yticks=range(scenario["rooms"]), yticklabels=[f"OR {index + 1}" for index in range(scenario["rooms"])],
                         xlabel="Model time (minutes)", xlim=(0, scenario["horizon"] * minutes),
                         title=f"{scenario['recovery_capacity']} recovery place(s) · start-slot sum {scenario['verification']['objective_start_slot_sum']}")
        axes[row, 1].step(np.arange(len(occupancy)) * minutes, occupancy, where="post", color=PURPLE, linewidth=2)
        axes[row, 1].axhline(scenario["recovery_capacity"], color=ORANGE, linestyle="--", label="Declared capacity")
        axes[row, 1].set(xlabel="Model time (minutes)", ylabel="Concurrent recovery reservations", ylim=(0, 2.5),
                         yticks=[0, 1, 2], xlim=(0, scenario["horizon"] * minutes), title="The downstream reservation must also fit")
        axes[row, 1].legend(frameon=False, fontsize=8)
        axes[row, 1].grid(axis="y")
    handles = [Patch(facecolor=color, label=key) for key, color in colors.items()]
    handles.append(Patch(facecolor="#ece3e9", hatch="///", edgecolor="#7d6579", label="Room turnover"))
    axes[0, 0].legend(handles=handles, fontsize=8, frameon=False, ncol=3, loc="upper right")
    finish(fig, Path(output) / "resource_scheduling", "The next constraint may sit beyond the operating room.",
           "Synthetic cases · two operating rooms held fixed · recovery capacity changes from one to two · 15-minute model slots",
           "Source: recorded SciPy/HiGHS fixture. Procedure bars and hatched turnover reserve OR capacity; recovery occupies its own time interval.\nSurgeon and instrument constraints also apply. No patient urgency, clinical staffing standard, stochastic-duration model, or live hospital feed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "studies/results/resource_scheduling.json")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/figures")
    args = parser.parse_args()
    render(json.loads(args.report.read_text()), args.output)
