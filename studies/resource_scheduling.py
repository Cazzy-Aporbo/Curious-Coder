"""A synthetic OR/PACU resource proof, not a patient-priority or clinical scheduler."""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from studies.data import ROOT


@dataclass(frozen=True)
class Case:
    case_id: str
    duration: int
    recovery: int
    surgeon: str
    instrument_set: str
    earliest: int = 0
    latest_start: int = 10
    room_turnover: int = 1
    instrument_turnaround: int = 2

    def __post_init__(self):
        integers = (self.duration, self.recovery, self.earliest, self.latest_start, self.room_turnover, self.instrument_turnaround)
        if any(type(value) is not int or value < 0 for value in integers) or self.duration < 1 or self.latest_start < self.earliest:
            raise ValueError("Use ordered, nonnegative integer slots and a positive procedure duration.")
        if not all(isinstance(value, str) and value.strip() for value in (self.case_id, self.surgeon, self.instrument_set)):
            raise ValueError("Case and resource identifiers are required.")


def reservations(case, room, start):
    end = start + case.duration
    return [("room", room, slot) for slot in range(start, end + case.room_turnover)] + [
        ("surgeon", case.surgeon, slot) for slot in range(start, end)] + [
        ("instrument", case.instrument_set, slot) for slot in range(start, end + case.instrument_turnaround)] + [
        ("recovery", "PACU", slot) for slot in range(end, end + case.recovery)]


def check_schedule(cases, schedule, rooms, recovery_capacity, horizon):
    lookup = {case.case_id: case for case in cases}
    if len(lookup) != len(cases) or Counter(row["case_id"] for row in schedule) != Counter(lookup.keys()):
        raise ValueError("Every declared case must occur exactly once.")
    occupancy = Counter()
    for row in schedule:
        case, start, room = lookup[row["case_id"]], row["start"], row["room"]
        if type(start) is not int or type(room) is not int or not 0 <= room < rooms or not case.earliest <= start <= case.latest_start:
            raise ValueError("Invalid assignment or start window.")
        expected_end = start + case.duration
        if row.get("end", expected_end) != expected_end or row.get("recovery_end", expected_end + case.recovery) != expected_end + case.recovery:
            raise ValueError("Reported end times differ from the case contract.")
        busy = reservations(case, room, start)
        if any(slot >= horizon for _, _, slot in busy):
            raise ValueError("A resource reservation crosses the modeled horizon.")
        occupancy.update(busy)
    for (kind, _, _), count in occupancy.items():
        if count > (recovery_capacity if kind == "recovery" else 1):
            raise ValueError(f"Resource capacity exceeded: {kind}")
    return {"feasible": True, "objective_start_slot_sum": sum(row["start"] for row in schedule),
            "peak_recovery_occupancy": max((count for (kind, _, _), count in occupancy.items() if kind == "recovery"), default=0)}


def solve(cases, rooms=2, recovery_capacity=1, horizon=14):
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Require nonempty, uniquely identified synthetic cases.")
    if any(type(value) is not int or value < 1 for value in (rooms, recovery_capacity, horizon)) or horizon > 200 or len(cases) > 50 or rooms > 10:
        raise ValueError("Resource counts must be positive and the teaching workload bounded.")
    placements = [(case, room, start) for case in cases for room in range(rooms)
                  for start in range(case.earliest, min(case.latest_start, horizon - case.duration - max(case.recovery, case.room_turnover, case.instrument_turnaround)) + 1)]
    entries = sum(1 + 3 * case.duration + case.room_turnover + case.instrument_turnaround + case.recovery for case, _, _ in placements)
    if len(placements) > 10000 or entries > 2000000:
        raise ValueError("The time-indexed formulation exceeds the declared teaching memory budget.")
    if any(not any(placement[0] == case for placement in placements) for case in cases):
        return {"status": "infeasible", "reason": "A case has no placement within its resource horizon."}
    keys = [("case", case.case_id, 0) for case in cases]
    keys += sorted({key for case, room, start in placements for key in reservations(case, room, start)}, key=str)
    row_lookup = {key: index for index, key in enumerate(keys)}
    row_indices, columns = [], []
    for column, (case, room, start) in enumerate(placements):
        occupied = [("case", case.case_id, 0), *reservations(case, room, start)]
        row_indices.extend(row_lookup[key] for key in occupied)
        columns.extend([column] * len(occupied))
    matrix = coo_matrix((np.ones(len(columns)), (row_indices, columns)), shape=(len(keys), len(placements))).tocsc()
    lower = np.array([1 if kind == "case" else 0 for kind, _, _ in keys], dtype=float)
    upper = np.array([recovery_capacity if kind == "recovery" else 1 for kind, _, _ in keys], dtype=float)
    started = perf_counter()
    result = milp(np.array([start for _, _, start in placements], dtype=float), integrality=np.ones(len(placements)), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, lower, upper), options={"time_limit": 10, "mip_rel_gap": 0})
    elapsed = 1000 * (perf_counter() - started)
    if result.status != 0:
        return {"status": "infeasible" if result.status == 2 else "not_proven_optimal", "solver_message": result.message, "solve_ms": elapsed}
    schedule = [{"case_id": case.case_id, "room": room, "start": start, "end": start + case.duration,
                 "recovery_end": start + case.duration + case.recovery} for selected, (case, room, start) in zip(result.x, placements) if selected > .5]
    verified = check_schedule(cases, schedule, rooms, recovery_capacity, horizon)
    return {"status": "optimal_for_declared_model", "schedule": sorted(schedule, key=lambda row: (row["start"], row["room"])),
            "verification": verified, "solve_ms": elapsed, "binary_variables": len(placements), "constraint_rows": len(keys),
            "solver_gap": float(result.mip_gap), "rooms": rooms, "recovery_capacity": recovery_capacity, "horizon": horizon}


def run(output=ROOT / "studies/results/resource_scheduling.json"):
    cases = [Case("case-A", 2, 2, "team-1", "tray-1"), Case("case-B", 2, 2, "team-2", "tray-2"),
             Case("case-C", 1, 2, "team-1", "tray-1", earliest=1), Case("case-D", 2, 1, "team-2", "tray-2")]
    report = {"evidence_type": "synthetic capacity example; no patient data or clinical priority model", "slot_minutes": 15,
              "cases": [asdict(case) for case in cases], "scenarios": {"one_recovery_place": solve(cases, recovery_capacity=1),
                  "two_recovery_places": solve(cases, recovery_capacity=2)},
              "limits": "Deterministic durations and simplified availability delays; no live ADT/FHIR feed, emergency triage, staffing-safety validation, sterilization validation, or EHR writes."}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "studies/results/resource_scheduling.json")
    run(parser.parse_args().output)
