from itertools import product

import numpy as np
import pytest

from studies.data_movement import inspect_buffers, run as memory_run
from studies.resource_scheduling import Case, check_schedule, solve


def test_numpy_tensor_copy_boundaries(tmp_path):
    path = tmp_path / "data.npy"
    np.save(path, np.arange(20, dtype=np.float32).reshape(10, 2))
    result = inspect_buffers(path)
    assert result["basic_slice_shares_memory"]
    assert not result["advanced_index_shares_memory"]
    assert not result["dtype_conversion_shares_memory"]
    assert result["tensor_shares_cpu_pointer"] and result["tensor_mutation_visible_in_view"]
    assert result["copy_on_write_preserves_disk"]


def test_spawned_reader_matches_analytic_result(tmp_path):
    report = memory_run(tmp_path / "memory.json", rows=1000)
    assert report["mapped_child_result"]["sum"] == 499500


def test_integer_schedule_matches_exhaustive_small_reference():
    cases = [Case("A", 1, 1, "team", "kit", latest_start=3, room_turnover=0, instrument_turnaround=0),
             Case("B", 1, 1, "team", "kit", latest_start=3, room_turnover=0, instrument_turnaround=0)]
    costs = []
    for first, second in product(range(4), repeat=2):
        schedule = [{"case_id": "A", "room": 0, "start": first}, {"case_id": "B", "room": 0, "start": second}]
        try:
            costs.append(check_schedule(cases, schedule, 1, 1, 5)["objective_start_slot_sum"])
        except ValueError:
            pass
    actual = solve(cases, rooms=1, recovery_capacity=1, horizon=5)
    assert actual["status"] == "optimal_for_declared_model"
    assert actual["verification"]["objective_start_slot_sum"] == min(costs)


def test_resource_conflicts_are_not_relaxed():
    cases = [Case("A", 2, 2, "one", "kit-A"), Case("B", 2, 2, "two", "kit-B")]
    with pytest.raises(ValueError, match="recovery"):
        check_schedule(cases, [{"case_id": "A", "room": 0, "start": 0}, {"case_id": "B", "room": 1, "start": 0}], 2, 1, 10)
    impossible = [Case("A", 5, 5, "one", "kit")]
    assert solve(impossible, horizon=4)["status"] == "infeasible"


@pytest.mark.parametrize("kind", ["room", "surgeon", "instrument"])
def test_each_exclusive_resource_has_its_own_reservation(kind):
    cases = [Case("A", 1, 0, "same" if kind == "surgeon" else "one", "same" if kind == "instrument" else "kit-A"),
             Case("B", 1, 0, "same" if kind == "surgeon" else "two", "same" if kind == "instrument" else "kit-B")]
    second_start = 0 if kind == "surgeon" else 1
    schedule = [{"case_id": "A", "room": 0, "start": 0},
                {"case_id": "B", "room": 0 if kind == "room" else 1, "start": second_start}]
    with pytest.raises(ValueError, match=kind):
        check_schedule(cases, schedule, 2, 1, 10)


def test_reported_end_cannot_disagree_with_the_verified_assignment():
    case = Case("A", 1, 1, "one", "kit")
    with pytest.raises(ValueError, match="end times"):
        check_schedule([case], [{"case_id": "A", "room": 0, "start": 0, "end": 9}], 1, 1, 10)
