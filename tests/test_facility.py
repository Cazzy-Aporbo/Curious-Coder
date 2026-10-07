from copy import deepcopy
import json
import sqlite3

import numpy as np
import pytest

from biotech.audit_ledger import EventSigner, canonical_bytes, verify_events
from biotech.bioprocess_telemetry import FacilityStore, fixed_measurement, utc_timestamp
from biotech.airflow_dynamics import Room, simulate
from biotech.process_statistics import window_diagnostics


@pytest.fixture
def store(tmp_path):
    instance = FacilityStore(tmp_path / "facility.sqlite")
    instance.register("facility", {"facility_id": "site", "name": "Synthetic site", "h3_cell": None, "provenance": "synthetic fixture"})
    instance.register("asset", {"asset_id": "reactor", "facility_id": "site", "kind": "bioreactor", "name": "Fixture reactor", "provenance": "synthetic fixture"})
    instance.register("sensor", {"sensor_id": "ph-1", "asset_id": "reactor", "metric": "ph", "input_unit": "pH"})
    instance.calibrate("cal-1", "ph-1", "2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z", "0.02", "synthetic certificate", recorded_at="2025-01-02T00:00:00Z")
    yield instance
    instance.close()


def message(event_id="event-1", **changes):
    return {"event_id": event_id, "sensor_id": "ph-1", "observed_at": "2025-01-10T00:00:00Z", "value": "7.100", "unit": "pH", **changes}


def test_calibration_recording_time_is_not_backdated_to_validity_start(store):
    event = json.loads(store.events()[-1]["envelope"])
    assert event["recorded_at"].startswith("2025-01-02")
    assert event["payload"]["valid_from"].startswith("2025-01-01")


def test_fixed_point_and_timestamp_contract():
    assert fixed_measurement("ph", "7.1005", "pH") == (7100, 1000, "pH")
    assert fixed_measurement("ph", "7.1015", "pH") == (7102, 1000, "pH")
    assert fixed_measurement("viable_cell_density", "3.2", "10^6 cells/mL") == (3200000, 1, "cells/mL")
    assert utc_timestamp("2025-01-01T01:00:00+01:00") == utc_timestamp("2025-01-01T00:00:00Z")
    with pytest.raises(ValueError):
        utc_timestamp("2025-01-01T00:00:00")
    with pytest.raises(ValueError):
        canonical_bytes({"float": 7.1})


def test_duplicate_ingestion_preserves_one_observation_and_one_signature(store):
    first = store.ingest(message(), "2025-01-10T00:00:02Z")
    count = len(store.events())
    second = store.ingest(message(), "2025-01-10T00:00:05Z")
    assert first["status"] == "accepted" and second["replayed"]
    assert len(store.events()) == count
    with pytest.raises(ValueError, match="different content"):
        store.ingest(message(value="7.2"), "2025-01-10T00:00:05Z")
    assert store.verify(store.attestation())["valid"]


@pytest.mark.parametrize("changes, reason", [
    ({"unit": "volts"}, "VALUE_OR_UNIT_OUTSIDE_CONTRACT"),
    ({"value": "NaN"}, "VALUE_OR_UNIT_OUTSIDE_CONTRACT"),
    ({"sensor_id": "unknown"}, "UNKNOWN_SENSOR"),
    ({"observed_at": "2025-02-01T00:00:00Z"}, "NO_UNAMBIGUOUS_VALID_CALIBRATION"),
])
def test_invalid_observations_are_retained_in_quarantine(store, changes, reason):
    result = store.ingest(message(**changes), "2025-02-02T00:00:00Z")
    assert result["status"] == "quarantined"
    row = store.connection.execute("SELECT * FROM observations").fetchone()
    assert reason in json.loads(row["reasons"])
    assert store.verify(store.attestation())["valid"]


def test_rollback_covers_observation_and_signed_event(store):
    before = len(store.events())
    with pytest.raises(RuntimeError, match="Injected"):
        store.ingest(message(), "2025-01-10T00:00:01Z", fail_before_commit=True)
    assert len(store.events()) == before
    assert store.connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0


def test_calibration_overlap_and_foreign_keys_are_enforced(store):
    with pytest.raises(ValueError, match="Overlapping"):
        store.calibrate("overlap", "ph-1", "2025-01-02T00:00:00Z", "2025-01-03T00:00:00Z", "0.1", "fixture", recorded_at="2025-01-02T00:00:00Z")
    with pytest.raises(sqlite3.IntegrityError):
        store.register("sensor", {"sensor_id": "orphan", "asset_id": "missing", "metric": "ph", "input_unit": "pH"})


def test_signatures_detect_changes_and_trusted_head_detects_truncation(store):
    store.ingest(message(), "2025-01-10T00:00:02Z")
    anchor = store.attestation()
    events = store.events()
    changed = deepcopy(events)
    payload = json.loads(changed[-1]["envelope"])
    payload["payload"]["scaled_value"] = 7000
    changed[-1]["envelope"] = json.dumps(payload)
    assert not verify_events(changed, anchor["trusted_public_keys"], anchor["expected_head"])["valid"]
    assert not verify_events(events[:-1], anchor["trusted_public_keys"], anchor["expected_head"])["valid"]
    assert verify_events(events[:-1], anchor["trusted_public_keys"])["valid"]
    stranger = EventSigner()
    assert not verify_events(events, {stranger.key_id: stranger.public_key.hex()})["valid"]


def test_particle_balance_and_cumulative_size_thresholds():
    rooms = [Room("A", 100, 20, 120, 100, (100, 10)), Room("B", 100, 0, 100, 120)]
    report = simulate(rooms, [("A", "B", 1)], [[1000, 100], [0, 0]], duration_h=.1, step_h=.001)
    np.testing.assert_allclose(report["conservation_residual_by_bin"], 0, atol=1e-9)
    assert report["rooms"][1]["final_particles_m3_ge_0_5"] >= report["rooms"][1]["final_particles_m3_ge_5"]
    assert report["flows"][0]["source"] == "A"
    with pytest.raises(ValueError, match="air balance"):
        simulate([Room("bad", 100, 0, 100, 0)], [], [[1, 1]])
    with pytest.raises(ValueError, match="positivity"):
        simulate([Room("fast", 1, 0, 100, 100)], [], [[1, 1]], duration_h=1, step_h=1)


def test_particle_solver_converges_toward_the_analytic_decay_solution():
    room = Room("decay", 100, 0, 200, 200, deposition_per_h=(0., 0.))
    coarse = simulate([room], [], [[1000., 10.]], duration_h=1, step_h=.02)
    fine = simulate([room], [], [[1000., 10.]], duration_h=1, step_h=.002)
    exact = np.array([1000., 10.]) * np.exp(-2)
    assert np.linalg.norm(np.array(fine["final_total_by_bin"]) - exact) < np.linalg.norm(np.array(coarse["final_total_by_bin"]) - exact)


def test_pressure_reversal_changes_transfer_direction_without_losing_particles():
    rooms = [Room("A", 100, 0, 100, 120), Room("B", 100, 20, 120, 100)]
    report = simulate(rooms, [("A", "B", 1)], [[0, 0], [1000, 100]], duration_h=.1, step_h=.001)
    assert report["flows"][0]["source"] == "B"
    assert report["flows"][0]["target"] == "A"
    np.testing.assert_allclose(report["conservation_residual_by_bin"], 0, atol=1e-9)


def test_autocorrelation_and_sparse_counts_do_not_get_unjustified_pvalues():
    reference = np.tile([1., 2., 3., 4.], 20)
    current = np.tile([1., 2., 3., 4.], 20)
    edges = [-np.inf, 1.5, 2.5, 3.5, np.inf]
    dependent = window_diagnostics(reference, current, edges)
    assert dependent["shannon_entropy_bits"] == pytest.approx(2)
    assert dependent["chi_square_p_value"] is None
    independent = window_diagnostics(reference, current, edges, independent_samples=True)
    assert independent["chi_square_p_value"] == pytest.approx(1)
    sparse = window_diagnostics(reference[:4], current[:4], edges, independent_samples=True)
    assert sparse["chi_square_p_value"] is None
