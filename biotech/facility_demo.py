"""Execute a synthetic facility's ingestion, audit, and particle-balance workflow."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import h3
import numpy as np

from biotech.airflow_dynamics import Room, simulate
from biotech.bioprocess_telemetry import FacilityStore, METRICS, projection_digest
from biotech.facility_api import inspect_asset
from biotech.process_statistics import rolling_diagnostics


ROOT = Path(__file__).resolve().parents[1]


def run(directory=ROOT / "artifacts/facility-demo", report_path=ROOT / "studies/results/facility.json"):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Use a new --directory; existing databases and attestations are not replaced.")
    directory.mkdir(parents=True, exist_ok=True)
    store = FacilityStore(directory / "facility.sqlite")
    try:
        store.register("facility", {"facility_id": "site-demo", "name": "Synthetic process-development suite", "h3_cell": h3.latlng_to_cell(33.4278, -112.0039, 9),
                                    "provenance": "Synthetic geographic anchor; not a surveyed or company-operated facility."})
        for identifier, kind, name in (("BR-01", "bioreactor", "Bioreactor fixture"), ("suite-a", "room", "Suite A"), ("suite-b", "room", "Suite B"), ("suite-c", "room", "Suite C"), ("HVAC-01", "utility", "Air-handling fixture")):
            store.register("asset", {"asset_id": identifier, "facility_id": "site-demo", "kind": kind, "name": name, "provenance": "synthetic engineering fixture"})
        sensors = {"PH-01": "ph", "DO-01": "dissolved_oxygen", "RPM-01": "agitation", "VCD-01": "viable_cell_density", "TEMP-EXPIRED": "temperature"}
        for sensor, metric in sensors.items():
            store.register("sensor", {"sensor_id": sensor, "asset_id": "BR-01", "metric": metric, "input_unit": METRICS[metric]["input_unit"]})
            until = "2025-01-05T00:00:00Z" if sensor == "TEMP-EXPIRED" else "2025-02-01T00:00:00Z"
            store.calibrate(f"CAL-{sensor}", sensor, "2025-01-01T00:00:00Z", until, "0.02", f"fixture-only certificate for {sensor}", recorded_at="2025-01-01T00:00:00Z")
        store.register("maintenance", {"maintenance_id": "MAINT-01", "asset_id": "HVAC-01", "performed_at": "2025-01-02T00:00:00Z", "activity": "Synthetic filter/balance inspection; no physical work performed."})
        store.register("sample", {"sample_id": "REFERENCE-ONLY", "asset_id": "BR-01", "accession": "UniProt:P69905", "coordinate_frame": "not spatially registered",
                                  "evidence_type": "Public reference for a separate model exercise; not measured in this reactor and not paired omics."})
        rng = np.random.default_rng(42)
        ph = 7.10 + rng.normal(0, .025, 120)
        ph[84:] += np.linspace(.03, .30, 36)
        signals = {"PH-01": ph, "DO-01": 65 + rng.normal(0, 1.2, 120), "RPM-01": 150 + rng.normal(0, 1, 120),
                   "VCD-01": np.linspace(1.2, 3.8, 120) + rng.normal(0, .02, 120)}
        start = datetime(2025, 1, 10, tzinfo=timezone.utc)
        source_rows = []
        for index in range(120):
            observed = start + timedelta(minutes=index)
            for sensor, values in signals.items():
                message = {"event_id": f"{sensor}:{index:03d}", "sensor_id": sensor, "observed_at": observed.isoformat(),
                           "value": f"{values[index]:.6f}", "unit": METRICS[sensors[sensor]]["input_unit"]}
                store.ingest(message, (observed + timedelta(seconds=2)).isoformat())
                source_rows.append(message)
        replay = store.ingest(source_rows[0], "2025-01-10T03:00:00Z")
        invalid = [
            ("bad-unit", "PH-01", "2025-01-10T02:00:00Z", "7.1", "volts"),
            ("unknown-sensor", "MISSING", "2025-01-10T02:00:00Z", "7.1", "pH"),
            ("expired-calibration", "TEMP-EXPIRED", "2025-01-10T02:00:00Z", "37", "degC"),
            ("clock-ahead", "PH-01", "2025-01-10T04:00:00Z", "7.1", "pH"),
        ]
        for values in invalid:
            store.ingest(dict(zip(("event_id", "sensor_id", "observed_at", "value", "unit"), values)), "2025-01-10T03:00:00Z")
        accepted_ph = [row[0] / row[1] for row in store.connection.execute("SELECT scaled_value, value_scale FROM observations WHERE sensor_id='PH-01' AND status='accepted' ORDER BY observed_at")]
        drift = rolling_diagnostics(accepted_ph)
        rooms = [Room("suite-a", 30, 30, 900, 840, (10000, 200)), Room("suite-b", 40, 15, 800, 800, (20000, 400)), Room("suite-c", 60, 0, 600, 660, (30000, 600))]
        airflow = simulate(rooms, [("suite-a", "suite-b", 4), ("suite-b", "suite-c", 4)], [[300000, 3000], [200000, 2000], [100000, 1000]], duration_h=.5)
        analysis_bytes = json.dumps({"drift": drift, "airflow": airflow}, sort_keys=True, allow_nan=False).encode()
        with store.transaction():
            store.ledger.append("analysis:fixture-01", "fixture-analyst", "analysis_completed", "2025-01-10T03:00:01.000000+00:00",
                                {"analysis_sha256": hashlib.sha256(analysis_bytes).hexdigest(), "input_head": store.events()[-1]["event_hash"],
                                 "baseline_observations": 48, "window_observations": 16, "projection_sha256": projection_digest(store.connection), "mode": "shadow-only; no equipment command"})
        attestation = store.attestation()
        verification = store.verify(attestation)
        counts = {row[0]: row[1] for row in store.connection.execute("SELECT status, COUNT(*) FROM observations GROUP BY status")}
        report = {"evidence_type": "synthetic seeded fixture; no real facility, validation, or equipment control", "seed": 42, "schema_version": 1,
                  "counts": counts, "duplicate_replay": replay, "audit_verification": verification, "attestation": attestation,
                  "assets": [dict(row) for row in store.connection.execute("SELECT * FROM assets ORDER BY asset_id")],
                  "observations": [dict(row) for row in store.connection.execute("SELECT * FROM observations ORDER BY recorded_event")],
                  "drift": drift, "airflow": airflow, "events": store.events(),
                  "projection_sha256": projection_digest(store.connection),
                  "asset_details": {row[0]: inspect_asset(store.connection, row[0]) for row in store.connection.execute("SELECT asset_id FROM assets ORDER BY asset_id")},
                  "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((ROOT / "biotech").glob("*")) if path.suffix in {".py", ".sql"}},
                  "boundaries": ["Signatures do not establish Part 11 or Annex 11 compliance.", "Drift review is not contamination detection or yield prediction.",
                                 "Well-mixed transport is not laminar CFD, sterility assurance, or ISO qualification.", "H3, room geometry, and molecular identifiers use distinct frames."]}
        (directory / "attestation.json").write_text(json.dumps(attestation, indent=2) + "\n")
        (directory / "source_messages.json").write_text(json.dumps(source_rows, indent=2) + "\n")
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"database": str(directory / "facility.sqlite"), "counts": counts, "audit_valid": verification["valid"], "events": verification["events"],
                          "particle_conservation_residual": airflow["conservation_residual_by_bin"]}, indent=2))
        return report
    finally:
        store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "artifacts/facility-demo")
    parser.add_argument("--report", type=Path, default=ROOT / "studies/results/facility.json")
    args = parser.parse_args()
    run(args.directory, args.report)
