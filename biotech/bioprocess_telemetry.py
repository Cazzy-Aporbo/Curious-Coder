"""Calibration-aware shadow-mode ingestion; never writes equipment setpoints."""

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import hashlib
import json
from pathlib import Path
import sqlite3

from biotech.audit_ledger import AuditLedger, EventSigner, canonical_bytes, verify_events


METRICS = {
    "ph": {"input_unit": "pH", "unit": "pH", "scale": 1000, "minimum": "0", "maximum": "14"},
    "dissolved_oxygen": {"input_unit": "% air saturation", "unit": "% air saturation", "scale": 1000, "minimum": "0", "maximum": "300"},
    "agitation": {"input_unit": "rpm", "unit": "rpm", "scale": 1, "minimum": "0", "maximum": "5000"},
    "viable_cell_density": {"input_unit": "10^6 cells/mL", "unit": "cells/mL", "scale": 1, "multiplier": 1000000, "minimum": "0", "maximum": "1000"},
    "temperature": {"input_unit": "degC", "unit": "degC", "scale": 1000, "minimum": "-100", "maximum": "150"},
}


def utc_timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("A timezone offset is required; local wall time is ambiguous.")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def fixed_measurement(metric, value, unit):
    if metric not in METRICS or not isinstance(value, str) or len(value) > 128:
        raise ValueError("Use a registered metric and bounded decimal-string value.")
    specification = METRICS[metric]
    if unit != specification["input_unit"]:
        raise ValueError("Unit does not match the sensor contract.")
    try:
        number = Decimal(value)
        if not number.is_finite() or not Decimal(specification["minimum"]) <= number <= Decimal(specification["maximum"]):
            raise ValueError("Value is outside the declared sensor-domain range.")
        scaled = int((number * specification.get("multiplier", 1) * specification["scale"]).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))
    except InvalidOperation as error:
        raise ValueError("Invalid decimal value.") from error
    return scaled, specification["scale"], specification["unit"]


def projection_digest(connection):
    tables = ("facilities", "assets", "sensors", "calibrations", "maintenance", "observations", "sample_links")
    state = {table: [dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY 1")] for table in tables}
    return hashlib.sha256(canonical_bytes({"schema_version": 1, "tables": state})).hexdigest()


class FacilityStore:
    def __init__(self, path, signer=None):
        self.connection = sqlite3.connect(path, timeout=5, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            tables = self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if tables:
                self.connection.close()
                raise ValueError("Refusing to initialize an unrelated database.")
            self.connection.executescript(Path(__file__).with_name("facility_schema.sql").read_text())
        elif version != 1:
            self.connection.close()
            raise ValueError("Unsupported schema version; use an explicit migration.")
        if version == 1 and signer is None:
            self.connection.close()
            raise ValueError("Appending to an existing database requires an explicit signing identity; read-only inspection does not.")
        self.signer = signer or EventSigner()
        self.ledger = AuditLedger(self.connection, self.signer)

    @contextmanager
    def transaction(self):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def register(self, kind, values, actor="fixture-loader", at="2025-01-01T00:00:00Z"):
        definitions = {
            "facility": ("facilities", ("facility_id", "name", "h3_cell", "provenance")),
            "asset": ("assets", ("asset_id", "facility_id", "kind", "name", "provenance")),
            "sensor": ("sensors", ("sensor_id", "asset_id", "metric", "input_unit")),
            "maintenance": ("maintenance", ("maintenance_id", "asset_id", "performed_at", "activity")),
            "sample": ("sample_links", ("sample_id", "asset_id", "accession", "coordinate_frame", "evidence_type")),
        }
        if kind not in definitions:
            raise ValueError("Unsupported registration kind.")
        table, columns = definitions[kind]
        if set(values) != set(columns):
            raise ValueError("Registration fields must match the schema exactly.")
        if kind == "sensor" and (values["metric"] not in METRICS or values["input_unit"] != METRICS[values["metric"]]["input_unit"]):
            raise ValueError("Sensor metric and unit must match a registered contract.")
        if kind == "maintenance":
            values = {**values, "performed_at": utc_timestamp(values["performed_at"])}
        with self.transaction():
            receipt = self.ledger.append(f"register:{kind}:{values[columns[0]]}", actor, f"register_{kind}", utc_timestamp(at), values)
            payload = [values[column] for column in columns]
            if kind != "facility":
                event_column = "registered_event" if kind in {"asset", "sensor"} else "recorded_event"
                columns = (*columns, event_column)
                payload.append(receipt["sequence"])
            self.connection.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", payload)
        return receipt

    def calibrate(self, calibration_id, sensor_id, valid_from, valid_until, uncertainty, certificate_reference, actor="fixture-calibrator", *, recorded_at):
        start, end = utc_timestamp(valid_from), utc_timestamp(valid_until)
        recorded = utc_timestamp(recorded_at)
        uncertainty_value = Decimal(uncertainty)
        if end <= start or not uncertainty_value.is_finite() or uncertainty_value < 0 or not certificate_reference.strip():
            raise ValueError("Calibration requires ordered validity, nonnegative uncertainty, and a source reference.")
        values = {"calibration_id": calibration_id, "sensor_id": sensor_id, "valid_from": start, "valid_until": end,
                  "uncertainty_decimal": str(uncertainty_value), "certificate_reference": certificate_reference}
        with self.transaction():
            overlap = self.connection.execute("SELECT 1 FROM calibrations WHERE sensor_id=? AND valid_from < ? AND valid_until > ?", (sensor_id, end, start)).fetchone()
            if overlap:
                raise ValueError("Overlapping calibration validity requires explicit reconciliation.")
            receipt = self.ledger.append(f"calibration:{calibration_id}", actor, "calibration_recorded", recorded, values)
            self.connection.execute("INSERT INTO calibrations VALUES (?, ?, ?, ?, ?, ?, ?)", (*values.values(), receipt["sequence"]))
        return receipt

    def ingest(self, message, received_at, actor="fixture-ingest", fail_before_commit=False):
        required = {"event_id", "sensor_id", "observed_at", "value", "unit"}
        if set(message) != required or any(not isinstance(message[key], str) or not message[key].strip() for key in required):
            raise ValueError("Telemetry requires exactly the five nonempty string fields in its contract.")
        observed, received = utc_timestamp(message["observed_at"]), utc_timestamp(received_at)
        intent_hash = hashlib.sha256(canonical_bytes(message)).hexdigest()
        with self.transaction():
            existing = self.connection.execute("SELECT * FROM observations WHERE source_event_id=?", (message["event_id"],)).fetchone()
            if existing:
                if existing["intent_hash"] != intent_hash:
                    raise ValueError("Source event identity was reused with different content.")
                return {"event_id": message["event_id"], "status": existing["status"], "replayed": True, "audit_sequence": existing["recorded_event"]}
            sensor = self.connection.execute("SELECT * FROM sensors WHERE sensor_id=?", (message["sensor_id"],)).fetchone()
            reasons, scaled, scale, normalized_unit, calibration_id = [], None, None, None, None
            if observed > received:
                reasons.append("OBSERVATION_AFTER_RECEIPT")
            if sensor is None:
                reasons.append("UNKNOWN_SENSOR")
            else:
                try:
                    scaled, scale, normalized_unit = fixed_measurement(sensor["metric"], message["value"], message["unit"])
                except ValueError:
                    reasons.append("VALUE_OR_UNIT_OUTSIDE_CONTRACT")
                calibration = self.connection.execute("SELECT calibration_id FROM calibrations WHERE sensor_id=? AND valid_from <= ? AND valid_until > ?", (message["sensor_id"], observed, observed)).fetchall()
                if len(calibration) != 1:
                    reasons.append("NO_UNAMBIGUOUS_VALID_CALIBRATION")
                else:
                    calibration_id = calibration[0][0]
            status = "quarantined" if reasons else "accepted"
            decision = {"message": message, "observed_utc": observed, "received_utc": received, "scaled_value": scaled,
                        "value_scale": scale, "normalized_unit": normalized_unit, "calibration_id": calibration_id,
                        "status": status, "reasons": reasons, "policy": "teaching-contract-v1", "intent_sha256": intent_hash}
            receipt = self.ledger.append(f"telemetry:{message['event_id']}", actor, "telemetry_assessed", received, decision)
            self.connection.execute("INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (message["event_id"], message["sensor_id"], message["sensor_id"] if sensor else None, observed, received,
                 scaled, scale, normalized_unit, calibration_id, status, json.dumps(reasons), intent_hash, receipt["sequence"]))
            if fail_before_commit:
                raise RuntimeError("Injected failure before commit")
        return {"event_id": message["event_id"], "status": status, "replayed": False, "audit_sequence": receipt["sequence"]}

    def events(self):
        return [dict(row) for row in self.connection.execute("SELECT * FROM audit_events ORDER BY sequence")]

    def attestation(self):
        events = self.events()
        return {"trusted_public_keys": {self.signer.key_id: self.signer.public_key.hex()},
                "expected_head": events[-1]["event_hash"] if events else "0" * 64,
                "scope": "Local demonstration anchor; independently protect keys and heads for a real trust boundary."}

    def verify(self, attestation):
        return verify_events(self.events(), attestation["trusted_public_keys"], attestation["expected_head"])

    def close(self):
        self.connection.close()
