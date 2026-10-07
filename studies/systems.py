"""Local system contracts: spatial indexing, geometry, and retry-safe state changes."""

from dataclasses import dataclass
import hashlib
import json
import math
import sqlite3

import h3
import numpy as np


@dataclass(frozen=True)
class Box:
    lower: tuple
    upper: tuple
    uncertainty_m: float = 0.0

    def __post_init__(self):
        if len(self.lower) != 3 or len(self.upper) != 3:
            raise ValueError("Boxes require three-dimensional coordinates in local metres.")
        if not all(math.isfinite(v) for v in (*self.lower, *self.upper, self.uncertainty_m)):
            raise ValueError("Geometry and uncertainty must be finite.")
        if self.uncertainty_m < 0 or any(lo >= hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("Require nonnegative uncertainty and strictly ordered box bounds.")


def possible_clash(left, right, clearance_m=0):
    if not math.isfinite(clearance_m) or clearance_m < 0:
        raise ValueError("Clearance must be finite and nonnegative.")
    margin = left.uncertainty_m + right.uncertainty_m + clearance_m
    return all(a - margin <= d and c - margin <= b
               for a, b, c, d in zip(left.lower, left.upper, right.lower, right.upper))


def heat_scenario(air_c, canopy, albedo, solar_w_m2=700, heat_transfer_w_m2_k=25):
    values = (air_c, canopy, albedo, solar_w_m2, heat_transfer_w_m2_k)
    if not all(math.isfinite(v) for v in values) or not 0 <= canopy <= 1 or not 0 <= albedo <= 1:
        raise ValueError("Require finite inputs and canopy/albedo fractions in [0, 1].")
    if solar_w_m2 < 0 or heat_transfer_w_m2_k <= 0:
        raise ValueError("Solar forcing must be nonnegative and heat transfer positive.")
    return air_c + (1 - albedo) * solar_w_m2 * (1 - canopy) / heat_transfer_w_m2_k


def spatial_cells(latitude, longitude, resolution=9, rings=2):
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180 or not 0 <= resolution <= 15 or not 0 <= rings <= 10:
        raise ValueError("Invalid geographic coordinate, H3 resolution, or bounded ring count.")
    center = h3.latlng_to_cell(latitude, longitude, resolution)
    return sorted(h3.grid_disk(center, rings))


def canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class MutationJournal:
    """Single-database transaction boundary, not a distributed compliance system."""

    def __init__(self, path):
        self.connection = sqlite3.connect(path, timeout=5, isolation_level=None)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS assets (tenant TEXT, asset TEXT, value TEXT NOT NULL, PRIMARY KEY(tenant, asset));
            CREATE TABLE IF NOT EXISTS events (sequence INTEGER PRIMARY KEY, tenant TEXT NOT NULL,
                request_key TEXT NOT NULL, payload TEXT NOT NULL, previous_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL, UNIQUE(tenant, request_key));
        """)

    def apply(self, tenant, request_key, asset, value, fail_before_commit=False):
        if not all(isinstance(v, str) and v.strip() for v in (tenant, request_key, asset)):
            raise ValueError("Tenant, request key, and asset must be nonempty strings.")
        payload = canonical({"asset": asset, "value": value})
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self.connection.execute("SELECT payload, event_hash FROM events WHERE tenant=? AND request_key=?",
                                               (tenant, request_key)).fetchone()
            if existing:
                if existing[0] != payload:
                    raise ValueError("An idempotency key cannot be reused for a different intent.")
                self.connection.commit()
                return {"event_hash": existing[1], "replayed": True}
            previous = self.connection.execute("SELECT sequence, event_hash FROM events ORDER BY sequence DESC LIMIT 1").fetchone()
            sequence, previous_hash = (previous[0] + 1, previous[1]) if previous else (1, "0" * 64)
            envelope = canonical({"sequence": sequence, "tenant": tenant, "request_key": request_key,
                                  "payload": payload, "previous_hash": previous_hash})
            event_hash = hashlib.sha256(envelope.encode()).hexdigest()
            self.connection.execute("INSERT INTO assets VALUES (?, ?, ?) ON CONFLICT(tenant, asset) DO UPDATE SET value=excluded.value",
                                    (tenant, asset, canonical(value)))
            self.connection.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                                    (sequence, tenant, request_key, payload, previous_hash, event_hash))
            if fail_before_commit:
                raise RuntimeError("Injected failure before commit")
            self.connection.commit()
            return {"event_hash": event_hash, "replayed": False}
        except BaseException:
            self.connection.rollback()
            raise

    def verify(self, expected_head=None):
        previous = "0" * 64
        rows = self.connection.execute("SELECT * FROM events ORDER BY sequence").fetchall()
        for expected_sequence, row in enumerate(rows, 1):
            sequence, tenant, request_key, payload, previous_hash, event_hash = row
            envelope = canonical({"sequence": sequence, "tenant": tenant, "request_key": request_key,
                                  "payload": payload, "previous_hash": previous_hash})
            if sequence != expected_sequence or previous_hash != previous or hashlib.sha256(envelope.encode()).hexdigest() != event_hash:
                return False
            previous = event_hash
        return expected_head is None or previous == expected_head

    def state(self):
        return {(tenant, asset): json.loads(value) for tenant, asset, value in self.connection.execute("SELECT * FROM assets")}

    def reconstructed_state(self):
        if not self.verify():
            raise ValueError("Cannot replay a journal with a broken hash chain.")
        result = {}
        for tenant, payload in self.connection.execute("SELECT tenant, payload FROM events ORDER BY sequence"):
            event = json.loads(payload)
            result[tenant, event["asset"]] = event["value"]
        return result

    def close(self):
        self.connection.close()


def synthetic_landscape(latitude, longitude):
    cells = spatial_cells(latitude, longitude)
    rng = np.random.default_rng(42)
    return [{"cell": cell, "canopy_fraction": float(rng.uniform(.05, .6)),
             "albedo": float(rng.uniform(.1, .5)), "evidence": "synthetic scenario, not surveyed"} for cell in cells]
