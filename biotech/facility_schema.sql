PRAGMA foreign_keys = ON;

CREATE TABLE facilities (
    facility_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    h3_cell TEXT,
    provenance TEXT NOT NULL
);
CREATE TABLE audit_events (
    sequence INTEGER PRIMARY KEY CHECK(sequence > 0),
    event_key TEXT NOT NULL UNIQUE,
    envelope TEXT NOT NULL,
    previous_hash TEXT NOT NULL CHECK(length(previous_hash) = 64),
    event_hash TEXT NOT NULL UNIQUE CHECK(length(event_hash) = 64),
    key_id TEXT NOT NULL CHECK(length(key_id) = 64),
    signature TEXT NOT NULL CHECK(length(signature) = 128)
);
CREATE TABLE assets (
    asset_id TEXT PRIMARY KEY,
    facility_id TEXT NOT NULL REFERENCES facilities(facility_id),
    kind TEXT NOT NULL CHECK(kind IN ('bioreactor', 'room', 'utility')),
    name TEXT NOT NULL,
    provenance TEXT NOT NULL,
    registered_event INTEGER NOT NULL REFERENCES audit_events(sequence)
);
CREATE TABLE sensors (
    sensor_id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL REFERENCES assets(asset_id),
    metric TEXT NOT NULL,
    input_unit TEXT NOT NULL,
    registered_event INTEGER NOT NULL REFERENCES audit_events(sequence)
);
CREATE TABLE calibrations (
    calibration_id TEXT PRIMARY KEY,
    sensor_id TEXT NOT NULL REFERENCES sensors(sensor_id),
    valid_from TEXT NOT NULL,
    valid_until TEXT NOT NULL CHECK(valid_until > valid_from),
    uncertainty_decimal TEXT NOT NULL,
    certificate_reference TEXT NOT NULL,
    recorded_event INTEGER NOT NULL REFERENCES audit_events(sequence)
);
CREATE INDEX calibration_lookup ON calibrations(sensor_id, valid_from, valid_until);
CREATE TABLE maintenance (
    maintenance_id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL REFERENCES assets(asset_id),
    performed_at TEXT NOT NULL,
    activity TEXT NOT NULL,
    recorded_event INTEGER NOT NULL REFERENCES audit_events(sequence)
);
CREATE TABLE observations (
    source_event_id TEXT PRIMARY KEY,
    declared_sensor TEXT NOT NULL,
    sensor_id TEXT REFERENCES sensors(sensor_id),
    observed_at TEXT NOT NULL,
    received_at TEXT NOT NULL,
    scaled_value INTEGER CHECK(scaled_value IS NULL OR typeof(scaled_value) = 'integer'),
    value_scale INTEGER CHECK(value_scale IS NULL OR (typeof(value_scale) = 'integer' AND value_scale > 0)),
    normalized_unit TEXT,
    calibration_id TEXT REFERENCES calibrations(calibration_id),
    status TEXT NOT NULL CHECK(status IN ('accepted', 'quarantined')),
    reasons TEXT NOT NULL,
    intent_hash TEXT NOT NULL CHECK(length(intent_hash) = 64),
    recorded_event INTEGER NOT NULL UNIQUE REFERENCES audit_events(sequence),
    CHECK((scaled_value IS NULL AND value_scale IS NULL AND normalized_unit IS NULL) OR
          (scaled_value IS NOT NULL AND value_scale IS NOT NULL AND normalized_unit IS NOT NULL)),
    CHECK(status = 'quarantined' OR (sensor_id IS NOT NULL AND scaled_value IS NOT NULL AND calibration_id IS NOT NULL))
);
CREATE INDEX observation_asset_time ON observations(sensor_id, observed_at, source_event_id);
CREATE INDEX observation_status ON observations(status, observed_at);
CREATE TABLE sample_links (
    sample_id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL REFERENCES assets(asset_id),
    accession TEXT NOT NULL,
    coordinate_frame TEXT NOT NULL,
    evidence_type TEXT NOT NULL,
    recorded_event INTEGER NOT NULL REFERENCES audit_events(sequence)
);
PRAGMA user_version = 1;
