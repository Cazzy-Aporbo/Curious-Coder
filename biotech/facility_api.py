"""Read-only loopback inspection API for the synthetic facility database."""

import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
from pathlib import Path
import re
import sqlite3
from urllib.parse import parse_qs, urlsplit

from biotech.audit_ledger import verify_events
from biotech.bioprocess_telemetry import projection_digest


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def reader(database):
    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    try:
        if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise ValueError("Unsupported facility schema.")
        yield connection
    finally:
        connection.close()


def inspect_asset(connection, asset_id):
    asset = connection.execute("SELECT * FROM assets WHERE asset_id=?", (asset_id,)).fetchone()
    if asset is None:
        raise KeyError(asset_id)
    return {"asset": dict(asset),
            "sensors": [dict(row) for row in connection.execute("SELECT * FROM sensors WHERE asset_id=? ORDER BY sensor_id", (asset_id,))],
            "calibrations": [dict(row) for row in connection.execute("SELECT c.*, s.input_unit AS uncertainty_unit FROM calibrations c JOIN sensors s USING(sensor_id) WHERE s.asset_id=? ORDER BY c.valid_from, c.calibration_id", (asset_id,))],
            "maintenance": [dict(row) for row in connection.execute("SELECT * FROM maintenance WHERE asset_id=? ORDER BY performed_at", (asset_id,))],
            "sample_links": [dict(row) for row in connection.execute("SELECT * FROM sample_links WHERE asset_id=? ORDER BY sample_id", (asset_id,))]}


def make_handler(database, report_path, attestation_path):
    database, report_path, attestation_path = map(Path, (database, report_path, attestation_path))
    with reader(database):
        pass
    attestation = json.loads(attestation_path.read_text())
    report = json.loads(report_path.read_text())
    with reader(database) as connection:
        events = [dict(row) for row in connection.execute("SELECT * FROM audit_events ORDER BY sequence")]
        verification = verify_events(events, attestation["trusted_public_keys"], attestation["expected_head"])
        if not verification["valid"] or not events:
            raise ValueError("The database does not match its supplied audit anchor.")
        signed_analysis = json.loads(events[-1]["envelope"])["payload"].get("analysis_sha256")
        actual_analysis = hashlib.sha256(json.dumps({"drift": report["drift"], "airflow": report["airflow"]}, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if signed_analysis != actual_analysis:
            raise ValueError("Simulation/report content does not match the signed analysis digest.")

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value, status=200):
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            parsed = urlsplit(self.path)
            static = {"/": (ROOT / "biotech/facility_console.html", "text/html; charset=utf-8"),
                      "/assets/facility_console.mjs": (ROOT / "assets/facility_console.mjs", "text/javascript; charset=utf-8")}
            if parsed.path in static:
                path, content_type = static[parsed.path]
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(path.read_bytes())
                return
            try:
                with reader(database) as connection:
                    if parsed.path == "/api/summary":
                        self.send_json({"mode": "synthetic fixture; read-only inspection; no equipment commands",
                            "assets": [dict(row) for row in connection.execute("SELECT * FROM assets ORDER BY asset_id")],
                            "observation_counts": {row[0]: row[1] for row in connection.execute("SELECT status, COUNT(*) FROM observations GROUP BY status")},
                            "audit_events": connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]})
                    elif parsed.path == "/api/simulation":
                        self.send_json({"airflow": report["airflow"], "drift": report["drift"], "evidence_type": report["evidence_type"]})
                    elif parsed.path == "/api/audit/verify":
                        events = [dict(row) for row in connection.execute("SELECT * FROM audit_events ORDER BY sequence")]
                        verification = verify_events(events, attestation["trusted_public_keys"], attestation["expected_head"])
                        if verification["valid"]:
                            recorded = json.loads(events[-1]["envelope"])["payload"].get("projection_sha256") if events else None
                            verification["projection_matches_signed_digest"] = projection_digest(connection) == recorded if recorded else None
                            if recorded and not verification["projection_matches_signed_digest"]:
                                verification["valid"] = False
                                verification["reason"] = "Materialized records differ from the signed projection digest"
                        self.send_json(verification)
                    elif parsed.path.startswith("/api/assets/"):
                        asset_id = parsed.path.removeprefix("/api/assets/")
                        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", asset_id):
                            raise ValueError("Invalid asset identifier.")
                        self.send_json(inspect_asset(connection, asset_id))
                    elif parsed.path == "/api/observations":
                        query = parse_qs(parsed.query)
                        limit, offset = int(query.get("limit", ["20"])[0]), int(query.get("offset", ["0"])[0])
                        status = query.get("status", [""])[0]
                        if not 1 <= limit <= 100 or not 0 <= offset <= 100000 or status not in {"", "accepted", "quarantined"}:
                            raise ValueError("Invalid pagination or status filter.")
                        where, parameters = ("WHERE status=?", [status]) if status else ("", [])
                        total = connection.execute(f"SELECT COUNT(*) FROM observations {where}", parameters).fetchone()[0]
                        rows = [dict(row) for row in connection.execute(f"SELECT * FROM observations {where} ORDER BY recorded_event LIMIT ? OFFSET ?", [*parameters, limit, offset])]
                        self.send_json({"total": total, "limit": limit, "offset": offset, "rows": rows})
                    elif re.fullmatch(r"/api/events/[1-9][0-9]*", parsed.path):
                        row = connection.execute("SELECT * FROM audit_events WHERE sequence=?", (int(parsed.path.rsplit("/", 1)[-1]),)).fetchone()
                        if row is None:
                            raise KeyError("event")
                        self.send_json(dict(row))
                    else:
                        self.send_json({"error": "Not found"}, 404)
            except (ValueError, TypeError):
                self.send_json({"error": "Invalid request"}, 400)
            except KeyError:
                self.send_json({"error": "Not found"}, 404)

        def do_POST(self):
            self.send_json({"error": "Read-only API; no mutation endpoint"}, 405)

        do_PUT = do_POST
        do_PATCH = do_POST
        do_DELETE = do_POST

    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=ROOT / "artifacts/facility-demo/facility.sqlite")
    parser.add_argument("--report", type=Path, default=ROOT / "studies/results/facility.json")
    parser.add_argument("--attestation", type=Path, default=ROOT / "artifacts/facility-demo/attestation.json")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.database, args.report, args.attestation))
    print(f"Read-only synthetic facility console: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
