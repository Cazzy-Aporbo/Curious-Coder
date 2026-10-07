from contextlib import contextmanager
from http.server import ThreadingHTTPServer
import json
import sqlite3
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from biotech.facility_api import make_handler
from biotech.facility_demo import run


@pytest.fixture
def fixture_run(tmp_path):
    directory = tmp_path / "run"
    report_path = tmp_path / "facility.json"
    report = run(directory, report_path)
    return directory, report_path, report


@contextmanager
def running_api(directory, report_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(directory / "facility.sqlite", report_path, directory / "attestation.json"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def get(url):
    with urlopen(url, timeout=10) as response:
        return json.load(response)


def test_api_inspection_pagination_and_read_only_boundary(fixture_run):
    directory, report_path, report = fixture_run
    assert report["counts"] == {"accepted": 480, "quarantined": 4}
    with running_api(directory, report_path) as base:
        assert get(base + "/api/summary")["audit_events"] == 503
        detail = get(base + "/api/assets/BR-01")
        assert len(detail["calibrations"]) == 5
        assert detail["sample_links"][0]["coordinate_frame"] == "not spatially registered"
        page = get(base + "/api/observations?status=quarantined&limit=2&offset=2")
        assert page["total"] == 4 and len(page["rows"]) == 2
        verification = get(base + "/api/audit/verify")
        assert verification["valid"] and verification["projection_matches_signed_digest"]
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/api/observations", data=b"{}", method="POST"), timeout=10)
        assert error.value.code == 405
        with pytest.raises(HTTPError) as error:
            get(base + "/api/observations?limit=100000")
        assert error.value.code == 400


def test_projection_tampering_is_distinct_from_signature_verification(fixture_run):
    directory, report_path, _ = fixture_run
    with running_api(directory, report_path) as base:
        connection = sqlite3.connect(directory / "facility.sqlite")
        try:
            connection.execute("UPDATE observations SET scaled_value=7000 WHERE source_event_id='PH-01:000'")
            connection.commit()
        finally:
            connection.close()
        verification = get(base + "/api/audit/verify")
        assert not verification["valid"]
        assert not verification["projection_matches_signed_digest"]


def test_unsigned_analysis_changes_are_rejected_at_api_startup(fixture_run):
    directory, report_path, report = fixture_run
    report["drift"]["window_size"] = 99
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="signed analysis digest"):
        make_handler(directory / "facility.sqlite", report_path, directory / "attestation.json")
