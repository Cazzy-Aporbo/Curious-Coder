"""Verify both static-snapshot and read-only API facility-console interactions."""

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading

from playwright.sync_api import expect, sync_playwright

from biotech.facility_api import make_handler


ROOT = Path(__file__).resolve().parents[1]


def main(directory, report):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(directory / "facility.sqlite", report, directory / "attestation.json"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 1000}, reduced_motion="reduce")
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}", wait_until="domcontentloaded")
            expect(page.locator("#mode")).to_have_text("Local database inspection API")
            expect(page.locator("#asset-title")).to_contain_text("BR-01")
            expect(page.locator("#play")).to_be_disabled()
            page.locator("#asset-detail .table-wrap").nth(1).locator("button").first.click()
            expect(page.locator("#event-detail")).to_contain_text("calibration_recorded")
            page.locator("#verify").click()
            expect(page.locator("#status")).to_contain_text("projection match: true")
            page.select_option("#filter", "quarantined")
            expect(page.locator("#observations tr")).to_have_count(4)
            page.locator("#observations button").first.click()
            expect(page.locator("#event-detail")).to_contain_text("telemetry_assessed")
            page.select_option("#filter", "")
            expect(page.locator("#observations tr")).to_have_count(12)
            page.locator("#next").click()
            expect(page.locator("#page-label")).to_contain_text("13–24")
            page.get_by_role("button", name="HVAC-01", exact=True).click()
            expect(page.locator("#asset-detail")).to_contain_text("Synthetic filter/balance inspection")
            page.get_by_role("button", name="Inspect suite-a", exact=True).click()
            expect(page.locator("#asset-title")).to_contain_text("suite-a")
            page.emulate_media(reduced_motion="no-preference")
            expect(page.locator("#play")).to_be_enabled()
            page.locator("#play").click()
            expect(page.locator("#play")).to_have_text("Pause simulation")
            page.locator("#play").click()
            page.locator("#reset").click()
            expect(page.locator("#time-label")).to_have_text("0.0 min")
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
            assert not errors, errors
            browser.close()
        print("Facility browser checks passed: audit verification, calibration/maintenance inspection, quarantine filter, pagination, signed-event detail, room interaction, playback, reduced motion, and mobile width.")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=ROOT / "studies/results/facility.json")
    args = parser.parse_args()
    main(args.directory, args.report)
