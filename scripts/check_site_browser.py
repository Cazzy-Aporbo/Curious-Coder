"""Exercise the built site in Chromium, including keyboard and mobile controls."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]


def main():
    handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT / "_site"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    address = f"http://127.0.0.1:{server.server_port}"
    errors = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
            context.grant_permissions(["clipboard-read", "clipboard-write"], origin=address)
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(address, wait_until="domcontentloaded")
            page.locator(".copy-button").first.wait_for()
            motion = page.locator(".protocol-play").first
            expect(motion).to_be_disabled()
            expect(page.locator("picture.protocol-motion").first).to_have_attribute("data-playing", "false")
            page.emulate_media(reduced_motion="no-preference")
            expect(motion).to_be_enabled()
            motion.click()
            expect(motion).to_have_text("Pause workflow")
            expect(page.locator("picture.protocol-motion").first).to_have_attribute("data-playing", "true")
            motion.click()
            expect(motion).to_have_text("Play workflow")
            page.emulate_media(reduced_motion="reduce")
            expect(motion).to_be_disabled()
            expect(page.locator("#count-tn")).to_have_text("71")
            expect(page.locator("#count-fn")).to_have_text("2")
            threshold = page.locator("#decision-threshold")
            threshold.focus()
            page.keyboard.press("ArrowRight")
            expect(page.locator("#threshold-value")).to_have_text("0.51")
            page.select_option("#model-choice", "Residual MLP")
            page.get_by_role("button", name="Reset comparison", exact=True).click()
            expect(page.locator("#model-choice")).to_have_value("Logistic regression")
            expect(page.locator("#threshold-value")).to_have_text("0.50")
            page.locator("#search-panel summary").click()
            page.get_by_label("Search studies", exact=True).fill("protein")
            expect(page.locator("#search-results a").first).to_be_visible()
            assert page.locator("#search-results a").count() <= 8
            page.get_by_role("button", name="Clear", exact=True).click()
            expect(page.locator("#search-results a")).to_have_count(0)
            page.locator(".copy-button").first.click()
            expect(page.locator(".copy-button").first).to_have_text("Copied")
            assert "python" in page.evaluate("navigator.clipboard.readText()")
            inspect = page.locator(".figure-controls button").first
            inspect.click()
            expect(page.locator("#figure-dialog")).to_be_visible()
            page.locator("#figure-zoom").focus()
            page.keyboard.press("ArrowRight")
            expect(page.locator("#zoom-value")).to_have_text("125%")
            page.keyboard.press("Escape")
            expect(page.locator("#figure-dialog")).not_to_be_visible()
            expect(inspect).to_be_focused()
            page.locator("#theme-toggle").click()
            page.reload(wait_until="domcontentloaded")
            expect(page.locator("html")).to_have_attribute("data-theme", "night")
            page.locator("#theme-toggle").click()
            page.evaluate("window.scrollTo(0, 0)")
            artifacts = ROOT / "artifacts"
            artifacts.mkdir(exist_ok=True)
            page.screenshot(path=str(artifacts / "browser-home.png"))
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
            assert page.evaluate("getComputedStyle(document.querySelector('button')).transitionDuration") == "0s"
            page.screenshot(path=str(artifacts / "browser-mobile.png"))
            page.goto(address + "/studies/clinical_benchmark.html", wait_until="domcontentloaded")
            page.locator(".copy-button").first.wait_for()
            expect(page.locator("#count-tp")).to_have_text("40")
            page.get_by_role("navigation", name="Learning route").get_by_role("link", name="Next · Examine uncertainty").click()
            expect(page.get_by_role("heading", name="What would make the comparison convincing?", exact=True)).to_be_visible()
            page.goto(address + "/biotech/facility_console.html", wait_until="domcontentloaded")
            expect(page.locator("#mode")).to_contain_text("Recorded synthetic snapshot")
            expect(page.locator("#asset-title")).to_contain_text("BR-01")
            page.locator("#verify").click()
            expect(page.locator("#status")).to_contain_text("Recorded verification")
            page.select_option("#filter", "quarantined")
            expect(page.locator("#observations tr")).to_have_count(4)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
            assert not errors, errors
            browser.close()
        print("Browser checks passed: search, copy, threshold controls, figure zoom, focus restoration, palette persistence, workflow animation controls, mobile width, and reduced motion.")
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


if __name__ == "__main__":
    main()
