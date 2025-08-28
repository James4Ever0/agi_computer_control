from playwright.sync_api import sync_playwright
import os

htmlfile = "./xtermjs-electron-xvfb/xtermjs_electron_headless.html"
assert os.path.isfile(htmlfile)

# question: why would we use playwright instead of agg-python-bindings?
# answer: we are just using it for familiarization, so we can integrate novnc. also we want to know the difference between different terminal emulators.

# we do not want the scrollbar to be visible

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    try:
        page = browser.new_page()
        url = "file://" + os.path.abspath(htmlfile)
        page.goto(url)
        page.wait_for_timeout(500)
        page.screenshot(path="screenshot.png")
        # use \r\n instead of \n
        page.evaluate("term.write('\\r\\nHello, world from Playwright!')")
        page.wait_for_timeout(500)
        page.screenshot(path="screenshot2.png")
        page.wait_for_timeout(500000)
    finally:
        browser.close()