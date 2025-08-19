# reference:
# https://playwright.dev/docs/docker
# https://playwright.dev/python/docs/docker (with python code snippets)

REMOTE_SERVER_URL = 'ws://127.0.0.1:3000/'

from playwright.sync_api import sync_playwright

url = "https://xtermjs.org"
screenshot_savepath = "playwright_client_xterm_js_homepage_screenshot.png"

with sync_playwright() as p:
    print("Connecting to remote playwright server:", REMOTE_SERVER_URL)
    browser = p.chromium.connect(REMOTE_SERVER_URL)
    # a simple demo: open the xterm.js homepage, take screenshot and exit
    print("Creating new page")
    page = browser.new_page()
    print("Visit URL:", url)
    page.goto(url)
    print("Waiting for networkidle")
    page.wait_for_load_state('networkidle')
    print("Taking screenshot at:", screenshot_savepath)
    page.screenshot(path = screenshot_savepath)

print("Exiting")