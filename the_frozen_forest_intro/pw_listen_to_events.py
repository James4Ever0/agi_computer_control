from playwright.sync_api import sync_playwright

# bad news: you cannot record/listen mouse events using playwright. very bad.
# good news: you can record these events on your own device using os specific keylogger, if you know these events are fired to the browser, around the effective area.
# bonus: use keylogger browser extensions
import uuid
# import json

page_url = "https://www.baidu.com"
serverPort = 4471

# def handle_keyboard_event(event):
#     print("Keyboard event:", event)

# def handle_mouse_event(event):
#     print("Mouse event:", event)


def print_request_sent(request):
    if "getIdentifier" in request.url:
        print("Request sent: " + request.url)


# def print_request_finished(request):
#   print("Request finished: " + request.url)

wait_sec = 10
import os

ext_path = "keylogger_extension/virtual-keylogger"
pathToExtension = os.path.abspath(ext_path)
pathToCORSExtension = os.path.abspath("ForceCORS")
pathToDarkReaderExtension = os.path.abspath(
    "darkreader-chrome"
)  # this will pop up window. make sure that you have persisted context
print("loading extension path:", pathToExtension)
import tempfile

pageIdentifierPrefix = "pageIdentifier_"
with tempfile.TemporaryDirectory() as tmpdir:
    with sync_playwright() as playwright:  # this is incognito. not so good.
        browser = playwright.chromium.launch_persistent_context(
            user_data_dir=tmpdir,
            headless=False,
            # https://www.chromium.org/developers/how-tos/run-chromium-with-flags/
            # https://peter.sh/experiments/chromium-command-line-switches/
            args=[
                # browser = playwright.chromium.launch(headless=False,  args= [
                # f"--disable-extensions-except={pathToExtension}",
                # "--force-dark-mode",
                f"--disable-extensions-except={pathToExtension},{pathToCORSExtension},{pathToDarkReaderExtension}",
                # f"--load-extension={pathToExtension}",
                f"--load-extension={pathToCORSExtension},{pathToExtension},{pathToDarkReaderExtension}",
                # f"--load-extension={pathToCORSExtension}",
            ],
        )
        # browser.on('keydown', handle_keyboard_event)
        # playwright.on('keydown', handle_keyboard_event)
        pageIdentifier = str(uuid.uuid4())
        page = browser.new_page()
        page.expose_binding( # ugly but effective hack
            f"{pageIdentifierPrefix}{pageIdentifier.replace('-', '_')}", lambda: None
        )
        # page.on('request', print_request_sent)
        # def route_page_identifier(route):
        #    print('routing') # routing, but not working.
        #    return route.fulfill(status = 200, json = {"client_id": pageIdentifier})
        # page.route(
        #     f"http://localhost:{serverPort}/getIdentifier",
        #     # route_page_identifier,
        #     lambda route:
        #     # route.abort('connectionfailed')
        #     route.continue_(url = route.request.url+"?client_id="+pageIdentifier)
        #     # route.fulfill(status = 200, json = {"client_id": pageIdentifier})
        # )
        # page.evaluate(f'window.generateUUID = () => {repr(pageIdentifier)}')
        # pageIdentifier = page.evaluate('pageIdentifier')

        # def generateUUID():
        #     return pageIdentifier
        # page.expose_function('generateUUID', generateUUID)
        # BUG: having trouble running exposed functions in browser extensions
        # we can simply expose callback and pass it to event listeners, without browser extension, but that cannot survive navigation.
        # ref: https://github.com/microsoft/vscode-test-web/issues/69
        # ref: https://github.com/microsoft/playwright/issues/12017
        print("page identifier:", pageIdentifier)

        # Enable input events on the page
        # no such thing.
        # page.set_input_interception(True)

        # Listen to keyboard events

        # page.on("request", print_request_sent)
        # page.on("requestfinished", print_request_finished)

        # we do not have these events.
        # page.on('keydown', handle_keyboard_event)
        # page.on('keyup', handle_keyboard_event)

        # # Listen to mouse events
        # page.on('mousedown', handle_mouse_event)
        # page.on('mouseup', handle_mouse_event)
        # page.on('mousemove', handle_mouse_event)

        # nothing showed up.

        # Navigate to a webpage
        page.goto(page_url)

        # Wait for events
        # print("exit in %d seconds" % wait_sec)
        while True:
            page.wait_for_timeout(1000 * wait_sec)

        # Clean up
        page.close()
        browser.close()
