from playwright.sync_api import sync_playwright

# bad news: you cannot record/listen mouse events using playwright. very bad.
# good news: you can record these events on your own device using os specific keylogger, if you know these events are fired to the browser, around the effective area.
# bonus: use keylogger browser extensions

page_url = "https://www.baidu.com"

# def handle_keyboard_event(event):
#     print("Keyboard event:", event)

# def handle_mouse_event(event):
#     print("Mouse event:", event)

# def print_request_sent(request):
#   print("Request sent: " + request.url)

# def print_request_finished(request):
#   print("Request finished: " + request.url)

wait_sec = 10
import os

ext_path = "keylogger_extension/virtual-keylogger"
pathToExtension = os.path.abspath(ext_path)
pathToCORSExtension = os.path.abspath("ForceCORS")
print("loading extension path:", pathToExtension)
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    with sync_playwright() as playwright:  # this is incognito. not so good.
        browser = playwright.chromium.launch_persistent_context(
            user_data_dir=tmpdir,
            headless=False,
            args=[
                # browser = playwright.chromium.launch(headless=False,  args= [
                # f"--disable-extensions-except={pathToExtension}",
                f"--disable-extensions-except={pathToExtension},{pathToCORSExtension}",
                f"--load-extension={pathToExtension}",
                f"--load-extension={pathToCORSExtension}",
            ],
        )
        # browser.on('keydown', handle_keyboard_event)
        # playwright.on('keydown', handle_keyboard_event)
        page = browser.new_page()

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
