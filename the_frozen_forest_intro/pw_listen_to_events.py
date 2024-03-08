from playwright.sync_api import sync_playwright, Page
# here comes the question: what is comparable to a browser for GUI than for a terminal

# now, we need to visualize the interface. i think website (jpeg based) is better than obs. for training, we have to develop a protocol based on websocket to send messages with time aligned events. also you can send commands with websocket

# but before that, we can just ditch the protocol and create demo.

# shell environment and repl
# bot says

# why not just run this from the web? not quick enough?

# https://hub.docker.com/r/replco/polygott
# https://github.com/replit/prybar
# https://github.com/replit/nixmodules

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

# here comes the question: how to train this bot?
# there is only one thing that you can do: metalearning.

def print_request_sent(request):
    if "getIdentifier" in request.url:
        print("Request sent: " + request.url)

def handle_page_event(page: Page):
    createAndExposePageIdentifierAsFunctionName(page)
    print('new page at:', page.url)

pageIdentifierPrefix = "pageIdentifier_"

def createAndExposePageIdentifierAsFunctionName(page:Page):
    pageIdentifier = str(uuid.uuid4())
    page.expose_binding( # ugly but effective hack
        f"{pageIdentifierPrefix}{pageIdentifier.replace('-', '_')}", lambda: None
    )
    print("page identifier:", pageIdentifier)
    setattr(page, 'pageIdentifier', pageIdentifier)
    return pageIdentifier

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

extensionPaths = ",".join([pathToExtension,pathToCORSExtension,pathToDarkReaderExtension])

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
                # "--hide-scrollbars", # so it won't bother
                f"--disable-extensions-except={extensionPaths}",
                # f"--load-extension={pathToExtension}",
                f"--load-extension={extensionPaths}",
                # f"--load-extension={pathToCORSExtension}",
            ],
        )
        # browser.on('keydown', handle_keyboard_event)
        # playwright.on('keydown', handle_keyboard_event)
        browser.on('page', handle_page_event)

        page = browser.new_page() # this thing is not emitted in the event listener.
        # createAndExposePageIdentifierAsFunctionName(page)
        # pageIdentifier = createAndExposePageIdentifierAsFunctionName(page)

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
        # you can expect for popups.
        # will you lose focus?
        while True:
            page.wait_for_timeout(1000 * wait_sec)

        # Clean up
        page.close()
        browser.close()
