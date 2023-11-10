import os

# os.environ["DEBUG"] = "pw:api"  # to check what the heck is going on with the screenshot
# also, just maybe the content script problem
import time
import deprecated

from playwright.sync_api import sync_playwright, Page, BrowserContext, CDPSession
import random
import string
import base64
import requests

request_session = requests.Session()
# need help?
# ref: https://www.tampermonkey.net/scripts.php
# ref: https://greasyfork.org/zh-CN/scripts/388540-bing-batch-open-tabs

# url = "https://www.bilibili.com"
# url = "https://www.baidu.com"
url = "https://www.bilibili.com/video/BV1J24y1c7kE"

# should you use javascript to capture the webpage, not here
# cause it is error prone here

max_page_limit = 5

# hint: you can run the browser in container, and connect to it using websocket
# chromium.launchServer()
# browserServer.wsEndpoint()
from typing import cast

# if we cannot take screenshots indefinitely, then let's just have the topmost browser session

# ref: https://github.com/microsoft/playwright/issues/15773
# this is called the chrome devtools protocol (cdp).

from contextlib import contextmanager


@contextmanager
def cdp_context(page: Page):
    cdp = page.context.new_cdp_session(page)
    try:
        yield cdp
    finally:
        cdp.detach()


@deprecated.deprecated
def quickScreenshotPNG(
    page: Page,
):  # sometimes, protocol error and unable to capture screenshot (still stuck)
    # protocol error.
    with cdp_context(page) as cdp:
        screenshot_data = cast(
            str,
            cdp.send(
                "Page.captureScreenshot",
                {
                    # ref: https://chromedevtools.github.io/devtools-protocol/tot/Page/#method-captureScreenshot
                    "optimizeForSpeed": True
                },
            )["data"],
        )  # type: str, base64 encoded
        png_bytes = base64.b64decode(screenshot_data)
        return png_bytes


def checkIfPageHidden(page: Page):
    with cdp_context(page) as cdp:
        return cdp.send("Page.isHidden")[""]


def random_character_generator():
    return random.choice(string.ascii_letters + string.digits)


# ref: https://github.com/microsoft/playwright/issues/16307
@deprecated.deprecated
def performCDPScreenshotFix(page):  # not working. let's start tracing?
    cdp = context.new_cdp_session(page)
    register_page_exit_event_handler(page, cdp)
    cdp.on(
        "Page.screencastFrame",
        lambda params: cdp.send(
            "Page.screencastFrameAck", {"sessionId": params["sessionId"]}
        ),
    )
    cdp.send("Page.startScreencast", dict())


def handle_page_event(page: Page):
    # performCDPScreenshotFix(page)
    createAndExposePageIdentifierAsFunctionName(page)
    register_io_event_handler(page)
    print("new page at:", page.url)


def register_io_event_handler(page: Page):
    page.on(
        "filechooser", lambda file_chooser: file_chooser.set_files([])
    )  # not to select a thing
    # page.on('filechooser', lambda file_chooser: file_chooser.set_files(os.path.abspath('pointer_tracing.html')))
    page.on("download", lambda download: download.cancel())  # not download anything.
    # page.on('dialog', lambda e: None)
    # let's not try to understand what is going on here.


def register_page_exit_event_handler(page: Page, cdp: CDPSession):
    page.on("close", lambda e: cdp.detach())
    page.on("crash", lambda e: cdp.detach())
    page.on("pageerror", lambda e: cdp.detach())


import uuid

pageIdentifierPrefix = "pageIdentifier_"


def createAndExposePageIdentifierAsFunctionName(page: Page):
    pageIdentifier = str(uuid.uuid4())
    page.expose_binding(  # ugly but effective hack
        f"{pageIdentifierPrefix}{pageIdentifier.replace('-', '_')}", lambda: None
    )
    print("page identifier:", pageIdentifier)
    setattr(page, "pageIdentifier", pageIdentifier)
    return pageIdentifier


def random_actor(page: Page, viewport_width: int, viewport_height: int):
    action_choices = [
        # lambda: page.keyboard.up(random_character_generator()),
        # lambda: page.keyboard.down(random_character_generator()),
        lambda: page.keyboard.press(random_character_generator()),
        # lambda: page.mouse.move(
        #     random.randint(0, viewport_width), random.randint(0, viewport_height)
        # ),
        lambda: page.mouse.click(
            random.randint(0, viewport_width), random.randint(0, viewport_height)
        )
        # lambda: page.mouse.down(),
        # lambda: page.mouse.up(),
    ]
    action = random.choice(action_choices)
    action()


SCREENSHOT_TIMEOUT = 3 * 1000
# SCREENSHOT_TIMEOUT = 2 * 1000
# SCREENSHOT_TIMEOUT = 1 * 1000

# timeout this overall
# import func_timeout
# ACTION_LOOP_TIMEOUT = 10


# you can separate client from server, so you can restart server or client separately
# @func_timeout.func_set_timeout(ACTION_LOOP_TIMEOUT)
def execute_action_loop(
    context: BrowserContext,
    counter: int,
    counter_threshold: int,
    viewport_width: int,
    viewport_height: int,
    screenshot_timeout,
):
    # so somehow pages in the 'background' is still being clicked
    kill_page_count = max(len(context.pages) - max_page_limit, 0)
    if kill_page_count > 0:
        for kp in random.sample(context.pages, kill_page_count):
            print("killing page:", kp.url)
            kp.close()
    print("active page count:", len(context.pages))  # sometimes, still more than 5
    for index, page in enumerate(context.pages):  # visible
        if page.is_closed():
            continue

        with cdp_context(page) as cdp:
            # you would do both
            cdp.send("Page.setWebLifecycleState", dict(state="active"))
            # bringing to front significantly improves performance
            page.bring_to_front()  # so this might (not) save your day
            if counter % counter_threshold == 0:
                # pass  # let's not take screenshot here.
                # page.bring_to_front()
                # but will get stuck easily
                # screenshot_data = quickScreenshotPNG(page)  # it is working. png data.
                # import base64
                # with open('screenshot.png', 'wb') as f:
                #     content = base64.b64decode(screenshot_data)
                #     f.write(content)
                # breakpoint()
                screenshot_data_bytes = page.screenshot(
                    type="png", timeout=screenshot_timeout
                )  # sometimes this will timeout. too slow. shall be fixed.

                screenshot_data = base64.b64encode(screenshot_data_bytes).decode()
                request_session.post(
                    "http://localhost:4471/submitScreenshot",
                    json=dict(
                        client_id=getattr(page, "pageIdentifier", "unknown"),
                        screenshot_data=screenshot_data,
                        timestamp = time.time(),
                    ),
                )
                # breakpoint()
                # page.screenshot(
                #     path=f"page_{index}.png", timeout=screenshot_timeout
                # )  # taking screenshot time exceeded?
            random_actor(page, viewport_width, viewport_height)
        # print('active page:', page.url)
        # print('state:',  page.evaluate('document.visibilityState'), 'url:', page.url)
    counter += 1
    if counter >= counter_threshold:
        counter = 0
    return counter


import os

COUNTER_THRESHOLD = 100
# COUNTER_THRESHOLD = 1000
# import time
BREAKTIME_LENGTH = 0.3
extensionPaths = ",".join(
    [
        os.path.abspath("keylogger_extension/virtual-keylogger"),
        os.path.abspath("ForceCORS"),
        os.path.abspath("darkreader-chrome"),
    ]
)
# it is only getting slower. so why not just use docker
google_chrome = r"C:\Users\z98hu\AppData\Local\Google\Chrome\Application\chrome.exe"  # let's play video.
with sync_playwright() as playwright:
    # use persistent context to load extensions.
    context = playwright.chromium.launch_persistent_context(
        "",  # what does this mean?
        # browser = playwright.chromium.launch(
        executable_path=google_chrome,
        headless=False,
        args=[
            "--headless=new",
            f"--disable-extensions-except={extensionPaths}",
            f"--load-extension={extensionPaths}",
        ]  # working.
        # not working.
        #   ignore_default_args=["--mute-audio"]
    )
    # context.tracing.start()
    context.on("close", lambda e: print("context closed"))
    # context.browser.on("disconnected", lambda e: print("browser disconnected"))
    # context = browser.new_context()
    context.on("page", handle_page_event)
    # keep clicking buttons might initiate download or uploading events
    # you need to prevent that.

    init_page = context.new_page()
    init_page.goto(url)
    # createAndExposePageIdentifierAsFunctionName(init_page)
    # this thing is duplicated. cause this event will be handled by the event listener already. don't have to trigger twice.
    # handle_page_event(init_page)

    viewport_width, viewport_height = (
        init_page.viewport_size["width"],
        init_page.viewport_size["height"],
    )
    counter = 0
    # for quite some time, you cannot type a thing into the browser. that is bad.
    while True:  # this loop can be troublesome.
        # close any unfocused page
        # topMostPage = random.choice(context.pages)
        # topMostPage.bring_to_front()

        try:
            counter = execute_action_loop(
                context,
                counter,
                COUNTER_THRESHOLD,
                viewport_width,
                viewport_height,
                SCREENSHOT_TIMEOUT,
            )
            # time.sleep(BREAKTIME_LENGTH)
        except Exception as e:
            print("exception:", e)

        # print('state:',  page.evaluate('document.hidden'), 'url:', page.url)
        # print('state:',  page.evaluate('window.statusbar'), 'url:', page.url)

        # for page in context.pages: # closing background pages (losing focus)
        # if page.evaluate('document.visibilityState') is False:
        #     print('closing background page:', page.url)
        #     page.close()
        # else:
        #     topMostPage = page
        # try:
        # random_actor(init_page, viewport_width, viewport_height)
        # except KeyboardInterrupt:
        #     print('exiting due to keyboard interrupt')
        #     break
        # except Exception as e:
        #     print(e)
    context.close()
    # browser.close()
