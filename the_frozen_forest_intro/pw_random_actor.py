from playwright.sync_api import sync_playwright, Page, BrowserContext
import random
import string

# url = "https://www.bilibili.com"
# url = "https://www.baidu.com"
url = "https://www.bilibili.com/video/BV1J24y1c7kE"

# should you use javascript to capture the webpage, not here
# cause it is error prone here

max_page_limit = 5


def random_character_generator():
    return random.choice(string.ascii_letters + string.digits)


def handle_page_event(page: Page):
    createAndExposePageIdentifierAsFunctionName(page)
    register_io_event_handler(page)
    print("new page at:", page.url)

def register_io_event_handler(page:Page):
    page.on('filechooser', lambda file_chooser: file_chooser.set_files([])) # not to select a thing
    # page.on('filechooser', lambda file_chooser: file_chooser.set_files(os.path.abspath('pointer_tracing.html')))
    page.on('download', lambda download: download.cancel()) # not download anything.
    # page.on('dialog', lambda e: None)
    # let's not try to understand what is going on here.

def register_page_exit_event_handler(page:Page):
    page.on('close', lambda e: None)
    page.on('crash', lambda e: None)
    page.on('pageerror', lambda e: None)


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
        if counter % counter_threshold == 0:
            page.bring_to_front()
            page.screenshot(
                path=f"page_{index}.png", timeout=screenshot_timeout
            )  # taking screenshot time exceeded?
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

google_chrome = r"C:\Users\z98hu\AppData\Local\Google\Chrome\Application\chrome.exe"  # let's play video.
with sync_playwright() as playwright:
    # use persistent context to load extensions.
    context = playwright.chromium.launch_persistent_context(
        "", # what does this mean?
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

    # context = browser.new_context()
    context.on("page", handle_page_event)
    # keep clicking buttons might initiate download or uploading events
    # you need to prevent that.

    init_page = context.new_page()
    init_page.goto(url)
    # createAndExposePageIdentifierAsFunctionName(init_page)
    handle_page_event(init_page)

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
