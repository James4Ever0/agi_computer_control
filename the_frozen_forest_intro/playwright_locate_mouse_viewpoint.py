from playwright.sync_api import Playwright, sync_playwright, expect

paint_url = "https://paint.js.org/"
screenshot_path = "paint.png"
from PIL import Image

def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto(paint_url)
    print(f'viewport: {page.viewport_size["width"]}x{page.viewport_size["height"]}')
    page.mouse.move(200, 200)
    page.mouse.down()
    page.mouse.move(400, 400) # working.
    page.mouse.up()
    page.screenshot(path=screenshot_path) # will not create new file, same size as saved file

    img = Image.open(screenshot_path)
    print(f'image size: {img.size[0]}x{img.size[1]}') # 1280x720

    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)
