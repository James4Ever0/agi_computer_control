from playwright.sync_api import sync_playwright
TERMINAL_VIEWPORT={"width":645,"height":350}

def html_to_png(input_path:str, output_path:str):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport=TERMINAL_VIEWPORT) # type: ignore
        page = context.new_page()
        with open(input_path, "r") as f:
            content = f.read()
        page.set_content(content)
        page.viewport_size
        page.screenshot(path=output_path)
        browser.close()

if __name__ == "__main__":
    input_path = "/tmp/test_session_preview.html"
    output_path = "screenshot.png"

    html_to_png(input_path, output_path)