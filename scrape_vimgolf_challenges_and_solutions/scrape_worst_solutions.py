import os
import json
from playwright.sync_api import sync_playwright
import playwright._impl._errors


def init_playwright_server():
    stop_playwright_server()
    start_playwright_server()


def start_playwright_server():
    start_command = 'docker run -d -p 3021:3000 --rm --init -it --ipc=host --name vimgolf_scraper cybergod_playwright_server /bin/sh -c "playwright run-server --port 3000 --host 0.0.0.0"'
    os.system(start_command)
    # wait till ready
    os.system("sleep 5")


def stop_playwright_server():
    stop_command = "docker rm -f vimgolf_scraper"
    os.system(stop_command)


def scrape_worst_solution(playwright_server_url: str, url: str):
    with sync_playwright() as p:
        # browser = p.chromium.launch(
        #     headless=False, proxy={"server": "http://127.0.0.1:7897"} # run with xvfb-run
        # )
        print("Connecting to remote playwright server:", playwright_server_url)
        browser = p.chromium.connect(playwright_server_url)
        print("Creating new page")
        page = browser.new_page()
        print("Visit URL:", url)
        page.goto(url, wait_until="commit")
        print("Waiting for selector")
        page.wait_for_selector("div.success.clearfix", timeout=60 * 1000)
        loc = page.locator("div.success.clearfix").first # sometimes there are two success divs with the same score, so we take the first one
        header_elem = loc.locator("h6")
        header = header_elem.inner_text()
        rank_elem = header_elem.locator("a.anchor")
        rank = rank_elem.get_attribute("name")
        solution_elem = loc.locator("pre")
        # note: the thing in the solution is not plain text.
        # we need to extract individual "token" elements
        solution = solution_elem.inner_text()[:-1]  # remove trailing newline
        ret = dict(rank=rank, solution=solution, header=header)
        return ret


def main(test=False):
    try:
        start_playwright_server()
        playwright_server_url = "ws://127.0.0.1:3021/"
        challenge_hashids = os.listdir("./challenges")

        for it in challenge_hashids:
            worst_solution_writepath = f"./challenges/{it}/worst_solution.json"
            if os.path.exists(worst_solution_writepath):
                print("Skipping scraped challenge solution:", it)
                continue
            print("Scraping worst solution for challenge:", it)
            url = f"https://www.vimgolf.com/challenges/{it}"
            while True:
                try:
                    solution = scrape_worst_solution(playwright_server_url, url)
                    break
                except playwright._impl._errors.TimeoutError:
                    print("Timeout error, retrying...")
                    continue
            if test:
                print(solution)
                continue
            with open(worst_solution_writepath, "w+") as f:
                f.write(json.dumps(solution))
    finally:
        stop_playwright_server()


def test_main():
    main(test=True)


if __name__ == "__main__":
    main()
    # test_main()
