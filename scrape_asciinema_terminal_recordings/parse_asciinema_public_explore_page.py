from bs4 import BeautifulSoup


def test():
    html_filepath = (
        "nonexistant_page_as_last_page_asciinema_explore_public_order_by_date.html"
    )

    with open(html_filepath, "r") as html_file:
        html_content = html_file.read()

    for _ in parse_asciinema_explore_public_page(html_content):
        ...


def parse_asciinema_explore_public_page(html_content: str):
    soup = BeautifulSoup(html_content, "lxml")

    # find the "active page" aka the real page number
    active_page_li_elem = soup.find("li", class_="active page-item")

    if active_page_li_elem:
        active_page_num = active_page_li_elem.text
        print(f"The active page number is: {active_page_num}")
        yield dict(active_page_num=active_page_num)
    else:
        # probably this would never happen. but if it does, could be rate limit, cloudflare waf.
        print("Active page num not found.")
        yield

    # now, find all div with class_="info"
    # inside each div, get the href, title, duration, author and submit time

    # TODO: wrap each element get method into try-except blocks, and if failed, just assign each attribute with None
    for div_info_elem in soup.find_all("div", class_="info"):
        try:
            a_tag = div_info_elem.find("a")
            href = a_tag["href"] if a_tag and a_tag.has_attr("href") else None
        except Exception:
            href = None

        try:
            a_tag = div_info_elem.find("a")
            title = a_tag.text if a_tag else None
        except Exception:
            title = None

        try:
            duration_span = div_info_elem.find("span", class_="duration")
            duration = duration_span.text if duration_span else None
        except Exception:
            duration = None

        try:
            author_span = div_info_elem.find("span", class_="author-avatar")
            if author_span:
                author_a = author_span.find("a")
                author = (
                    author_a["title"]
                    if author_a and author_a.has_attr("title")
                    else None
                )
            else:
                author = None
        except Exception:
            author = None

        try:
            small_tag = div_info_elem.find("small")
            if small_tag:
                time_tag = small_tag.find("time")
                submit_time = (
                    time_tag["datetime"]
                    if time_tag and time_tag.has_attr("datetime")
                    else None
                )
            else:
                submit_time = None
        except Exception:
            submit_time = None

        print()
        print("Href:", href)
        print("Title:", title)
        print("Duration:", duration)
        print("Author:", author)
        print("Submit time:", submit_time)
        yield dict(
            href=href,
            title=title,
            duration=duration,
            author=author,
            submit_time=submit_time,
        )


if __name__ == "__main__":
    test()
