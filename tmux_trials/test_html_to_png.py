from lib import html_to_png


def test():
    input_path = "/tmp/test_session_preview.html"
    output_path = "screenshot.png"

    with open(input_path, "r") as f:
        html = f.read()
    image_bytes = html_to_png(html)
    with open(output_path, "wb") as f:
        f.write(image_bytes)


if __name__ == "__main__":
    test()
