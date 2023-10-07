# you can input commands here.
# after input, you may take screenshot and get it as text.
# now you can move, click, and type.

import fastapi
import uvicorn
from port_util import port
import pyautogui
from PIL import Image

pyautogui.FAILSAFE = False

cursor_image = "cursor.png"

cur = Image.open(cursor_image)


def screenshot_with_cursor():
    shot = pyautogui.screenshot()
    pos = pyautogui.position()
    shot.paste(cur, pos, cur)
    return shot


from ascii_magic import AsciiArt
import pytesseract


def image_to_ascii(img: Image, columns=60):
    art = AsciiArt.from_pillow_image(img)
    ascii_text = art._img_to_art(columns=columns, monochrome=True)
    return ascii_text


def image_to_words(img: Image):
    words = pytesseract.image_to_string(img)
    return words


def image_to_ascii_and_words(img: Image):
    ascii_text = image_to_ascii(img)
    words = image_to_words(img)

    text = f"""
Ascii image:

{ascii_text}

Text in image:

{words}
"""
    return text


app = fastapi.FastAPI()


@app.get("/position")
def get_position():
    return pyautogui.position()


@app.get("/resolution")
def get_resolution():
    return pyautogui.size()


@app.get("/text_screenshot")
def get_text_screenshot():
    shot = screenshot_with_cursor()
    text = image_to_ascii_and_words(shot)
    return text


@app.get("/move_abs")
def move_cursor_abs(x: int, y: int):
    pyautogui.moveTo(x, y)


@app.get("/move_rel")
def move_cursor_rel(x: int, y: int):
    pyautogui.moveRel(x, y)


from typing import Literal


@app.get("/click")
def click_cursor(button: Literal["left", "right", "middle"]):
    pyautogui.click(button=button)


@app.get("/type")
def type_text(text: str):
    pyautogui.typewrite(text)


@app.get("/write")
def type_text(text: str):
    pyautogui.write(text)


@app.get("/scroll")
def scroll_down(x: float, y: float, clicks: float):
    pyautogui.scroll(clicks=clicks, x=x, y=y)


if __name__ == "__main__":
    host = "0.0.0.0"
    print("gui server running at:", f"http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
