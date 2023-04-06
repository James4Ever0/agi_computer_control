from PIL import Image
import mss

screenshot_factory = mss.mss()

image = screenshot_factory.grab(screenshot_factory.monitors[0])

load_image = Image.frombytes("RGB", image.size, bytes(image.raw), "raw", "BGRX")

load_image.show()