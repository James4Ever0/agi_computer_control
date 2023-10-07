image_path = "vscode_screenshot.png"

from ascii_magic import AsciiArt
columns = 60

my_art = AsciiArt.from_image(image_path)
# str_art = my_art.to_terminal(columns=columns, monochrome=True)
str_art = my_art._img_to_art(columns=columns, monochrome=True)
print('artwork:')
print()
print(str_art)

# then we need to visualize it.
import pytesseract

from PIL import Image

image = Image.open(image_path)

extracted_text = pytesseract.image_to_string(image)

print("extracted text:")
print(extracted_text)