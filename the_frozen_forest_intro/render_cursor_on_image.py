from PIL.Image import Image as ImageClass

from PIL import Image

# typing issue reference:
# https://github.com/microsoft/pylance-release/issues/5313

# cursor_image = "mouse_cursor_white_16x16.png"

# now for the fun part. how do we record events in browser?
# listen to keyboard and mouse events.

# so, would you let me know where the cursur is at?

# client size: 992x647
# size: 2232x1456
# saved size: 2232x1456


def convert_cursor_in_red(cursor: ImageClass):
    red = (255, 0, 0)
    cursor_red = Image.new('RGB', cursor.size, red)

    # Paste the red image onto the black and white image, using the black and white image as a mask
    cursor_red.paste(cursor_red, mask=cursor)

    return cursor_red

def convert_to_grayscale(image: ImageClass) -> ImageClass:
    # Convert the image to grayscale
    grayscale = image.convert('L')

    # Convert the grayscale image back to RGB, while keeping the grayscale values in all three color channels
    grayscale_rgb = Image.merge('RGB', (grayscale, grayscale, grayscale))

    return grayscale_rgb
def place_cursor_to_screen(cursor:ImageClass, screen:ImageClass, position:tuple[int,int], red_on_grey:bool): 
    mask = cursor.copy()
    if red_on_grey:
        cursor = convert_cursor_in_red(cursor)
        screen = convert_to_grayscale(screen)
    ret = screen.copy()
    ret.paste(cursor, position, mask)
    return ret
    

def test(red_on_grey:bool):
    position = (400, 400)
    
    cursor_image_path = "cursor.png"

    screen_image_path = "paint.png"
    # screen_image_path = "baidu_4.png"
    # screen_image_path = "baidu_7.png"

    cursor_image = Image.open(cursor_image_path)
    screen_image = Image.open(screen_image_path).resize((992,647))
    
    output_path = "image_with_cursor.png"
    
    output_image = place_cursor_to_screen(cursor_image, screen_image, position, red_on_grey)
    
    print("[*] Saving to:", output_path)
    output_image.save(output_path)

if __name__ == "__main__":
    # test(False)
    test(True)
