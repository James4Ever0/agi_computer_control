from PIL import Image
# cursor_image = "mouse_cursor_white_16x16.png"
cursor_image = 'cursor.png'
cur = Image.open(cursor_image)
pos = (400,400)
pw = True
if pw:
    screen_image = "paint.png"
else:
    screen_image = "baidu_7.png"
img = Image.open(screen_image)

# client size: 992x647
# size: 2232x1456
# saved size: 2232x1456
if pw:
    new_img = img
else:
    new_img = img.resize((992,647)) # 992x647, without scrollbar.
# problem with scrollbar.
new_img.paste(cur, pos, cur) # misaligned.

new_img.show()