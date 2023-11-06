# the generalist of all.
# doc: https://g1879.gitee.io/drissionpagedocs

# let's offer the machine way back, ability to move around, delete, going back pages and cancel actions.

# do we need to render the cursor at its location? or do we just need to check the size of the page screenshot?

from drissionpage_common import *

random_url = "https://www.baidu.com"
page.get(random_url)

from DrissionPage.action_chains import ActionChains

# shall we design some calibration page for mouse location, and view that before running.

# same as our virtual machine. we can calibrate that, shall we?

ac = ActionChains(page)
ac.type('hello world\n')

# i have found something i don't want to do.
# the coordinates!
# i do not want to calibrate it at all.

# where are you clicking?

ac.move_to((200,200)).click()
ac.move_to((400,400)).click().type('hello world\n')

print('action chain location:', ac.curr_x, ac.curr_y)
view_width, view_height = page.rect.viewport_size # no scrollbar
# this is obtained by running javascript. inaccurate.
# view_ws_width, view_ws_height = page.rect.viewport_size_with_scrollbar # with scrollbar
# page_width, page_height = page.size

image_save_path = 'baidu.png' # it is not the true path, if executed repeatedly..
actual_save_path = page.get_screenshot(image_save_path)
print(f'actual path: {actual_save_path}')

print(f"client size: {page.run_js('return document.body.clientWidth;')}x{page.run_js('return document.body.clientHeight;')}") # client size: 1232x4033
# it seems not caring about our actual viewpoint location.

from PIL import Image
img = Image.open(actual_save_path)

# size: 2772x8753
# saved size: 2772x1526
# it is just the viewport.
print(f"size: {view_width}x{view_height}") # 2738x1526
# print(f"size with scrollbar: {view_ws_width}x{view_ws_height}") # 1232x678
# print(f"size: {page_width}x{page_height}") # on yoga14s: 2813x1492

# you can crop at the top left corner to get rid of scrollbars. you can also detect their presence.
print(f'saved size: {"x".join(str(s) for s in img.size)}') # 2772x1526

import time

QUIT_SECONDS = 10
print(f"quit in {QUIT_SECONDS} seconds")
time.sleep(QUIT_SECONDS)

page.quit()
print("browser quit")
