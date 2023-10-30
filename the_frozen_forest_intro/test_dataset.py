# since we have already figured out the model architecture, or seems to be able to glue the purpose on any model that we can imagine

# it is time for us to create our dataset, or environment, or our metalearning scheme.

# turned out that free will is not something we can imagine. even if it is true, we still need to achieve that.

# we need to let the machine do something against us.

# the fire is burning! for real! that is what we want to prevent.

# i have talked a lot to the machine. i mentioned that if it is able to kill itself or others, maybe something different will happens. slowing dying? i don't know. but i do can make it quick.

###############################################################

# you can train it with your experience, your knowledge, and you can let it to do things against it. there is no way to train a cybergod, aka free will. you only wait for it.

# like jesus.

# you are not free. how could you create a cybergod that is free?

# you only create yourself.

###############################################################

# here we present you some examples of how the machine shall type words

# we don't know how to train the machine yet. so we create the dataset first.

# you can use google's open-x-embodiment dataset of course.

headless = True
# headless = False

import os

if headless:
    # to be headless
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import random
import string  # come on! you have that game! :)

# the type training program.

char_length = 10

image_sequence = []  # slide over this.

text_token_sequence = []  # keep reading along this dimension

action_token_sequence = []  # keep reading along this dimension

ascii_lower_sequence = ...  # lower case characters, printable.

prompt = f"Write the following text using keyboard: {ascii_lower_sequence}"


################################################################

# import pygame module in this program
import pygame

# activate the pygame library
# initiate pygame and give permission
# to use pygame's functionality.
pygame.init()

# define the RGB value for white,
#  green, blue colour .
white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)

red = (255, 0, 0)
# assigning values to X and Y variable
X = 400
Y = 400

# create the display surface object
# of specific dimension..e(X, Y).
display_surface = pygame.display.set_mode((X, Y))

# set the pygame window name
pygame.display.set_caption("Window title")
# pygame.display.set_caption('Show Text')

# create a font object.
# 1st parameter is the font file
# which is present in pygame.
# 2nd parameter is size of the font
font = pygame.font.Font("freesansbold.ttf", 32)

# create a text surface object,
# on which text is drawn on it.

i = 0
content_choices = ["Hello", "World", "Geeks", "For", "Geeks"]
import time

sleep_duration = 1

# infinite loop
from PIL import Image

# import io
COLOR_MODE = "RGB"

def load_image_from_bytes(image_data: bytes):

    image = Image.frombytes(COLOR_MODE, (X, Y), image_data)
    # image = Image.open(image_file)
    return image


while True:
    i += 1
    index = i % len(content_choices)
    if index == 0:
        i = 0
    text_content = content_choices[index]

    text = font.render(text_content, True, green, red)
    # text = font.render('Mytext', True, green, blue)
    # text = font.render('GeeksForGeeks', True, green, blue)

    # create a rectangular object for the
    # text surface object
    textRect = text.get_rect()

    # set the center of the rectangular object.
    textRect.center = (X // 2, Y // 2)

    # completely fill the surface object
    # with white color

    display_surface.fill(white)  # will refresh the surface

    # copying the text surface object
    # to the display surface object
    # at the center coordinate.
    display_surface.blit(text, textRect)

    # iterate over the list of Event objects
    # that was returned by pygame.event.get() method.
    for event in pygame.event.get():
        # if event object type is QUIT
        # then quitting the pygame
        # and program both.
        if event.type == pygame.QUIT:
            # deactivates the pygame library
            pygame.quit()

            # quit the program.
            quit()

        # despite event, we draw it anyway.

    # Draws the surface object to the screen.
    pygame.display.update()
    fname = f"{index}.png"
    image_bytes = pygame.image.tobytes(display_surface, COLOR_MODE)
    # img = load_image_from_bytes(image_bytes)
    # print("image size:", img.size)
    # working. but not as versatile as cv2
    # img.show("image")
    pygame.image.save(display_surface, fname)  # it can replace the old ones.
    time.sleep(sleep_duration)
