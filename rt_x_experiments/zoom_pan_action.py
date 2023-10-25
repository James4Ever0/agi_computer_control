# since we can only handle specific data in the scope, how about take transformations as actions
# distinguish between discrete and continuous actions, like keyboard strokes, clicks and movements.

import numpy as np


def crop_at_location(img, location, size):
    lx, ly = location
    rx, ry = lx + size, ly + size
    cropped_img = img[:, lx:rx, ly:ry]
    return cropped_img


def relative_move_crop(img, location, size, dx, dy):
    lx, ly = location
    nx, ny = lx + dx, ly + dy
    new_location = (nx, ny)
    cropped_img = crop_at_location(img, new_location, size)
    return cropped_img


if __name__ == "__main__":
    img = np.zeros((3, 1920, 1080))
    cropped_img = crop_at_location(img, (10, 10), 224)
    print("%s -> %s" % (img.shape, cropped_img.shape))
