# contrary to external/environmental attention mechanism such as adjusting the zoom level, we use internal bisect multihead mechanism instead.

# first let's define the patch size, 256x256 pixels, and anything larger than that will be downscaled.
# we will extract the attended area and bisect. we feed it again into the network and recurse.

# while extracting the attended area, we will mask out the padding area for sure, to avoid crop being misplaced.

image_path = ...
max_zoom_level = 3  # should this be adjustable.

# if the attention center array is like: [(0, 0), (0, 0), (0, 0)]
# we will do cropping right at the center, for three times

# every number in attention center array shall be ranged from -1 to 1.

# so how do you combine these recursive embeddings? fft?


def crop_at_interested_area_recursive(
    image, attention_center_array: list[tuple[float, float]]
):
    ret = image.copy()
    for center in attention_center_array:
        ret = crop_at_interested_area(ret, center)
    return ret


def check_if_numer_in_range(number: float, _min: float, _max: float):
    assert _min < _max
    assert number >= _min
    assert number <= _max


def crop_at_interested_area(image, attention_center: tuple[float, float]):
    x_c, y_c = attention_center
    assert check_if_numer_in_range(x_c, -1, 1)
    assert check_if_numer_in_range(y_c, -1, 1)
    _, x, y = image.shape()
    half_x, half_y = x // 2, y // 2
    quad_x, quad_y = half_x // 2, half_y // 2
    new_x = half_x + x_c * half_x
    new_y = half_y + y_c * half_y
    ret = image[:, new_x - quad_x : new_x + quad_x, new_y - quad_y : new_y + quad_y]
    return ret


# use integral or convolution and select the max index, to reduce computation cost.
# if you want to use multihead or something like that, you would:
# 1 -> 1 -> 1 ...
# 1 -> 2 -> 4 ...
# 1 -> 3 -> 9 ...
import cv2
import numpy as np
from scipy.signal import convolve2d


def analyze_grayscale_image_and_get_crop_center(_grayscale_image):
    xs, ys = _grayscale_image.shape()
    x_size, y_size = xs // 2, ys // 2
    kernel = np.ones((x_size, y_size))
    grayscale_image = _grayscale_image.copy()
    convoluted_image = convolve2d(grayscale_image, kernel, mode="valid")
    min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(convoluted_image)
    left_corner = max_indx
    # now patch the attended area
    x_start = left_corner[0]
    y_start = left_corner[1]

    x_end = x_start + x_size
    y_end = y_start + y_size
    grayscale_image[x_start:x_end, y_start:y_end] = 0
    c_x = ((x_end - x_start) // 2 - x_size) / x_size
    c_y = ((y_end - y_start) // 2 - y_size) / y_size
    center = (c_x, c_y)
    return grayscale_image, center


# do it again.
def analyze_image_and_get_crop_center_list(image, center_count: int = 1):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    center_list = []
    for _ in range(center_count):
        grayscale_image, center = analyze_grayscale_image_and_get_crop_center(
            grayscale_image
        )
        center_list.append(center)
    return center_list
