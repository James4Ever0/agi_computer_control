# contrary to external/environmental attention mechanism such as adjusting the zoom level, we use internal bisect multihead mechanism instead.

# first let's define the patch size, 256x256 pixels, and anything larger than that will be downscaled.
# we will extract the attended area and bisect. we feed it again into the network and recurse.

# while extracting the attended area, we will mask out the padding area for sure, to avoid crop being misplaced.

image_path = ...
max_zoom_level = 3 # should this be adjustable.

# if the attention center array is like: [(0, 0), (0, 0), (0, 0)]
# we will do cropping right at the center, for three times

# every number in attention center array shall be ranged from -1 to 1.

# so how do you combine these recursive embeddings? fft?

def crop_at_interested_area_recursive(image, attention_center_array:list[tuple[float,float]]):
    ret = image.copy()
    for center in attention_center_array:
        ret = crop_at_interested_area(ret, center)
    return ret

def check_if_numer_in_range(number:float, _min:float, _max:float):
    assert _min < _max
    assert number >= _min
    assert number <= _max

def crop_at_interested_area(image, attention_center: tuple[float, float]):
    x_c, y_c = attention_center
    assert check_if_numer_in_range(x_c, -1, 1)
    assert check_if_numer_in_range(y_c, -1, 1)
    _, x, y = image.shape()
    half_x, half_y = x//2, y//2
    quad_x, quad_y = half_x//2, half_y//2
    new_x = half_x + x_c*half_x
    new_y = half_y + y_c*half_y
    ret = image[:, new_x-quad_x: new_x+quad_x, new_y-quad_y: new_y+quad_y]
    return ret