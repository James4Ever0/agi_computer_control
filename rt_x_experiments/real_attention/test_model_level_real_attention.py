# contrary to external/environmental attention mechanism such as adjusting the zoom level, we use internal bisect multihead mechanism instead.

# first let's define the patch size, 256x256 pixels, and anything larger than that will be downscaled.
# we will extract the attended area and bisect. we feed it again into the network and recurse.

image_path = ...
max_zoom_level = 3 # should this be adjustable.

# if the attention center array is like: [0, 0, 0] we will do cropping right at the center, for three times

def crop_at_interested_area(original_image, attention_center_array:list[float]):...