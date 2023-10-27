# contrary to external/environmental attention mechanism such as adjusting the zoom level, we use internal bisect multihead mechanism instead.

# first let's define the patch size, 256x256 pixels, and anything larger than that will be downscaled.
# we will extract the attended area and bisect. we feed it again into the network and recurse.