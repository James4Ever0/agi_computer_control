# according to ai, use imageio

# Pillow will need manual image pasting and patching

# ffmpeg can convert gif into image seqneces
# we might want to specify fps, skip duplicate frames
# ffmpeg -i input.gif frame_%04d.png

import imageio.v3 as iio

input_gif_path = "input.gif"

# Get total frames
total_frames = len(iio.imread(input_gif_path, index=None))

# Verify frame exists
if 150 < total_frames:
    frame = iio.imread(input_gif_path, index=150)  # Correct composited frame
else:
    print("Frame index out of range")