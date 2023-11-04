# do we need a new dimension?
# pip install positional-encodings[pytorch]

import torch
import einops
from positional_encodings.torch_encodings import PositionalEncoding2D
import torch.nn.functional as F
import matplotlib.pyplot as plt

# according to the formula, shall we recalculate or interpolate the encodings?
# or we just downscale the picture?

# we will view the thing.

# it is not gaussian.
channel_count = 3
batch_size = 1

original_image_width_or_height = 256
# original_image_width_or_height = 1024
scale_factor = 0.5
scaled_image_width_or_height = int(original_image_width_or_height * scale_factor)

input_tensor_shape = (
    batch_size,
    original_image_width_or_height,
    original_image_width_or_height,
    channel_count,
)

input_tensor = torch.zeros(input_tensor_shape)

# how to rescale the thing?

posenc_2d = PositionalEncoding2D(channel_count)
output_tensor = posenc_2d(input_tensor)
output_tensor_rearranged = einops.rearrange(output_tensor, "b h w c -> b c h w")

new_size = (scaled_image_width_or_height, scaled_image_width_or_height)

# first check the result of the rescaled tensor.
interpolated_tensor = F.interpolate(
    output_tensor_rearranged, size=new_size, mode="bilinear", align_corners=False
)

print(f"Original tensor shape: {input_tensor.shape}")
print(f"Output tensor shape: {output_tensor.shape}")
print(f"Output tensor (rearranged) shape: {output_tensor_rearranged.shape}")
print(f"Interpolated tensor shape: {interpolated_tensor.shape}")

image = output_tensor_rearranged.numpy()
image = einops.rearrange(image, "b c h w -> b h w c")
# now, view the tensor.

# Normalize the image between 0 and 1
image = (image - image.min()) / (image.max() - image.min())

# Display the image using matplotlib
plt.imshow(image[0, :, :, :])
# plt.imshow(image)
plt.axis("off")  # Remove axis ticks and labels
plt.show()
