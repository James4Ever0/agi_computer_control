import numpy as np

vit_encoding_gen = lambda: np.random.random((1, 64, 512))

# 64: attention heads.
# 512: feature size.

# maybe we should not do 2d
# do 1d instead?

# don't know. really. just try.

vit_encoding_1 = vit_encoding_gen()
vit_encoding_2 = vit_encoding_gen()

print(vit_encoding_1 + vit_encoding_2)

# what are you doing! fft -> ifft is the same as direct addition
