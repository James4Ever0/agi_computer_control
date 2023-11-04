import torch

# you can also use fft to further enhance the performance, if it supports autograd

original_height_or_width = 1024
rank = 2

# ifft?

mat1 = torch.zeros((original_height_or_width, rank))
mat2 = torch.zeros((rank, original_height_or_width))
posenc_real= mat1 @ mat2 # instead of elementwise multiplication

mat3 = torch.zeros((original_height_or_width, rank))
mat4 = torch.zeros((rank, original_height_or_width))
posenc_imag = mat3 @ mat4

# posenc_fft = torch.fft.fft2(posenc)
posenc = posenc_real + 1j * posenc_imag
posenc_final = torch.fft.ifft2(posenc) # complex number
# print(posenc.shape)
print(posenc.real)
# print(posenc_final)
# torch.Size([1024, 1024])
# picture + (mat1 * mat2 = positional_encoding)

# if i insert something different into the model output, like 'read forward' or 'write forward', line up actions with perceptions, maybe the model will learn more.