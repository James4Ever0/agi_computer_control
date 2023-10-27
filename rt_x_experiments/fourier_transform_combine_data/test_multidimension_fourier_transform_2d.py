from test_common import *


axes = [1,2]

# now you can choose to do fft over 1d or 2d
# what a problem.

fft1 = np.fft.fft2(vit_encoding_1, axes=axes)
fft2 = np.fft.fft2(vit_encoding_2, axes=axes)

fft_sum = fft1+fft2
# how about let's use elementwise multiplication to replace the addition?
fft_mul = fft1*fft2

fft_sum_and_mul = fft_sum+fft_mul

vit_final = np.fft.ifft2(fft_sum, axes = axes)
vit_final_real = vit_final.real

vit_final_mul_real = np.fft.ifft2(fft_mul, axes = axes).real
vit_final_sum_and_mul_real = np.fft.ifft2(fft_sum_and_mul, axes = axes).real

# print(vit_final)
print(fft_sum)
print(vit_final_real)
print(vit_final_mul_real)
print(vit_final_sum_and_mul_real)
print(vit_encoding_1.shape, vit_final_real.shape) # shape is the same. however, we have strange imaginary parts. let's discard them.

# now we can just sum. it does not have to be complex.