from test_common import *

axis = 2

fft1 = np.fft.fft(vit_encoding_1, axis=axis)
fft2 = np.fft.fft(vit_encoding_2, axis=axis)

fft_sum = fft1 + fft2

vit_final = np.fft.ifft(fft_sum, axis=axis)
vit_final_real = vit_final.real
print(fft_sum)
print(vit_final_real)
print(vit_encoding_1.shape, vit_final_real.shape)
