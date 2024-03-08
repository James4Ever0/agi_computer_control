# binaries? this reminds me of binary neural networks. whoa there!

# import numpy as np

filepath = "README.md"
output_path = "converted_binary_view.txt"
# import binascii
# this can do the job of hex
import itertools
with open(filepath, "rb") as f:
    _bytes = f.read()
    # char = f.read(1)
    # arr = np.frombuffer(char, dtype=np.bool_) # this will not work.
    # arr = np.frombuffer(char, dtype=np.uint8)
    # arr = np.array(list(char)).astype(np.uint8)
    # print(arr)
    # print(char)
    # bin_array = np.unpackbits(arr)
    # bin_array = np.unpackbits(arr).astype(np.bool_)
    # print(bin_array)  # [False  True False False  True False False  True]
    bit_repr = [format(it, '08b') for it in _bytes]
    line_list = []
    for index, bit_group in itertools.groupby(enumerate(bit_repr), key=lambda x: x[0]//8):
        line = " ".join([it for _, it in bit_group])
        line_list.append(line)
    with open(output_path, 'w+') as fw:
        fw.write("\n".join(line_list))