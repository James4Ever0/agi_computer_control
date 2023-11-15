# binaries? this reminds me of binary neural networks. whoa there!

import numpy as np

filepath = "README.md"
with open(filepath, "rb") as f:
    char = f.read(1)
    # arr = np.frombuffer(char, dtype=np.bool_) # this will not work.
    arr = np.frombuffer(char, dtype=np.uint8)
    # arr = np.array(list(char)).astype(np.uint8)
    # print(arr)
    # print(char)
    bin_array = np.unpackbits(arr).astype(np.bool_)
    print(bin_array)  # [False  True False False  True False False  True]
