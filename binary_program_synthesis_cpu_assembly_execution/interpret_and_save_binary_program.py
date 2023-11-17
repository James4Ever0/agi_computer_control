# filepath = "binary_program_windows_bin.txt"
filepath = "binary_program.txt"
# output_path = "program.bin"
# output_path = "program.exe"
output_path = "program" # linux program?

byte_int_list = []

with open(filepath, 'r') as f:
    content = f.read()
    byte_list = content.replace("\n", ' ').split()
    for byte_str in byte_list:
        byte_int = int(byte_str, 2)
        byte_int_list.append(byte_int)

binary_bytes = bytes(byte_int_list)
with open(output_path, 'wb') as f:
    f.write(binary_bytes)
# import os
# os.system(output_path)