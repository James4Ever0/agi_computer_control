input_path = "hardware_capture_hid_power_control/resources/decompressed/Kcom3资料/keys.txt"

from typing import Annotated
import beartype
from beartype.vale import Is

null_string = Annotated[str, Is[lambda s: len(s) == 0]]
table_entry = Annotated[str, Is[lambda s: len(s) >0 and len(s)]]

with open(input_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        