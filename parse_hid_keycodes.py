input_path = (
    "hardware_capture_hid_power_control/resources/decompressed/Kcom3资料/keys.txt"
)

from typing import Annotated
import beartype
from beartype.vale import Is
from beartype.door import is_bearable

null_string = Annotated[str, Is[lambda s: len(s) == 0]]
table_entry = Annotated[str, Is[lambda s: len(s) > 0 and len(s) < 30]]

table_header_count = 7


with open(input_path, "r", encoding="utf-8") as f:
    for line in f.readlines():
        line = line.strip()

        if is_bearable(line, null_string):
            header_index=-1
        elif is_bearable(line, table_entry):
            header_index +=1