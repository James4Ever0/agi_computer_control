input_path = (
    "./resources/decompressed/Kcom3资料/keys.txt"
)

from typing import Annotated
# import beartype
from beartype.vale import Is
from beartype.door import is_bearable

null_string = Annotated[str, Is[lambda s: len(s) == 0]]
table_entry = Annotated[str, Is[lambda s: len(s) < 30]]

table_header_count = 7
table_rows = []
row = []
header_index = -1
with open(input_path, "r", encoding="utf-8") as f:
    for line in f.readlines():
        line = line.strip()

        if is_bearable(line, null_string):
            if header_index == table_header_count-1:
                table_rows.append(row.copy())
            row = []
            header_index=-1
        elif is_bearable(line, table_entry):
            if header_index !=-2:
                header_index +=1
                row.append(line)
        else: # too long
            header_index = -2

# import rich

# rich.print(table_rows)

import pandas

df = pandas.DataFrame(table_rows[1:], columns = table_rows[0])
print(df)