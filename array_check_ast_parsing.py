code_path = "array_static_typecheck.py"

import ast
import rich

with open(code_path, "r") as f:
    content = f.read()
    tree = ast.parse(content)
    for el in tree.body:
        # rich.print(dir(el))
        print(getattr(el, ""))
