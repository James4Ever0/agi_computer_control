code_path = "array_static_typecheck.py"

import ast
import rich

with open(code_path, "r") as f:
    content = f.read()
    tree = ast.parse(content)
    for el in tree.body:
        rich.print(el.__dict__)
        # rich.print(dir(el))
        print(ann:=getattr(el, "annotation", None))
        if ann:
            print(ast.unparse(el), el) # ast.AnnAssign
