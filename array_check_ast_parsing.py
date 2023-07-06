code_path = "array_static_typecheck.py"

import ast

with open(code_path, 'r') as f:
    content = f.read()
    tree = ast.parse(content)
    print(tree)