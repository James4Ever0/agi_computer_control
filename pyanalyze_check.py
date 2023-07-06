from pyanalyze.extensions import CustomCheck
from pyanalyze.symbols import SymbolTable
from pyanalyze import
class MyCustomCheck(CustomCheck):
    name = "my_custom_check"  # A unique name for the check

    def run(self, symtable: SymbolTable) -> None:
        # Define your check here
        # `symtable` is a `SymbolTable` object that contains information about the symbols in the code
        # You can use `self.warn()` or `self.error()` to report errors or warnings

        for node in symtable.ast_node_iter():
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                self.warn("Use of print statement", node)

# Run the check on a Python file
from pyanalyze.checker import run_checkers_on_files
run_checkers_on_files(["myfile.py"], [MyCustomCheck()])