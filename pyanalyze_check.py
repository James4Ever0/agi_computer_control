from pyanalyze.extensions import CustomCheck
import pyanalyze

class LiteralOnly(CustomCheck):
    def can_assign(self, value: "Value", ctx: "CanAssignContext") -> "CanAssign":
        for subval in pyanalyze.value.flatten_values(value):
            if not isinstance(subval, pyanalyze.value.KnownValue):
                return pyanalyze.value.CanAssignError("Value must be a literal")
        return {}