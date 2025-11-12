# exceptions.py

class OctalCalcError(Exception):
    """Base exception for octal calculator errors."""
    pass


class ParseError(OctalCalcError):
    """Raised when parsing fails."""
    def __init__(self, message, position=None):
        super().__init__(f"ParseError{(' at '+str(position)) if position is not None else ''}: {message}")
        self.position = position


class EvaluationError(OctalCalcError):
    """Raised when evaluation fails (runtime)."""
    pass


class DivisionByZeroError(EvaluationError):
    """Division or modulo by zero."""
    def __init__(self):
        super().__init__("Division or modulo by zero")


class RecursionLimitError(EvaluationError):
    """Recursive call depth exceeded."""
    def __init__(self, limit):
        super().__init__(f"Recursion depth limit exceeded ({limit})")
        self.limit = limit


class ArityError(EvaluationError):
    """Wrong number of arguments to a function."""
    def __init__(self, name, expected, got):
        super().__init__(f"Function '{name}' expected {expected} args, got {got}")


class NameErrorEval(EvaluationError):
    """Undefined variable or function."""
    def __init__(self, name):
        super().__init__(f"Name error: '{name}' is not defined")


class ConversionError(OctalCalcError):
    """Errors in octal string conversion."""
    def __init__(self, invalid_string):
        super().__init__(f"Invalid octal number: '{invalid_string}'")


class AssertionFailure(OctalCalcError):
    """Raised when an internal assertion fails."""
    def __init__(self, message):
        super().__init__(f"Assertion failed: {message}")
