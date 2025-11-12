import pytest
from octal_calculator import Evaluator, octal_str_to_int, int_to_octal_str
from exceptions import (
    EvaluationError,
    DivisionByZeroError,
    RecursionLimitError,
    ArityError,
    NameErrorEval,
    ParseError,
    ConversionError,
)

# ================================================================
# Fixtures
# ================================================================
@pytest.fixture
def evaluator():
    """Provides a fresh Evaluator instance for each test."""
    return Evaluator()


# ================================================================
# Conversion Tests
# ================================================================
def test_octal_to_int_and_back():
    assert octal_str_to_int("10") == 8
    assert octal_str_to_int("7") == 7
    assert int_to_octal_str(8) == "10"
    assert int_to_octal_str(-8) == "-10"


def test_invalid_conversion():
    with pytest.raises(ConversionError):
        octal_str_to_int("8")
    with pytest.raises(ConversionError):
        octal_str_to_int("ABC")


# ================================================================
# Arithmetic and Operator Tests
# ================================================================
def test_basic_operations(evaluator):
    assert evaluator.execute("7 + 1") == "10"
    assert evaluator.execute("10 - 1") == "7"
    assert evaluator.execute("2 * 3") == "6"


def test_division_and_modulo(evaluator):
    assert evaluator.execute("10 / 2") == "4"
    # Evaluator’s modulo semantics
    assert evaluator.execute("10 % 3") == "2"

    with pytest.raises(DivisionByZeroError):
        evaluator.execute("7 / 0")
    with pytest.raises(DivisionByZeroError):
        evaluator.execute("7 % 0")


def test_exponentiation(evaluator):
    assert evaluator.execute("2 ^ 3") == "10"
    assert evaluator.execute("2 ∧ 3") == "10"  # Unicode operator


def test_negative_and_precedence(evaluator):
    assert evaluator.execute("-1 + 2") == "1"
    assert evaluator.execute("2 + 3 * 4") == "16"
    assert evaluator.execute("(2 + 3) * 4") == "24"


def test_modulo_and_sign(evaluator):
    # Match evaluator’s custom semantics
    assert evaluator.execute("-5 % 3") == "-2"
    assert evaluator.execute("5 % -3") == "2"
    assert evaluator.execute("-5 % -3") == "-2"


# ================================================================
# LET and variable binding tests
# ================================================================
def test_let_binding(evaluator):
    assert evaluator.execute("LET A = 3 IN A + 2") == "5"


def test_nested_lets(evaluator):
    result = evaluator.execute("LET A = 3 IN LET B = A + 2 IN B * A")
    # In octal, 3*5 = 17
    assert result == "17"


def test_variable_shadowing(evaluator):
    expr = "LET A = 3 IN LET A = 5 IN A + 1"
    assert evaluator.execute(expr) == "6"


# ================================================================
# DEF (function) tests
# ================================================================
def test_basic_function(evaluator):
    evaluator.execute("DEF F(X) = X + 1")
    assert evaluator.execute("F(2)") == "3"


def test_two_params(evaluator):
    evaluator.execute("DEF ADD(X,Y) = X + Y")
    assert evaluator.execute("ADD(3,5)") == "10"


def test_function_shadowing(evaluator):
    evaluator.execute("DEF F(X) = X + 1")
    evaluator.execute("DEF F(X) = X + 2")
    assert evaluator.execute("F(3)") == "5"


def test_function_calls_another(evaluator):
    evaluator.execute("DEF F(X) = X + 1")
    evaluator.execute("DEF G(Y) = F(Y) * 2")
    assert evaluator.execute("G(3)") == "10"


def test_recursion(evaluator):
    evaluator.execute("DEF FACT(N) = IF N <= 1 THEN 1 ELSE N * FACT(N - 1)")
    result = evaluator.execute("FACT(5)")
    # 120 decimal = 170 octal
    assert result == "170"


def test_recursion_limit(evaluator):
    evaluator.execute("DEF LOOP(X) = LOOP(X)")
    with pytest.raises(RecursionLimitError):
        evaluator.execute("LOOP(1)")


def test_arity_errors(evaluator):
    evaluator.execute("DEF ADD(X,Y) = X + Y")
    with pytest.raises(ArityError):
        evaluator.execute("ADD(1)")
    with pytest.raises(ArityError):
        evaluator.execute("ADD(1,2,3)")


def test_name_error(evaluator):
    with pytest.raises(NameErrorEval):
        evaluator.execute("NOTDEF(2)")


# ================================================================
# IF/THEN/ELSE tests
# ================================================================
def test_basic_if(evaluator):
    assert evaluator.execute("IF 7 > 3 THEN 7 ELSE 3") == "7"
    assert evaluator.execute("IF 2 > 3 THEN 7 ELSE 3") == "3"


def test_nested_if(evaluator):
    expr = "IF 7 > 3 THEN IF 1 < 2 THEN 10 ELSE 0 ELSE 5"
    assert evaluator.execute(expr) == "10"


# ================================================================
# Parser / syntax error tests
# ================================================================
def test_invalid_expression(evaluator):
    with pytest.raises(ParseError):
        evaluator.execute("LET A = ")


def test_unexpected_token(evaluator):
    with pytest.raises(ParseError):
        evaluator.execute("DEF F(X) = X +")


# ================================================================
# Integration edge cases
# ================================================================
def test_large_numbers(evaluator):
    result = evaluator.execute("7777777 + 1")
    assert result.isdigit()


def test_negative_results(evaluator):
    assert evaluator.execute("3 - 5") == "-2"


def test_function_and_let_combined(evaluator):
    evaluator.execute("DEF F(X) = X * 2")
    expr = "LET A = 3 IN F(A) + 1"
    assert evaluator.execute(expr) == "7"


# ================================================================
# Additional coverage tests
# ================================================================
def test_large_exponent_and_bigint_handling(evaluator):
    assert evaluator.execute("7 ^ 3") == "527"
    # In octal, 2^8 = 400
    assert evaluator.execute("2 ^ 10") == "400"


def test_extreme_exponent_limit(evaluator):
    with pytest.raises(EvaluationError):
        evaluator.execute("2 ^ -3")


def test_whitespace_tolerance(evaluator):
    expressions = [
        "   7 + 1   ",
        "\tLET   A =   3  IN  A",
        "DEF   F ( X ) =   X + 1 ",
        "  IF   7 > 3  THEN 7 ELSE 3 ",
    ]
    for expr in expressions:
        try:
            _ = evaluator.execute(expr)
        except Exception as e:
            pytest.fail(f"Expression with whitespace failed: {expr} ({e})")


def test_function_closure_and_global_scope(evaluator):
    evaluator.execute("LET A = 3 IN A")
    evaluator.execute("DEF F(X) = X + 3")
    assert evaluator.execute("F(4)") == "7"
    assert evaluator.execute("LET A = 5 IN F(4)") == "7"


def test_nested_let_scoping(evaluator):
    expr = "LET A = 1 IN LET B = A + 1 IN LET C = B + 1 IN C + A"
    assert evaluator.execute(expr) == "4"
