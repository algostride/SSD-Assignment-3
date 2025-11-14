# Octal Calculator

Files:
- exceptions.py : custom exception classes
- octal_calculator.py : main implementation
- test_cases.py : tests

Requirements & Assumptions:
- Python 3.8+
- pytest 9.0+
- All numeric literals in input must be valid octal strings (digits 0-7).
- Exponentiation operator: ^ or Unicode ∧ (both supported).
- DEF statements persist in the in-memory registry for the session.
- Function definitions return 0 when evaluated as an expression.

Run tests:
    pytest -v

Notes:
- This project implements manual octal<->decimal conversion without using oct() or int(x,8).
- Recursion depth is limited to 1000 calls.
