# octal_calculator.py
"""
Octal Calculator
- Parses and evaluates expressions using the octal number system (base 8).
- Supports LET bindings, user-defined recursive functions (DEF), conditionals (IF/THEN/ELSE).
- No use of eval/exec, no int(...,8) or oct().
- All inputs are octal strings; outputs are octal strings.
"""

import re
from exceptions import (
    ParseError, EvaluationError, DivisionByZeroError, RecursionLimitError,
    ArityError, NameErrorEval, ConversionError, AssertionFailure
)

# -------------------------
# Configuration / Limits
# -------------------------
RECURSION_LIMIT = 1000

# -------------------------
# Utilities: Octal <-> int
# -------------------------
def octal_str_to_int(octal_string):
    """Convert an octal string (possibly with leading '-') to an integer.
       Manual conversion (no int(..., 8)). Raises ConversionError for invalid digits.
    """
    assert isinstance(octal_string, str)
    trimmed = octal_string.strip()
    if trimmed == '':
        raise ConversionError(octal_string)
    is_negative = False
    if trimmed[0] == '-':
        is_negative = True
        trimmed = trimmed[1:]
    if trimmed == '':
        raise ConversionError(octal_string)
    decimal_value = 0
    for ch in trimmed:
        if not ('0' <= ch <= '7'):
            raise ConversionError(octal_string)
        digit = ord(ch) - ord('0')
        if digit < 0 or digit > 7:
            raise ConversionError(octal_string)
        decimal_value = decimal_value * 8 + digit
    return -decimal_value if is_negative else decimal_value

def int_to_octal_str(integer_value):
    """Convert integer to octal string representation (no oct())."""
    assert isinstance(integer_value, int)
    if integer_value == 0:
        return "0"
    is_negative = integer_value < 0
    abs_value = -integer_value if is_negative else integer_value
    digits = []
    while abs_value > 0:
        d = abs_value % 8
        digits.append(chr(ord('0') + d))
        abs_value //= 8
    digits.reverse()
    s = ''.join(digits)
    return ('-' + s) if is_negative else s

# -------------------------
# Tokenizer
# -------------------------
TOKEN_SPEC = [
    ('NUMBER',   r'-?[0-7]+'),                          # octal integer (allow leading -)
    ('NAME',     r'[A-Za-z_][A-Za-z_0-9]*'),            # identifiers
    # include the unicode ∧ symbol explicitly so it tokenizes
    ('OP',       r'==|!=|<=|>=|∧|[+\-*/%^<>()=,%]'),    # operators and punctuation; '^' exponent
    ('SKIP',     r'[ \t]+'),
    ('NEWLINE',  r'[\r\n]+'),
]

TOKEN_RE = re.compile('|'.join('(?P<%s>%s)' % pair for pair in TOKEN_SPEC))
KEYWORDS = {'LET', 'IN', 'DEF', 'IF', 'THEN', 'ELSE'}

class Token:
    def __init__(self, type_, value, position):
        self.type = type_
        self.value = value
        self.pos = position
    def __repr__(self):
        return f"Token({self.type!r}, {self.value!r}, {self.pos})"

def tokenize(source_text):
    """Tokenize source_text into a list of Token objects."""
    current_pos = 0
    tokens = []
    while current_pos < len(source_text):
        match = TOKEN_RE.match(source_text, current_pos)
        if not match:
            raise ParseError(f"Unexpected character: '{source_text[current_pos]}'", current_pos)
        token_type = match.lastgroup
        token_value = match.group(token_type)
        if token_type in ('SKIP', 'NEWLINE'):
            current_pos = match.end()
            continue
        if token_type == 'NAME' and token_value.upper() in KEYWORDS:
            tokens.append(Token(token_value.upper(), token_value.upper(), current_pos))
        elif token_type == 'NAME':
            tokens.append(Token('NAME', token_value, current_pos))
        elif token_type == 'NUMBER':
            tokens.append(Token('NUMBER', token_value, current_pos))
        elif token_type == 'OP':
            tokens.append(Token('OP', token_value, current_pos))
        else:
            raise ParseError("Unknown token type", current_pos)
        current_pos = match.end()
    tokens.append(Token('EOF', '', current_pos))
    return tokens

# -------------------------
# Parser (Recursive Descent)
# -------------------------
# Grammar (informal):
# program       := statement*
# statement     := DEF NAME '(' params ')' '=' expression
#                | expression
# expression    := let_expr | if_expr | comparison
# let_expr      := LET NAME '=' expression IN expression
# if_expr       := IF expression THEN expression ELSE expression
# comparison    := addsub (('=='|'!='|'<'|'>'|'<='|'>=') addsub)*
# addsub        := muldiv (('+'|'-') muldiv)*
# muldiv        := pow (('*'|'/'|'%') pow)*
# pow           := unary ('^' pow)?   # right-associative
# unary         := ('+'|'-') unary | primary
# primary       := NUMBER | NAME | NAME '(' args ')' | '(' expression ')'
# params        := [NAME (',' NAME)*]
# args          := [expression (',' expression)*]

class Parser:
    def __init__(self, token_list):
        self.tokens = token_list
        self.position = 0

    def current(self):
        return self.tokens[self.position]

    def eat(self, expected_type=None, expected_value=None):
        current_token = self.current()
        if expected_type and current_token.type != expected_type:
            raise ParseError(f"Expected token type {expected_type}, got {current_token.type}", current_token.pos)
        if expected_value and current_token.value != expected_value:
            raise ParseError(f"Expected token {expected_value}, got {current_token.value}", current_token.pos)
        self.position += 1
        return current_token

    def parse(self):
        """Parses one expression or a DEF statement (top-level). Returns AST."""
        current_token = self.current()
        if current_token.type == 'DEF':
            ast = self.parse_def()
        else:
            ast = self.parse_expression()

        # Ensure entire input consumed
        if self.current().type != 'EOF':
            raise ParseError("Unexpected extra input after valid expression", self.current().pos)
        return ast

    def parse_def(self):
        # DEF NAME '(' params ')' '=' expression
        self.eat('DEF')
        name_token = self.eat('NAME')
        function_name = name_token.value
        self.eat('OP', '(')
        params = []
        if self.current().type == 'NAME':
            params.append(self.eat('NAME').value)
            while self.current().value == ',':
                self.eat('OP', ',')
                params.append(self.eat('NAME').value)
        self.eat('OP', ')')
        self.eat('OP', '=')
        body_expr = self.parse_expression()
        return ('DEF', function_name, params, body_expr)

    def parse_expression(self):
        current_token = self.current()
        if current_token.type == 'LET':
            return self.parse_let()
        if current_token.type == 'IF':
            return self.parse_if()
        return self.parse_comparison()

    def parse_let(self):
        # LET NAME = expression IN expression
        self.eat('LET')
        var_name = self.eat('NAME').value
        self.eat('OP', '=')
        value_expression = self.parse_expression()
        self.eat('IN')
        body_expression = self.parse_expression()
        return ('LET', var_name, value_expression, body_expression)

    def parse_if(self):
        # IF <expression> THEN <expression> ELSE <expression>
        # allow a full expression as condition (so nested IFs/LETs are valid)
        self.eat('IF')
        condition_expr = self.parse_expression()
        self.eat('THEN')
        true_expr = self.parse_expression()
        self.eat('ELSE')
        false_expr = self.parse_expression()
        return ('IF', condition_expr, true_expr, false_expr)

    def parse_comparison(self):
        left_expr = self.parse_addsub()
        while True:
            current_token = self.current()
            if current_token.type == 'OP' and current_token.value in ('==', '!=', '<', '>', '<=', '>='):
                operator = current_token.value
                self.eat('OP')
                right_expr = self.parse_addsub()
                left_expr = ('CMP', operator, left_expr, right_expr)
            else:
                break
        return left_expr

    def parse_addsub(self):
        node = self.parse_muldiv()
        while True:
            current_token = self.current()
            if current_token.type == 'OP' and current_token.value in ('+', '-'):
                operator = current_token.value
                self.eat('OP')
                right_node = self.parse_muldiv()
                node = ('BINOP', operator, node, right_node)
            else:
                break
        return node

    def parse_muldiv(self):
        node = self.parse_pow()
        while True:
            current_token = self.current()
            if current_token.type == 'OP' and current_token.value in ('*', '/', '%'):
                operator = current_token.value
                self.eat('OP')
                right_node = self.parse_pow()
                node = ('BINOP', operator, node, right_node)
            else:
                break
        return node

    def parse_pow(self):
        # right-associative exponentiation
        node = self.parse_unary()
        current_token = self.current()
        if current_token.type == 'OP' and current_token.value in ('^', '∧'):
            # normalize both ^ and unicode ∧ to '^' in AST
            self.eat('OP')
            right_node = self.parse_pow()
            node = ('BINOP', '^', node, right_node)
        return node

    def parse_unary(self):
        current_token = self.current()
        if current_token.type == 'OP' and current_token.value in ('+', '-'):
            operator = current_token.value
            self.eat('OP')
            operand_node = self.parse_unary()
            return ('UNARY', operator, operand_node)
        return self.parse_primary()

    def parse_primary(self):
        current_token = self.current()
        if current_token.type == 'NUMBER':
            self.eat('NUMBER')
            return ('NUMBER', current_token.value)
        if current_token.type == 'NAME':
            name_value = current_token.value
            self.eat('NAME')
            # function call?
            if self.current().value == '(':
                self.eat('OP', '(')
                args = []
                if self.current().value != ')':
                    args.append(self.parse_expression())
                    while self.current().value == ',':
                        self.eat('OP', ',')
                        args.append(self.parse_expression())
                self.eat('OP', ')')
                return ('CALL', name_value, args)
            else:
                return ('VAR', name_value)
        if current_token.value == '(':
            self.eat('OP', '(')
            inner = self.parse_expression()
            self.eat('OP', ')')
            return inner
        raise ParseError(f"Unexpected token {current_token.type}({current_token.value})", current_token.pos)

# -------------------------
# Evaluator
# -------------------------
class Function:
    def __init__(self, name, params, body):
        self.name = name
        self.params = list(params)
        self.body = body

class Evaluator:
    def __init__(self):
        self.functions = {}  # name -> Function
        # total nested call depth
        self.call_depth = 0

    # Public API
    def execute(self, source_text):
        """Parse and evaluate a top-level source string.
           If it's a DEF statement we store the function and return None.
           Otherwise return octal string result.
        """
        token_list = tokenize(source_text)
        parser = Parser(token_list)
        ast_tree = parser.parse()
        if ast_tree[0] == 'DEF':
            # store definition
            _, function_name, params, body = ast_tree
            assert isinstance(function_name, str)
            assert isinstance(params, list)
            self.functions[function_name] = Function(function_name, params, body)
            assert function_name in self.functions
            return None
        else:
            value = self._eval(ast_tree, env={})
            assert isinstance(value, int)
            return int_to_octal_str(value)

    def _eval(self, node, env):
        """Evaluate AST node with given variable environment (dict name->int)."""
        assert isinstance(env, dict)
        if not isinstance(node, tuple) or len(node) == 0:
            raise EvaluationError("Invalid AST node")
        nodetype = node[0]

        if nodetype == 'NUMBER':
            literal = node[1]
            return octal_str_to_int(literal)

        if nodetype == 'VAR':
            var_name = node[1]
            if var_name in env:
                return env[var_name]
            if var_name in self.functions:
                raise EvaluationError(f"Function '{var_name}' used without call")
            raise NameErrorEval(var_name)

        if nodetype == 'UNARY':
            operator = node[1]
            operand_node = node[2]
            operand_value = self._eval(operand_node, env)
            if operator == '+':
                return +operand_value
            else:
                return -operand_value

        if nodetype == 'BINOP':
            operator = node[1]
            left_value = self._eval(node[2], env)
            right_value = self._eval(node[3], env)
            assert isinstance(left_value, int) and isinstance(right_value, int)
            if operator == '+':
                return left_value + right_value
            if operator == '-':
                return left_value - right_value
            if operator == '*':
                return left_value * right_value
            if operator == '/':
                if right_value == 0:
                    raise DivisionByZeroError()
                # truncation toward zero
                return int(left_value / right_value)
            if operator == '%':
                if right_value == 0:
                    raise DivisionByZeroError()
                quotient = int(left_value / right_value)  # trunc toward zero
                remainder = left_value - right_value * quotient
                return remainder
            if operator == '^':
                if right_value < 0:
                    raise EvaluationError("Negative exponent results in non-integer")
                result_value = pow(left_value, right_value)
                assert isinstance(result_value, int)
                return result_value
            raise EvaluationError(f"Unknown binary operator '{operator}'")

        if nodetype == 'CMP':
            operator = node[1]
            left_value = self._eval(node[2], env)
            right_value = self._eval(node[3], env)
            if operator == '==':
                return 1 if left_value == right_value else 0
            if operator == '!=':
                return 1 if left_value != right_value else 0
            if operator == '<':
                return 1 if left_value < right_value else 0
            if operator == '>':
                return 1 if left_value > right_value else 0
            if operator == '<=':
                return 1 if left_value <= right_value else 0
            if operator == '>=':
                return 1 if left_value >= right_value else 0
            raise EvaluationError(f"Unknown comparison '{operator}'")

        if nodetype == 'LET':
            _, var_name, value_expr, body_expr = node
            value_evaluated = self._eval(value_expr, env)
            new_env = dict(env)
            new_env[var_name] = value_evaluated
            return self._eval(body_expr, new_env)

        if nodetype == 'IF':
            _, cond_expr, true_expr, false_expr = node
            cond_value = self._eval(cond_expr, env)
            if cond_value != 0:
                return self._eval(true_expr, env)
            else:
                return self._eval(false_expr, env)

        if nodetype == 'CALL':
            _, func_name, arg_nodes = node
            if func_name not in self.functions:
                raise NameErrorEval(func_name)
            function_obj = self.functions[func_name]
            if len(arg_nodes) != len(function_obj.params):
                raise ArityError(func_name, len(function_obj.params), len(arg_nodes))

            if self.call_depth >= RECURSION_LIMIT:
                raise RecursionLimitError(RECURSION_LIMIT)

            # Evaluate args in caller env
            arg_values = [self._eval(arg_node, env) for arg_node in arg_nodes]

            # prepare local env for function
            local_env = {p: v for p, v in zip(function_obj.params, arg_values)}

            self.call_depth += 1
            try:
                try:
                    result = self._eval(function_obj.body, local_env)
                except RecursionError:
                    raise RecursionLimitError(RECURSION_LIMIT)
                return result
            finally:
                self.call_depth -= 1

        raise EvaluationError(f"Unknown AST node type: {nodetype}")

# -------------------------
# Simple REPL helper (if needed)
# -------------------------
if __name__ == "__main__":
    print("Octal Calculator REPL. Type 'exit' to quit.")
    evaluator = Evaluator()
    while True:
        try:
            user_input = input('> ').strip()
            if not user_input:
                continue
            if user_input.lower() in ('exit', 'quit'):
                break
            output = evaluator.execute(user_input)
            if output is None:
                print("[DEF stored]")
            else:
                print(output)
        except Exception as e:
            print("Error:", e)
