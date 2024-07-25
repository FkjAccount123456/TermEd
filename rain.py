# import rain10 as nrn
from enum import Enum, unique
from typing import NamedTuple, Any
import sys
import copy
import time


@unique
class TokenType(Enum):
    EOF = 0

    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    MOD = 5
    EQ = 6
    NE = 7
    GT = 8
    GE = 9
    LT = 10
    LE = 11
    LSH = 12
    RSH = 13
    NOT = 16
    BITAND = 17
    BITOR = 18
    XOR = 19
    INV = 501

    LPAREN = 100
    RPAREN = 101
    LSQBR = 102
    RSQBR = 103
    COMMA = 104
    SEMI = 105
    COLON = 106
    DOT = 107
    LBRACE = 109
    RBRACE = 110

    ASSIGN = 408
    ADD_EQ = 409
    SUB_EQ = 410
    MUL_EQ = 411
    DIV_EQ = 412
    MOD_EQ = 413
    LSH_EQ = 414
    RSH_EQ = 415
    BITAND_EQ = 416
    BITOR_EQ = 417
    XOR_EQ = 418

    END = 201
    AND = 14
    OR = 15
    IF = 204
    ELSE = 205
    WHILE = 206
    FOR = 207
    TO = 208
    IN = 209
    RETURN = 210
    BREAK = 211
    CONTINUE = 212
    VAR = 213
    FUNC = 214
    TRUE = 215
    FALSE = 216
    NONE = 217
    ELIF = 219
    STEP = 220
    MODULE = 221
    REQUIRE = 222

    ID = 300
    CONST = 301
    EOL = 302


class Token(NamedTuple):
    type: TokenType
    val: Any = None


operators = {
    "+": TokenType.ADD,
    "-": TokenType.SUB,
    "*": TokenType.MUL,
    "/": TokenType.DIV,
    "%": TokenType.MOD,
    "==": TokenType.EQ,
    "!=": TokenType.NE,
    ">": TokenType.GT,
    ">=": TokenType.GE,
    "<": TokenType.LT,
    "<=": TokenType.LE,
    "<<": TokenType.LSH,
    ">>": TokenType.RSH,
    "&": TokenType.BITAND,
    "|": TokenType.BITOR,
    "^": TokenType.XOR,
    "~": TokenType.INV,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "[": TokenType.LSQBR,
    "]": TokenType.RSQBR,
    ",": TokenType.COMMA,
    ";": TokenType.SEMI,
    ":": TokenType.COLON,
    ".": TokenType.DOT,
    "=": TokenType.ASSIGN,
    "{": TokenType.LBRACE,
    "}": TokenType.RBRACE,
    "<<=": TokenType.LSH_EQ,
    ">>=": TokenType.RSH_EQ,
    "&=": TokenType.BITAND_EQ,
    "|=": TokenType.BITOR_EQ,
    "^=": TokenType.XOR_EQ,
    "+=": TokenType.ADD_EQ,
    "-=": TokenType.SUB_EQ,
    "*=": TokenType.MUL_EQ,
    "/=": TokenType.DIV_EQ,
    "%=": TokenType.MOD_EQ,
}

keywords = {
    "end": TokenType.END,
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "while": TokenType.WHILE,
    "for": TokenType.FOR,
    "to": TokenType.TO,
    "in": TokenType.IN,
    "return": TokenType.RETURN,
    "break": TokenType.BREAK,
    "continue": TokenType.CONTINUE,
    "var": TokenType.VAR,
    "func": TokenType.FUNC,
    "True": TokenType.TRUE,
    "False": TokenType.FALSE,
    "None": TokenType.NONE,
    "step": TokenType.STEP,
    "elif": TokenType.ELIF,
    "module": TokenType.MODULE,
    "require": TokenType.REQUIRE,
}

escapes = {
    "r": "\r",
    "t": "\t",
    "a": "\a",
    "f": "\f",
    "v": "\v",
    "b": "\b",
    "n": "\n",
    "\\": "\\",
    "'": "'",
    '"': '"',
}

priority = {
    TokenType.MUL: 100,
    TokenType.DIV: 100,
    TokenType.MOD: 100,
    TokenType.ADD: 99,
    TokenType.SUB: 99,
    TokenType.LSH: 98,
    TokenType.RSH: 98,
    TokenType.EQ: 97,
    TokenType.NE: 97,
    TokenType.GT: 96,
    TokenType.GE: 96,
    TokenType.LT: 96,
    TokenType.LE: 96,
    TokenType.BITAND: 95,
    TokenType.BITOR: 94,
    TokenType.XOR: 93,
    TokenType.AND: 92,
    TokenType.OR: 91,
}

assignop2op = {
    TokenType.ADD_EQ: TokenType.ADD,
    TokenType.SUB_EQ: TokenType.SUB,
    TokenType.MUL_EQ: TokenType.MUL,
    TokenType.DIV_EQ: TokenType.DIV,
    TokenType.MOD_EQ: TokenType.MOD,
    TokenType.LSH_EQ: TokenType.LSH,
    TokenType.RSH_EQ: TokenType.RSH,
    TokenType.BITAND_EQ: TokenType.BITAND,
    TokenType.BITOR_EQ: TokenType.BITOR,
    TokenType.XOR_EQ: TokenType.XOR,
}

binary_fntable = [
    None,
    lambda x, y: x + y,
    lambda x, y: x - y,
    lambda x, y: x * y,
    lambda x, y: x / y,
    lambda x, y: x % y,
    lambda x, y: x == y,
    lambda x, y: x != y,
    lambda x, y: x > y,
    lambda x, y: x >= y,
    lambda x, y: x < y,
    lambda x, y: x <= y,
    lambda x, y: x << y,
    lambda x, y: x >> y,
    lambda x, y: x and y,
    lambda x, y: x or y,
    None,
    lambda x, y: x & y,
    lambda x, y: x | y,
    lambda x, y: x ^ y,
]


class RainError(Exception):
    pass


class RainSyntaxError(RainError):
    pass


class RainNameError(RainError):
    pass


def tokenize(code: str) -> list[Token]:
    res = []
    pos = 0
    pair_begins = []

    while pos < len(code):
        while pos < len(code) and (code[pos] in " \t" or code[pos] == "#"):
            if code[pos] == "#":
                while pos < len(code) and code[pos] != "\n":
                    pos += 1
            else:
                pos += 1

        if pos >= len(code):
            res.append(Token(TokenType.EOL))
            break
        elif code[pos] in "\r\n":
            if not pair_begins or pair_begins[-1] not in (
                TokenType.LBRACE,
                TokenType.LPAREN,
                TokenType.LSQBR,
            ):
                res.append(Token(TokenType.EOL))
            pos += 1
        elif code[pos].isdigit():
            num = ""
            while pos < len(code) and (code[pos].isdigit() or code[pos] == "."):
                num += code[pos]
                pos += 1
            if num.count(".") > 1:
                raise RainSyntaxError(f"Invalid number: {num}")
            elif "." not in num:
                res.append(Token(TokenType.CONST, int(num)))
            else:
                res.append(Token(TokenType.CONST, float(num)))
        elif code[pos].isalpha() or code[pos] == "_":
            ident = ""
            while pos < len(code) and (code[pos].isalnum() or code[pos] == "_"):
                ident += code[pos]
                pos += 1
            if ident in keywords:
                res.append(Token(keywords[ident]))
                if res[-1].type in (
                    TokenType.IF,
                    TokenType.WHILE,
                    TokenType.FOR,
                    TokenType.FUNC,
                ):
                    pair_begins.append(res[-1].type)
                elif res[-1].type == TokenType.END:
                    if pair_begins and pair_begins[-1] in (
                        TokenType.IF,
                        TokenType.WHILE,
                        TokenType.FOR,
                        TokenType.FUNC,
                    ):
                        pair_begins.pop()
                    else:
                        raise RainSyntaxError(f"Mismatched TokenType.END")
            else:
                res.append(Token(TokenType.ID, ident))
        elif code[pos] in "'\"":
            quote = code[pos]
            pos += 1
            string = ""
            while pos < len(code) and code[pos] != quote:
                if code[pos] == "\\":
                    pos += 1
                    if pos >= len(code):
                        raise RainSyntaxError(f"Unexpected EOF")
                    elif code[pos] in escapes:
                        string += escapes[code[pos]]
                        pos += 1
                    elif code[pos] == "x":
                        pos += 1
                        if pos + 2 >= len(code):
                            raise RainSyntaxError(f"Unexpected EOF")
                        string += chr(int(code[pos : pos + 2], 16))
                        pos += 2
                    elif code[pos] == "u":
                        pos += 1
                        if pos + 4 >= len(code):
                            raise RainSyntaxError(f"Unexpected EOF")
                        string += chr(int(code[pos : pos + 4], 16))
                        pos += 4
                    else:
                        raise RainSyntaxError(f"Invalid escape sequence: \\{code[pos]}")
                else:
                    string += code[pos]
                    pos += 1
            if pos >= len(code):
                raise RainSyntaxError(f"Unexpected EOF")
            pos += 1
            res.append(Token(TokenType.CONST, string))
        elif code[pos : pos + 3] in operators:
            res.append(Token(operators[code[pos : pos + 3]]))
            pos += 3
        elif code[pos : pos + 2] in operators:
            res.append(Token(operators[code[pos : pos + 2]]))
            pos += 2
        elif code[pos] in operators:
            if code[pos] == "(":
                pair_begins.append(TokenType.LPAREN)
            elif code[pos] == ")":
                if pair_begins and pair_begins[-1] == TokenType.LPAREN:
                    pair_begins.pop()
                else:
                    raise RainSyntaxError(f"Mismatched parentheses")
            elif code[pos] == "[":
                pair_begins.append(TokenType.LSQBR)
            elif code[pos] == "]":
                if pair_begins and pair_begins[-1] == TokenType.LSQBR:
                    pair_begins.pop()
                else:
                    raise RainSyntaxError(f"Mismatched square brackets")
            elif code[pos] == "{":
                pair_begins.append(TokenType.LBRACE)
            elif code[pos] == "}":
                if pair_begins and pair_begins[-1] == TokenType.LBRACE:
                    pair_begins.pop()
                else:
                    raise RainSyntaxError(f"Mismatched braces")
            res.append(Token(operators[code[pos]]))
            pos += 1
        else:
            raise RainSyntaxError(f"Unexpected character: {code[pos]}")

    res.append(Token(TokenType.EOF))
    return res


def parse(tokens: list[Token]):
    token = tokens[0]
    pos = 0

    def eat(tp: TokenType | None = None):
        nonlocal token, pos
        if tp is not None and token.type != tp:
            raise RainSyntaxError(f"Expected {tp}, got {token}")
        old = token
        pos += 1
        token = tokens[pos]
        return old

    def parse_factor():
        if token.type == TokenType.CONST:
            res = "const", eat().val
        elif token.type == TokenType.ID:
            res = "var", eat().val
        elif token.type == TokenType.LPAREN:
            eat()
            res = parse_expr()
            eat(TokenType.RPAREN)
        elif token.type == TokenType.TRUE:
            res = "const", True
            eat()
        elif token.type == TokenType.FALSE:
            res = "const", False
            eat()
        elif token.type == TokenType.NONE:
            res = "const", None
            eat()
        elif token.type == TokenType.LSQBR:
            eat()
            l = []
            if token.type != TokenType.RSQBR:
                l.append(parse_expr())
                while token.type == TokenType.COMMA:
                    eat()
                    l.append(parse_expr())
            eat(TokenType.RSQBR)
            res = "list", l
        elif token.type == TokenType.LBRACE:
            eat()
            kv_pairs = []
            if token.type != TokenType.RBRACE:
                k = eat(TokenType.ID).val
                eat(TokenType.COLON)
                v = parse_expr()
                kv_pairs.append((k, v))
                while token.type == TokenType.COMMA:
                    eat()
                    if token.type == TokenType.RBRACE:
                        break
                    k = eat(TokenType.ID).val
                    eat(TokenType.COLON)
                    v = parse_expr()
                    kv_pairs.append((k, v))
            eat(TokenType.RBRACE)
            res = "dict", kv_pairs
        elif token.type == TokenType.MODULE:
            eat()
            eat(TokenType.LBRACE)
            kv_pairs = []
            if token.type != TokenType.RBRACE:
                k = eat(TokenType.ID).val
                eat(TokenType.COLON)
                v = parse_expr()
                kv_pairs.append((k, v))
                while token.type == TokenType.COMMA:
                    eat()
                    if token.type == TokenType.RBRACE:
                        break
                    k = eat(TokenType.ID).val
                    eat(TokenType.COLON)
                    v = parse_expr()
                    kv_pairs.append((k, v))
            eat(TokenType.RBRACE)
            res = "module", kv_pairs
        elif token.type == TokenType.REQUIRE:
            eat()
            path = eat(TokenType.CONST).val
            if not isinstance(path, str):
                raise RainSyntaxError(f"Invalid require path: {path}")
            return "require", path
        elif token.type in (TokenType.ADD, TokenType.SUB, TokenType.NOT, TokenType.INV):
            op = eat().type
            res = "unary", op, parse_factor()
        elif token.type == TokenType.FUNC:
            eat()
            args = []
            eat(TokenType.LPAREN)
            if token.type != TokenType.RPAREN:
                args.append(eat(TokenType.ID).val)
                while token.type == TokenType.COMMA:
                    eat()
                    args.append(eat(TokenType.ID).val)
            eat(TokenType.RPAREN)
            eat(TokenType.EOL)
            body = parse_block({TokenType.END})
            eat()
            return "func", args, body
        else:
            raise RainSyntaxError(f"Unexpected token: {token.type}")

        while token.type in (TokenType.DOT, TokenType.LSQBR, TokenType.LPAREN):
            if token.type == TokenType.DOT:
                eat()
                attr = eat(TokenType.ID).val
                res = "attr", res, attr
            elif token.type == TokenType.LSQBR:
                eat()
                index = parse_expr()
                eat(TokenType.RSQBR)
                res = "index", res, index
            elif token.type == TokenType.LPAREN:
                args = []
                eat()
                if token.type != TokenType.RPAREN:
                    args.append(parse_expr())
                    while token.type == TokenType.COMMA:
                        eat()
                        args.append(parse_expr())
                eat(TokenType.RPAREN)
                res = "call", res, args

        return res

    def parse_expr():
        res = [parse_factor()]
        stack = []
        while token.type in priority:
            op = eat().type
            while stack and priority[op] <= priority[stack[-1]]:
                res.append(stack.pop())
            stack.append(op)
            res.append(parse_factor())
        while stack:
            res.append(stack.pop())
        if len(res) == 1:
            return res[0]
        return "expr", res

    def parse_stmt():
        if token.type == TokenType.EOL:
            return ("nop",)
        elif token.type == TokenType.VAR:
            eat()
            varlist = []
            name = eat(TokenType.ID).val
            if token.type == TokenType.ASSIGN:
                eat()
                val = parse_expr()
            else:
                val = "const", None
            varlist.append((name, val))
            while token.type == TokenType.COMMA:
                eat()
                name = eat(TokenType.ID).val
                if token.type == TokenType.ASSIGN:
                    eat()
                    val = parse_expr()
                else:
                    val = "const", None
                varlist.append((name, val))
            return "var", varlist
        elif token.type == TokenType.FUNC and tokens[pos + 1].type == TokenType.ID:
            eat()
            name = eat(TokenType.ID).val
            args = []
            eat(TokenType.LPAREN)
            if token.type != TokenType.RPAREN:
                args.append(eat(TokenType.ID).val)
                while token.type == TokenType.COMMA:
                    eat()
                    args.append(eat(TokenType.ID).val)
            eat(TokenType.RPAREN)
            eat(TokenType.EOL)
            body = parse_block({TokenType.END})
            eat()
            return "func", name, args, body
        elif token.type == TokenType.IF:
            eat()
            cases = []
            cond = parse_expr()
            eat(TokenType.EOL)
            body = parse_block({TokenType.ELIF, TokenType.ELSE, TokenType.END})
            cases.append((cond, body))
            while token.type == TokenType.ELIF:
                eat()
                cond = parse_expr()
                eat(TokenType.EOL)
                body = parse_block({TokenType.ELIF, TokenType.ELSE, TokenType.END})
                cases.append((cond, body))
            if token.type == TokenType.ELSE:
                eat()
                eat(TokenType.EOL)
                else_case = parse_block({TokenType.END})
            else:
                else_case = []
            eat()
            return "if", cases, else_case
        elif token.type == TokenType.WHILE:
            eat()
            cond = parse_expr()
            eat(TokenType.EOL)
            body = parse_block({TokenType.END})
            eat()
            return "while", cond, body
        elif token.type == TokenType.RETURN:
            eat()
            val = parse_expr()
            return "return", val
        elif token.type == TokenType.BREAK:
            eat()
            return ("break",)
        elif token.type == TokenType.CONTINUE:
            eat()
            return ("continue",)
        elif token.type == TokenType.FOR:
            eat()
            iname = eat(TokenType.ID).val
            if token.type == TokenType.ASSIGN:
                eat()
                begin = parse_expr()
                eat(TokenType.TO)
                end = parse_expr()
                if token.type == TokenType.STEP:
                    eat()
                    step = parse_expr()
                else:
                    step = "const", 1
                eat(TokenType.EOL)
                body = parse_block({TokenType.END})
                eat()
                return "forto", iname, begin, end, step, body
            else:
                raise RainSyntaxError(f"Unexpected token: {token.type}")
        else:
            left = parse_expr()
            if token.type.value >= TokenType.ASSIGN.value:
                op = eat().type
                right = parse_expr()
                return "assign", op, left, right
            else:
                return "exprstmt", left

    def parse_block(end_tokens: set[TokenType]):
        stmts = []
        while token.type != TokenType.EOF and token.type not in end_tokens:
            stmts.append(parse_stmt())
            eat(TokenType.EOL)
        if token.type == TokenType.EOF:
            raise RainSyntaxError("Unexpected EOF")
        return stmts

    prog = []
    while token.type != TokenType.EOF:
        prog.append(parse_stmt())
        eat(TokenType.EOL)
    return prog


def compile(prog):
    output = []
    scope = [{}]
    while_jmpends = []
    while_beginposs = []

    def hoist(fn_block):
        vars = []
        for i in fn_block:
            if i[0] == "var":
                vars.extend(map(lambda x: x[0], i[1]))
            elif i[0] == "func":
                vars.append(i[1])
            elif i[0] == "if":
                for j in i[1]:
                    vars.extend(hoist(j[1]))
                if i[2]:
                    vars.extend(hoist(i[2]))
            elif i[0] == "while":
                vars.extend(hoist(i[2]))
            elif i[0] == "forto":
                vars.append(i[1])
                vars.extend(hoist(i[5]))
        return vars

    def find_var(name: str):
        for n, i in enumerate(reversed(scope)):
            if name in i:
                return n + 1, i[name]
        raise RainNameError(f"Undefined variable: {name}")

    def compile_expr(expr):
        match expr:
            case "const", val:
                output.append(("push", val))
            case "var", name:
                output.append(("load", *find_var(name)))
            case "list", l:
                for i in l:
                    compile_expr(i)
                output.append(("list", len(l)))
            case "dict", l:
                ks = []
                for k, v in l:
                    ks.append(k)
                    compile_expr(v)
                output.append(("dict", ks))
            case "module", l:
                ks = []
                for k, v in l:
                    ks.append(k)
                    compile_expr(v)
                output.append(("module", ks))
            case "unary", op, arg:
                compile_expr(arg)
                output.append(("unary", op))
            case "expr", l:
                for i in l:
                    if isinstance(i, TokenType):
                        output.append(("binary", i))
                    else:
                        compile_expr(i)
            case "attr", obj, attr:
                compile_expr(obj)
                output.append(("attr", attr))
            case "index", obj, index:
                compile_expr(obj)
                compile_expr(index)
                output.append(("index",))
            case "call", fn, args:
                for i in args:
                    compile_expr(i)
                compile_expr(fn)
                output.append(("call", len(args)))
            case "func", args, body:
                fn_vars = args + hoist(body)
                output.append(
                    ("func", len(output) + 1, "<lambda>", len(args), len(fn_vars))
                )
                output.append(None)
                pos = len(output) - 1
                scope.append({})
                for i, arg in enumerate(fn_vars):
                    scope[-1][arg] = i
                compile_block(body)
                output.append(("push", None))
                output.append(("ret",))
                output[pos] = "jmp", len(output) - 1
                scope.pop()
            case "require", path:
                # 打包成函数
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        code = f.read()
                except FileNotFoundError:
                    raise RainNameError(f"Module not found: {path}")
                tokens = tokenize(code)
                prog = parse(tokens)
                vars = hoist(prog)
                output.append(
                    ("func", len(output) + 1, f"<module {path}>", 0, len(vars))
                )
                output.append(None)
                pos = len(output) - 1
                scope.append({})
                for i, arg in enumerate(vars):
                    scope[-1][arg] = i
                compile_block(prog)
                output.append(("push", None))
                output.append(("ret",))
                output[pos] = "jmp", len(output) - 1
                scope.pop()
                output.append(("call", 0))
            case _:
                raise RainError(f"Invalid expression: {expr}")

    def compile_stmt(stmt):
        match stmt:
            case "nop",:
                return
            case "var", varlist:
                for name, val in varlist:
                    compile_expr(val)
                    output.append(("store", *find_var(name)))
            case "func", name, args, body:
                fn_vars = args + hoist(body)
                output.append(("func", len(output) + 2, name, len(args), len(fn_vars)))
                output.append(("store", *find_var(name)))
                output.append(None)
                pos = len(output) - 1
                scope.append({})
                for i, arg in enumerate(fn_vars):
                    scope[-1][arg] = i
                compile_block(body)
                output.append(("push", None))
                output.append(("ret",))
                output[pos] = "jmp", len(output) - 1
                scope.pop()
            case "if", cases, else_case:
                # 懒得自己再推一遍了，参考betterlang吧
                jmps = []
                for cond, body in cases:
                    compile_expr(cond)
                    jnz = len(output)
                    output.append(None)
                    compile_block(body)
                    jmps.append(len(output))
                    output[jnz] = "jnz", len(output)
                    output.append(None)
                if else_case:
                    compile_block(else_case)
                for i in jmps:
                    output[i] = "jmp", len(output) - 1
            case "while", cond, body:
                while_jmpends.append([])
                while_beginposs.append(len(output) - 1)
                compile_expr(cond)
                jnz = len(output)
                output.append(None)
                compile_block(body)
                output.append(("jmp", while_beginposs[-1]))
                jmpends = while_jmpends.pop()
                output[jnz] = "jnz", len(output) - 1
                for i in jmpends:
                    output[i] = "jmp", len(output) - 1
            case "forto", iname, begin, end, step, body:
                # 为什么感觉比betterlang还复杂
                # 为什么感觉像Fortran
                varpos = find_var(iname)
                compile_expr(begin)
                output.append(("store", *varpos))
                while_jmpends.append([])
                while_beginposs.append(len(output) - 1)
                output.append(("load", *varpos))
                compile_expr(end)
                output.append(("binary", TokenType.LT))
                jnz = len(output)
                output.append(None)
                compile_block(body)
                compile_expr(step)
                output.append(("load", *varpos))
                output.append(("binary", TokenType.ADD))
                output.append(("store", *varpos))
                output.append(("jmp", while_beginposs[-1]))
                jmpends = while_jmpends.pop()
                output[jnz] = "jnz", len(output) - 1
                for i in jmpends:
                    output[i] = "jmp", len(output) - 1
            case "return", val:
                compile_expr(val)
                output.append(("ret",))
            case "break",:
                output.append(None)
                while_jmpends[-1].append(len(output) - 1)
            case "continue",:
                output.append(("jmp", while_beginposs[-1]))
            case "exprstmt", expr:
                compile_expr(expr)
                output.append(("pop",))
            case "assign", op, left, right:
                if op == TokenType.ASSIGN:
                    compile_expr(right)
                else:
                    compile_expr(left)
                    compile_expr(right)
                    output.append(("binary", assignop2op[op]))
                compile_expr(left)
                if output[-1][0] not in ("load", "attr", "index"):
                    raise RainError(f"Invalid l-value: {left}")
                if output[-1][0] == "load":
                    output[-1] = "store", *output[-1][1:]
                elif output[-1][0] == "attr":
                    output[-1] = "setattr", *output[-1][1:]
                elif output[-1][0] == "index":
                    output[-1] = "setindex", *output[-1][1:]
            case _:
                raise RainError(f"Invalid statement: {stmt}")

    def compile_block(block):
        for stmt in block:
            compile_stmt(stmt)

    globals = builtins[0] + hoist(prog)
    for i, n in enumerate(globals):
        scope[-1][n] = i
    compile_block(prog)
    bytecode_translate(output)
    output.append((0,))
    return output, len(globals)


builtins = [
    "print",
    "println",
    "readln",
    "ord",
    "chr",
    "len",
    "getchar",
    "flush",
    "time",
], [
    lambda *a: print(*a, sep="", end=""),
    lambda *a: print(*a, sep=""),
    input,
    ord,
    chr,
    len,
    lambda: sys.stdin.read(1),
    lambda: print(end="", flush=True),
    time.time,
]

bytecodes2i = {
    "exit": 0,
    "binary": 1,
    "push": 2,
    "pop": 3,
    "load": 4,
    "store": 5,
    "attr": 6,
    "index": 7,
    "call": 8,
    "ret": 9,
    "jmp": 10,
    "jnz": 11,
    "jz": 12,
    "unary": 13,
    "setattr": 14,
    "setindex": 15,
    "list": 16,
    "dict": 17,
    "func": 18,
    "module": 19,
}

i2bytecodes = dict(zip(bytecodes2i.values(), bytecodes2i.keys()))


def bytecode_translate(code):
    for i in range(len(code)):
        code[i] = bytecodes2i[code[i][0]], *code[i][1:]


class Func(NamedTuple):
    argcnt: int
    name: str
    pc: int
    closure: list[list]
    reserve_cnt: int


class Method(NamedTuple):
    obj: Any
    func: Any


class Module:
    def __init__(self, d: dict):
        self.d = d

    def __str__(self):
        return f"Module"

    def __repr__(self):
        return f"Module"

    def __getitem__(self, key):
        return self.d[key]

    def __setitem__(self, key, val):
        self.d[key] = val


def run(prog: list[tuple], reserve_cnt: int, pc: int = 0, scope: list[list] = None):
    stack = []
    pc_stack = []
    scope_stack = []
    if scope is None:
        scope = [builtins[1]]
    for i in range(reserve_cnt - len(scope[-1])):
        scope[-1].append(None)
    while pc < len(prog) and prog[pc] != (0,):
        # print(prog[pc])
        match prog[pc]:
            case 1, op:  # binary
                b = stack.pop()
                stack[-1] = binary_fntable[op.value](stack[-1], b)
            case 10, p:  # jmp
                pc = p
            case 11, p:  # jnz
                if not stack.pop():
                    pc = p
            case 2, val:  # push
                stack.append(val)
            case 3,:  # pop
                stack.pop()
            case 4, n, i:  # load
                stack.append(scope[-n][i])
            case 5, n, i:  # store
                scope[-n][i] = stack.pop()
            case 8, n:  # call
                func = stack.pop()
                args = []
                # print(func)
                if isinstance(func, Method):
                    args = [func.obj] + args
                    func = func.func
                else:
                    args = []
                if n:
                    args += stack[-n:]
                    del stack[-n:]
                else:
                    pass
                # print(func, "args:", args)
                if isinstance(func, Func):
                    if len(args) != func.argcnt:
                        # print(func)
                        raise RainError(
                            f"Invalid argument count for function {func.name}"
                        )
                    scope_stack.append(scope)
                    pc_stack.append(pc)
                    pc = func.pc
                    scope = func.closure.copy()
                    scope.append(args)
                    for i in range(func.reserve_cnt - len(args)):
                        scope[-1].append(None)
                elif callable(func):
                    # print(f"call {args}")
                    stack.append(func(*args))
                else:
                    raise RainError(f"Invalid function: {func}")
            case 9,:  # ret
                if not scope_stack:
                    return
                # scope.pop()
                scope = scope_stack.pop()
                pc = pc_stack.pop()
            case 14, attr:  # setattr
                base, val = stack.pop(), stack.pop()
                base[attr] = val
            case 15,:  # setindex
                index, base, val = stack.pop(), stack.pop(), stack.pop()
                base[index] = val
            case 13, op:  # unary
                if op == TokenType.ADD:
                    stack[-1] = +stack[-1]
                elif op == TokenType.SUB:
                    stack[-1] = -stack[-1]
                elif op == TokenType.NOT:
                    stack[-1] = not stack[-1]
                elif op == TokenType.INV:
                    stack[-1] = ~stack[-1]
            case 6, attr:  # attr
                base = stack.pop()
                res = base[attr]
                # print(base, res)
                if isinstance(base, dict) and (isinstance(res, Func) or callable(res)):
                    stack.append(Method(base, res))
                else:
                    stack.append(res)
            case 7,:  # index
                index = stack.pop()
                stack[-1] = stack[-1][index]
            case 12, p:  # jz
                if stack.pop():
                    pc = p
            case 16, n:  # list
                x = stack[-n:]
                del stack[-n:]
                stack.append(x)
            case 17, ks:  # dict
                x = stack[-len(ks) :]
                del stack[-len(ks) :]
                stack.append(dict(zip(ks, x)))
            case 18, p, name, argcnt, reserve_cnt:  # func
                func = Func(argcnt, name, p, scope.copy(), reserve_cnt)
                stack.append(func)
            case 19, ks:  # module
                x = stack[-len(ks) :]
                del stack[-len(ks) :]
                stack.append(Module(dict(zip(ks, x))))
            case _:
                raise RainError(f"Invalid operation: {prog[pc]}")
        pc += 1


def print_bytecode(code):
    for i, op in enumerate(code):
        print(i, i2bytecodes[op[0]], *map(repr, op[1:]))


def get_time(c):
    start = time.time()
    c()
    print("Time:", time.time() - start)


def py_test():
    def fib(n):
        if n == 0:
            return 0
        elif n == 1:
            return 1
        else:
            return fib(n - 1) + fib(n - 2)

    print(fib(30))


def py_test2():
    res = 1
    for i in range(1, 100001):
        res *= i
    # print(res)


def add_builtin(name: str, val):
    builtins[0].append(name)
    builtins[1].append(val)


def run_file(file: str):
    with open(file, "r", encoding="utf-8") as f:
        code = f.read()
    tokens = tokenize(code)
    prog = parse(tokens)
    bytecode = compile(prog)
    run(*bytecode)
    return bytecode[0]


def call_Func(func: Func, code: list, args: list):
    new_scope = func.closure + [args]
    # print(new_scope)
    # print_bytecode(code)
    run(code, func.reserve_cnt, func.pc + 1, new_scope)


if __name__ == "__main__":
    code = """
    var mod = require "mod.rain"
    println(mod.pi)
    println(mod.getPi())
    println(mod.isPrime(17))
    println(mod.isPrime(10))
    """
    tokens = tokenize(code)
    # print(*tokens, sep = '\n')
    prog = parse(tokens)
    bytecode = compile(prog)
    # print(prog)
    # print_bytecode(code[0])
    # get_time(lambda: py_test2())
    run(*bytecode)
"""nrn.run_code(nrn.parse(nrn.lex('''
var t=time.time()
var num=100000
var res=res where(
    var res=1
    for i=1 to num+1
        res=res*i
    end
)
#println(res)
println("Time: ",time.time()-t)
''')), nrn.scope)"""

"""
func hello(name)
    println("Hello, ", name, "!")
end

hello("world")

var res = 1, i = 1
while i <= 10
    res = res * i
    i += 1
end
print(res, "\\n")

func fib(n)
    if n == 0
        return 0
    elif n == 1
        return 1
    else
        return fib(n-1) + fib(n-2)
    end
end

println(fib(10))

var res = 1
for i = 1 to 11
    res = res * i
end
print(res, "\\n")

var i = 0
while 1
    i += 1
    if i == 10
        break
    end
    if i == 5
        continue
    end
    println(i)
end

println({
    a: 1,
    b: 2,
})

var obj = {
    a: 1,
    setA: func(self, a)
        self.a = a
    end,
    print: func(self)
        println(self.a)
    end,
}

obj.print()
obj.setA(2)
obj.print()

var l = [1, 2, [2, 4, 5]]
println(l)
println(l[2][0])
l[2][0] = 3
println(l)
println(l[2][0])

var t = time()
var res = 1
for i = 1 to 100001
    res = res * i
end
# print(res, "\\n")
println("Time: ", time() - t)

func Counter(start)
    return func()
        start += 1
        return start
    end
end

var c = Counter(0)
println(c())
println(c())
println(c())

var mod = module {
    pi: 3.14,
    getPi: func()
        return mod.pi
    end,
}

println(mod.pi)
println(mod.getPi())
"""
