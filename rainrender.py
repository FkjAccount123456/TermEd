from utils import get_width
from keyword import kwlist
from rain import TokenType

op = {
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
}.keys()

kwlist = {
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
}.keys()


def rainrenderer(code: str, width: int, cs: dict):
    ret = [[[]]]
    ret_cur_w = 0

    def add_ret(color: str):
        nonlocal ret_cur_w, p
        if p >= len(code):
            return
        ch = code[p]
        p += 1
        if ch == "\n":
            ret.append([[]])
            ret_cur_w = 0
        else:
            w = get_width(ch)
            if ret_cur_w + w > width:
                ret_cur_w = 0
                ret[-1].append([])
            ret_cur_w += w
            ret[-1][-1].append(color + ch)

    p = 0
    while p < len(code):
        if code[p].isdigit():
            while p < len(code) and (code[p].isdigit() or code[p] in "."):
                add_ret(cs["number"])
        elif code[p].isalpha() or code[p] == "_":
            s = ""
            t = p
            while t < len(code) and (code[t].isalnum() or code[t] == "_"):
                s += code[t]
                t += 1
            if s in kwlist:
                for i in range(len(s)):
                    add_ret(cs["keyword"])
            else:
                for i in range(len(s)):
                    add_ret(cs["identifier"])
        elif code[p] in "'\"":
            s = code[p]
            x = code[p]
            t = p + 1
            while t < len(code) and code[t] != x:
                if code[t] == "\\":
                    s += code[t]
                    t += 1
                    if code[t] in "'\"":
                        s += code[t]
                        t += 1
                else:
                    s += code[t]
                    t += 1
            if t < len(code):
                s += code[t]
                t += 1
            for i in range(len(s)):
                add_ret(cs["string"])
        elif code[p] == "#":
            while p < len(code) and code[p] != "\n":
                add_ret(cs["comment"])
        elif code[p : p + 3] in op:
            add_ret(cs["operator"])
            add_ret(cs["operator"])
            add_ret(cs["operator"])
        elif code[p : p + 2] in op:
            add_ret(cs["operator"])
            add_ret(cs["operator"])
        elif code[p] in op:
            add_ret(cs["operator"])
        else:
            add_ret(cs["others"])

    return ret
