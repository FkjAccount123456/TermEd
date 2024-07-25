from utils import get_width
from keyword import kwlist

op = [
    "+",
    "-",
    "*",
    "/",
    "%",
    "==",
    "!=",
    ">",
    "<",
    ">=",
    "<=",
    "<<",
    ">>",
    "&",
    "|",
    "^",
    "~",
    "=",
]

# builtins = dir(__builtins__).copy()


def pyrenderer(code: str, width: int, cs: dict):
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
            while p < len(code) and (code[p].isdigit() or code[p] in ".xbo"):
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
            # elif s in builtins or s in ('self', 'cls'):
            #     for i in range(len(s)):
            #         add_ret(cs["builtin"])
            else:
                for i in range(len(s)):
                    add_ret(cs["identifier"])
        elif code[p : p + 3] in ('"""', "'''"):
            s = code[p : p + 3]
            x = code[p : p + 3]
            t = p + 3
            while t < len(code) and code[t : t + 3] != x:
                if code[t] == "\\":
                    s += code[t]
                    t += 1
                    if code[t] in "'\"":
                        s += code[t]
                        t += 1
                else:
                    s += code[t]
                    t += 1
            s += code[t : t + 3]
            t += 3
            for i in range(len(s)):
                add_ret(cs["string"])
        elif code[p] in "'\"":
            s = code[p]
            x = code[p]
            t = p + 1
            while t < len(code) and code[t] != "\n" and code[t] != x:
                if code[t] == "\\":
                    s += code[t]
                    t += 1
                    if code[t] in "'\"\n":
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
        elif code[p : p + 2] in op:
            add_ret(cs["operator"])
            add_ret(cs["operator"])
        elif code[p] in op:
            add_ret(cs["operator"])
        else:
            add_ret(cs["others"])

    return ret
