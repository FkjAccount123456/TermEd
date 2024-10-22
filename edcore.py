from msvcrt import getwch as getch, kbhit
import copy
import pyperclip
import os
import sys
from utils import *
from pyrender import pyrenderer
from rainrender import rainrenderer
from colorschemes import colorschemes
from rain import run_file, add_builtin, Module, Func, call_Func
from edbase import TextSpace
from lexers import pylexer, unlexer

lexers = {
    '.py': (pylexer.PyLexer, pylexer.py_tok2hl),
}


class Screen:
    def __init__(self, h: int, w: int):
        self.h, self.w = h, w
        self.data = [[" " for i in range(w)] for j in range(h)]
        self.color = [["" for i in range(w)] for j in range(h)]
        self.changed: set[Pos] = set()

        gotoxy(1, 1)
        for i in range(h):
            print(" " * self.w)

    def change(self, y: int, x: int, ch: str, color: str):
        if y < 0 or x < 0 or y >= self.h or x >= self.w:
            return
        if self.data[y][x] != ch or self.color[y][x] != color:
            self.changed.add((y, x))
            self.data[y][x] = ch
            self.color[y][x] = color

    def refresh(self):
        print("\033[0m", end="")
        gotoxy(1, 1)
        last = ""
        lastpos = 0, -1
        for y, x in sorted(self.changed):
            # print(y, x, end=' ')
            if y != lastpos[0] or x != lastpos[1] + 1:
                gotoxy(y + 1, x + 1)
            if last == self.color[y][x]:
                print(self.color[y][x] + self.data[y][x], end="")
            else:
                print("\033[0m" + self.color[y][x] + self.data[y][x], end="")
            last = self.color[y][x]
            lastpos = y, x
        self.changed = set()


class Editor(TextSpace):
    def __init__(self, h: int, w: int, file: str | None = None):
        super().__init__(w - 1)
        self.screen = Screen(h, w)
        self.selecty, self.selectx = [0, 0], 0
        self.screen_h = h
        self.textspace_h = self.screen_h - 2
        self.screen_w = w - 1
        self.linum_w = 0

        self.scroll_begin = [0, 0]

        self.exit = False
        self.mode = "NORMAL"

        self.cmd_input = ""
        self.cmd_x = 0

        self.file = file

        self.show_linum = True

        if self.show_linum:
            self.textspace_w -= 6
            self.linum_w = 6

        self.colorscheme = colorschemes["default"]

        self.undo_history = []  # 闭区间
        self.undo_version = 0
        self.saved_version = 0

        self.search_pattern = ""

        self.lexer_cls, self.tok2hl = unlexer.UnLexer, unlexer.un_tok2hl
        self.lexer = self.lexer_cls(self.text)

        self.keymaps = {
            "NORMAL": {
                "i": lambda: self.__setattr__("mode", "INSERT"),
                "v": self.setmode_select,
                ":": self.setmode_command,
                # 'q': lambda: self.__setattr__("exit", True),
                "h": lambda: self.move_cursor("left"),
                "j": lambda: self.move_cursor("down"),
                "k": lambda: self.move_cursor("up"),
                "l": lambda: self.move_cursor("right"),
                "0": lambda: self.move_cursor("home"),
                "^": lambda: self.move_cursor("linebegin"),
                "$": lambda: self.move_cursor("end"),
                "p": lambda: self.insert_any(pyperclip.paste()),
                "g": {
                    "g": lambda: self.move_cursor("start"),
                },
                "G": lambda: self.move_cursor("final"),
                "u": self.undo,
                "n": lambda: self.search_next(self.search_pattern),
                "N": lambda: self.search_prev(self.search_pattern),
                "f": lambda: self.find_char_next(getch()),
                "F": lambda: self.find_char_prev(getch()),
                "w": self.goto_next_word,
                "W": self.goto_next_word,
                "b": self.goto_prev_word,
                "B": self.goto_prev_word,
                "\x12": self.redo,
                "\xe0": {
                    "K": lambda: self.move_cursor("left"),
                    "P": lambda: self.move_cursor("down"),
                    "H": lambda: self.move_cursor("up"),
                    "M": lambda: self.move_cursor("right"),
                    "G": lambda: self.move_cursor("home"),
                    "O": lambda: self.move_cursor("end"),
                    "I": lambda: self.move_cursor("pageup"),
                    "Q": lambda: self.move_cursor("pagedown"),
                },
            },
            "INSERT": {
                "\x1b": lambda: self.__setattr__("mode", "NORMAL"),
                "\x08": self.del_before_cursor,
                "\t": lambda: self.insert([" ", " ", " ", " "]),
                "\xe0": {
                    "K": lambda: self.move_cursor("left"),
                    "P": lambda: self.move_cursor("down"),
                    "H": lambda: self.move_cursor("up"),
                    "M": lambda: self.move_cursor("right"),
                    "G": lambda: self.move_cursor("home"),
                    "O": lambda: self.move_cursor("end"),
                    "I": lambda: self.move_cursor("pageup"),
                    "Q": lambda: self.move_cursor("pagedown"),
                },
            },
            "SELECT": {
                "\x1b": lambda: self.__setattr__("mode", "NORMAL"),
                "h": lambda: self.move_cursor("left"),
                "j": lambda: self.move_cursor("down"),
                "k": lambda: self.move_cursor("up"),
                "l": lambda: self.move_cursor("right"),
                "0": lambda: self.move_cursor("home"),
                "$": lambda: self.move_cursor("end"),
                "d": self.del_selected,
                "y": lambda: pyperclip.copy(self.get_selected()),
                "g": {
                    "g": lambda: self.move_cursor("start"),
                },
                "G": lambda: self.move_cursor("final"),
                ":": self.setmode_command,
                "n": lambda: self.search_next(self.search_pattern),
                "N": lambda: self.search_prev(self.search_pattern),
                "f": lambda: self.find_char_next(getch()),
                "F": lambda: self.find_char_prev(getch()),
                "\xe0": {
                    "K": lambda: self.move_cursor("left"),
                    "P": lambda: self.move_cursor("down"),
                    "H": lambda: self.move_cursor("up"),
                    "M": lambda: self.move_cursor("right"),
                    "G": lambda: self.move_cursor("home"),
                    "O": lambda: self.move_cursor("end"),
                    "I": lambda: self.move_cursor("pageup"),
                    "Q": lambda: self.move_cursor("pagedown"),
                },
            },
            "COMMAND": {
                "\x1b": self.esc_cmdmode,
                "\x08": self.cmd_del_before_cursor,
                "\x19": lambda: self.cmd_paste(pyperclip.paste()),
                "\n": self.accept_cmd,
                "\r": self.accept_cmd,
                "\xe0": {
                    "K": lambda: self.cmd_move_cursor("left"),
                    "P": lambda: self.cmd_move_cursor("down"),
                    "H": lambda: self.cmd_move_cursor("up"),
                    "M": lambda: self.cmd_move_cursor("right"),
                    "G": lambda: self.cmd_move_cursor("home"),
                    "O": lambda: self.cmd_move_cursor("end"),
                    "I": lambda: self.cmd_move_cursor("pageup"),
                    "Q": lambda: self.cmd_move_cursor("pagedown"),
                },
            },
        }

        self.rain_add_builtins()
        self.config_code = self.load_config()

        self.open_file()

    def rain_add_builtins(self):
        mod = {}
        for i in dir(self):
            mod[i] = getattr(self, i)
        add_builtin("editor", Module(mod))
        add_builtin("edattr", Module(self.__dict__))

    def load_config(self):
        return run_file("config.rain")

    def echo(self, text):
        if self.mode != "COMMAND":
            self.cmd_input = text

    def set(self, key: str, val):
        setattr(self, key, val)

    def add_undo_history(self, arg):
        del self.undo_history[self.undo_version :]
        self.undo_history.append(arg)
        self.undo_version += 1

    def undo(self):
        if self.undo_version:
            self.undo_version -= 1
            cur_undo = self.undo_history[self.undo_version]
            if cur_undo[4] == "i":
                self.del_inrange(*cur_undo[:4])
                self.y, self.x = copy.deepcopy(cur_undo[:2])
            elif cur_undo[4] == "d":
                self.y, self.x = copy.deepcopy(cur_undo[:2])
                self.insert_any(cur_undo[5], True)

    def redo(self):
        if self.undo_version < len(self.undo_history):
            cur_undo = self.undo_history[self.undo_version]
            self.undo_version += 1
            if cur_undo[4] == "d":
                self.del_inrange(*cur_undo[:4])
                self.y, self.x = copy.deepcopy(cur_undo[:2])
            elif cur_undo[4] == "i":
                self.y, self.x = copy.deepcopy(cur_undo[:2])
                self.insert_any(cur_undo[5], True)

    def open_file(self):
        if self.file:
            self.lexer_cls, self.tok2hl = lexers.get(
                os.path.splitext(os.path.split(self.file)[1])[1],
                (unlexer.UnLexer, unlexer.un_tok2hl),
            )
            self.lexer = self.lexer_cls(self.text)
            try:
                with open(self.file, "r", encoding="utf-8") as f:
                    self.text.clear()
                    self.text.append([[]])
                    self.y = [0, 0]
                    self.ideal_x = self.x = 0
                    self.scroll_begin = [0, 0]
                    self.mode = "NORMAL"
                    self.insert_any(f.read())
                    self.saved_version = self.undo_version
            except FileNotFoundError:
                pass

    def get_all(self):
        return "\n".join(map(lambda a: "".join(map("".join, a)), self.text))

    def write_file(self):
        if self.file:
            try:
                with open(self.file, "w", encoding="utf-8") as f:
                    text = self.get_all()
                    f.write(text)
                    self.saved_version = self.undo_version
            except FileNotFoundError:
                pass

    def setmode_select(self):
        self.mode = "SELECT"
        self.selecty, self.selectx = copy.deepcopy(self.y), self.x

    def setmode_command(self):
        self.prev_mode = self.mode
        self.mode = "COMMAND"
        self.cmd_x = 1
        self.cmd_input = ":"

    def esc_cmdmode(self):
        self.mode = self.prev_mode
        self.cmd_x = 0
        self.cmd_input = ""

    def in_select(self, y, x):
        if self.mode != "SELECT":
            return False
        if self.cmp_2D(self.y, self.x, self.selecty, self.selectx) <= 0:
            return (
                self.cmp_2D(self.y, self.x, y, x) <= 0
                and self.cmp_2D(y, x, self.selecty, self.selectx) <= 0
            )
        else:
            return (
                self.cmp_2D(self.selecty, self.selectx, y, x) <= 0
                and self.cmp_2D(y, x, self.y, self.x) <= 0
            )

    def del_selected(self):
        content = self.get_selected()
        if self.cmp_2D(self.y, self.x, self.selecty, self.selectx) <= 0:
            beginy, beginx, endy, endx = self.y, self.x, self.selecty, self.selectx
        else:
            endy, endx, beginy, beginx = self.y, self.x, self.selecty, self.selectx
            self.y, self.x = copy.deepcopy(self.selecty), self.selectx
        self.lexer.change(beginy)
        self.add_undo_history(
            (copy.deepcopy(beginy), beginx, copy.deepcopy(endy), endx, "d", content)
        )
        self.del_inrange(beginy, beginx, endy, endx)
        self.mode = "NORMAL"
        self.ideal_x = self.x

    def get_selected(self):
        if self.cmp_2D(self.y, self.x, self.selecty, self.selectx) <= 0:
            beginy, beginx, endy, endx = self.y, self.x, self.selecty, self.selectx
        else:
            endy, endx, beginy, beginx = self.y, self.x, self.selecty, self.selectx
        return self.get_inrange(beginy, beginx, endy, endx)

    def scroll(self):
        if self.y_cmp(self.y, self.scroll_begin) < 0:
            self.scroll_begin = copy.deepcopy(self.y)
        else:
            cnt = 0
            y_copy = copy.deepcopy(self.y)
            while cnt < self.textspace_h - 1 and self.y_dec(y_copy):
                cnt += 1
            if self.y_cmp(y_copy, self.scroll_begin) > 0:
                self.scroll_begin = y_copy

    def search_next(self, text: str):
        if not text:
            self.echo("Pattern not defined")
            return
        y, x = copy.deepcopy(self.y), self.x
        y, x = self.inc_pos(y, x)
        cur = self.get_cur_char(y, x)
        matched = 0
        start_pos = [0, 0], 0
        while cur != None:
            if matched == len(text):
                break
            if cur == text[matched]:
                if matched == 0:
                    start_pos = copy.deepcopy(y), x
                matched += 1
            else:
                matched = 0
            if matched == len(text):
                break
            y, x = self.inc_pos(y, x)
            cur = self.get_cur_char(y, x)
        if matched == len(text):
            self.y, self.x = start_pos[0], start_pos[1]
        else:
            self.echo("Unable to find pattern")

    def search_prev(self, text: str):
        if not text:
            self.echo("Pattern not defined")
            return
        y, x = copy.deepcopy(self.y), self.x
        y, x = self.dec_pos(y, x)
        cur = self.get_cur_char(y, x)
        matched = 0
        while True:
            if matched == len(text):
                break
            if cur == text[-(matched + 1)]:
                matched += 1
            else:
                matched = 0
            if matched == len(text):
                break
            if y == [0, 0] and x == 0:
                break
            y, x = self.dec_pos(y, x)
            cur = self.get_cur_char(y, x)
        if matched == len(text):
            self.y, self.x = y, x
        else:
            self.echo("Unable to find pattern")

    def find_char_next(self, ch: str):
        if ch.isprintable():
            self.search_next(ch)

    def find_char_prev(self, ch: str):
        if ch.isprintable():
            self.search_prev(ch)

    # 不愧是Python，轻易就做到了我们无法做到的事
    # 终于知道为什么要在代码里写Fuck了（
    # 有一种依托答辩的美感（
    # 这要用i33绝对没这事
    def insert(self, text: list[str], is_undo=False):
        self.lexer.change(self.y)
        begin = copy.deepcopy(self.y), self.x
        next_pos = sum(map(len, self.text[self.y[0]][: self.y[1]])) + self.x + len(text)
        self.text[self.y[0]][self.y[1]] = (
            self.text[self.y[0]][self.y[1]][: self.x]
            + text
            + self.text[self.y[0]][self.y[1]][self.x :]
        )
        self.correct_line(self.y[0])
        self.y[1] = 0
        while (
            self.y[1] < len(self.text[self.y[0]]) - 1
            and next_pos - len(self.text[self.y[0]][self.y[1]]) >= 0
        ):
            next_pos -= len(self.text[self.y[0]][self.y[1]])
            self.y[1] += 1
        self.x = next_pos
        self.ideal_x = self.x
        end = copy.deepcopy(self.y), self.x
        end = self.dec_pos(*end)
        if not is_undo:
            self.add_undo_history((*begin, *end, "i", "".join(text)))

    def insert_enter(self, is_undo=False):
        self.lexer.change(self.y)
        begin = copy.deepcopy(self.y), self.x
        self.text[self.y[0]].insert(
            self.y[1] + 1, self.text[self.y[0]][self.y[1]][self.x :]
        )
        self.text[self.y[0]][self.y[1]] = self.text[self.y[0]][self.y[1]][: self.x]
        self.text.insert(self.y[0] + 1, self.text[self.y[0]][self.y[1] + 1 :])
        self.text[self.y[0]] = self.text[self.y[0]][: self.y[1] + 1]
        self.correct_line(self.y[0] + 1)
        self.y[0] += 1
        self.y[1] = 0
        self.x = 0
        self.ideal_x = self.x
        if not is_undo:
            self.add_undo_history((*begin, *begin, "i", "\n"))

    def del_before_cursor(self):
        if self.x == 0 and self.y[1] == 0:  # 删换行
            if self.y[0] == 0:
                return
            self.y_dec(self.y)
            self.x = len(self.text[self.y[0]][self.y[1]])
            self.text[self.y[0]].extend(self.text[self.y[0] + 1])
            del self.text[self.y[0] + 1]
            self.correct_line(self.y[0])
            begin = copy.deepcopy(self.y), self.x
            self.add_undo_history((*copy.deepcopy(begin), *begin, "d", "\n"))
        elif self.x == 0:  # 删字符
            self.y[1] -= 1
            self.x = len(self.text[self.y[0]][self.y[1]]) - 1
            content = self.text[self.y[0]][self.y[1]][self.x]
            del self.text[self.y[0]][self.y[1]][self.x]
            self.correct_line(self.y[0])
            begin = copy.deepcopy(self.y), self.x
            self.add_undo_history((*copy.deepcopy(begin), *begin, "d", content))
        else:
            self.x -= 1
            content = self.text[self.y[0]][self.y[1]][self.x]
            del self.text[self.y[0]][self.y[1]][self.x]
            self.correct_line(self.y[0])
            if self.x == 0 and self.y[1] != 0:
                self.y[1] -= 1
                self.x = len(self.text[self.y[0]][self.y[1]])
            begin = copy.deepcopy(self.y), self.x
            self.add_undo_history((*copy.deepcopy(begin), *begin, "d", content))
        self.ideal_x = self.x
        self.lexer.change(self.y)

    def insert_any(self, text: str, is_undo=False):
        if text == "":
            return
        self.lexer.change(self.y)
        begin = copy.deepcopy(self.y), self.x
        super().insert_any(text)
        end = copy.deepcopy(self.y), self.x
        end = self.dec_pos(*end)
        if not is_undo:
            self.add_undo_history((*begin, *end, "i", text))

    def move_cursor(self, dir: str):
        if dir == "up":
            if self.y_dec(self.y):
                self.x = min(self.ideal_x, len(self.text[self.y[0]][self.y[1]]))
        elif dir == "down":
            if self.y_inc(self.y):
                self.x = min(self.ideal_x, len(self.text[self.y[0]][self.y[1]]))
        elif dir == "left":
            self.y, self.x = self.dec_pos(self.y, self.x)
            self.ideal_x = self.x
        elif dir == "right":
            self.y, self.x = self.inc_pos(self.y, self.x)
            self.ideal_x = self.x
        elif dir == "home":
            self.ideal_x = self.x = 0
        elif dir == "linebegin":
            self.y[1] = 0
            self.x = 0
            while (
                self.y[1] < len(self.text[self.y[0]])
                and self.x < len(self.text[self.y[0]][self.y[1]])
                and self.text[self.y[0]][self.y[1]][self.x].isspace()
            ):
                if self.x < len(self.text[self.y[0]][self.y[1]]):
                    self.x += 1
                elif self.y[1] < len(self.text[self.y[0]]) - 1:
                    self.y_inc(self.y)
                    self.x = 0
            self.ideal_x = self.x
        elif dir == "end":
            self.ideal_x = self.x = len(self.text[self.y[0]][self.y[1]])
        elif dir == "pageup":
            for i in range(self.textspace_h):
                if not self.y_dec(self.y):
                    break
            self.x = min(self.ideal_x, len(self.text[self.y[0]][self.y[1]]))
        elif dir == "pagedown":
            for i in range(self.textspace_h):
                if not self.y_inc(self.y):
                    break
            self.x = min(self.ideal_x, len(self.text[self.y[0]][self.y[1]]))
        elif dir == "start":
            self.y = [0, 0]
            self.ideal_x = self.x = 0
        elif dir == "final":
            self.y = [len(self.text) - 1, len(self.text[-1]) - 1]
            self.ideal_x = self.x = len(self.text[self.y[0]][self.y[1]])

    def cmd_insert(self, ch: str):
        self.cmd_input = (
            self.cmd_input[: self.cmd_x] + ch + self.cmd_input[self.cmd_x :]
        )
        self.cmd_x += 1

    def cmd_paste(self, text: str):
        if "\n" in text or "\r" in text:
            return
        self.cmd_input += (
            self.cmd_input[: self.cmd_x] + text + self.cmd_input[self.cmd_x :]
        )
        self.cmd_x += len(text)

    def cmd_move_cursor(self, dir: str):
        if dir == "left":
            self.cmd_x -= 1
        elif dir == "right":
            self.cmd_x += 1
        elif dir == "begin" or dir == "up" or dir == "pageup" or dir == "start":
            self.cmd_x = 0
        elif dir == "end" or dir == "down" or dir == "pageup" or dir == "final":
            self.cmd_x = len(self.cmd_input)

    def cmd_del_before_cursor(self):
        self.cmd_x -= 1
        self.cmd_input = self.cmd_input[: self.cmd_x] + self.cmd_input[self.cmd_x + 1 :]

    def goto_next_word(self):
        # self.y, self.x = self.inc_pos(self.y, self.x)
        cur = self.get_cur_char(self.y, self.x)
        if cur is None:
            return
        if cur.isspace():
            while cur and cur.isspace():
                self.y, self.x = self.inc_pos(self.y, self.x)
                cur = self.get_cur_char(self.y, self.x)
        elif cur.isidentifier() or cur.isdigit():
            while cur and (cur.isidentifier() or cur.isdigit()):
                self.y, self.x = self.inc_pos(self.y, self.x)
                cur = self.get_cur_char(self.y, self.x)
        else:
            while cur and not (cur.isspace() or cur.isidentifier() or cur.isdigit()):
                self.y, self.x = self.inc_pos(self.y, self.x)
                cur = self.get_cur_char(self.y, self.x)

    def goto_prev_word(self):
        self.y, self.x = self.dec_pos(self.y, self.x)
        cur = self.get_cur_char(self.y, self.x)
        assert cur
        if cur.isspace():
            while cur and cur.isspace():
                if self.y == [0, 0] and self.x == 0:
                    return
                self.y, self.x = self.dec_pos(self.y, self.x)
                cur = self.get_cur_char(self.y, self.x)
        elif cur.isidentifier() or cur.isdigit():
            while cur and (cur.isidentifier() or cur.isdigit()):
                if self.y == [0, 0] and self.x == 0:
                    return
                self.y, self.x = self.dec_pos(self.y, self.x)
                cur = self.get_cur_char(self.y, self.x)
        else:
            while cur and not (cur.isspace() or cur.isidentifier() or cur.isdigit()):
                if self.y == [0, 0] and self.x == 0:
                    return
                self.y, self.x = self.dec_pos(self.y, self.x)
                cur = self.get_cur_char(self.y, self.x)
        self.y, self.x = self.inc_pos(self.y, self.x)

    def run_cmd(self, cmd: str):
        cmda = cmd[1:].split(" ", 1)
        while len(cmda) < 2:
            cmda.append("")
        head, arg = cmda
        head = head.strip()
        arg = arg.strip()
        if head == "q":
            self.exit = True
        elif head == "o":
            if arg:
                self.file = arg
            self.open_file()
        elif head == "w":
            if arg:
                self.file = arg
            self.write_file()
        elif head == "f":
            self.search_pattern = arg
            self.search_next(self.search_pattern)
        elif head == "F":
            self.search_pattern = arg
            self.search_prev(self.search_pattern)
        else:
            self.echo("Unknown cmd")

    def accept_cmd(self):
        self.run_cmd(self.cmd_input)
        self.cmd_x = 0
        self.cmd_input = ""
        self.mode = self.prev_mode

    def draw_modeline_minibuffer(self):
        mb_h = 1
        cur_w = 0
        for ch in self.cmd_input:
            ch_w = get_width(ch)
            if cur_w + ch_w > self.textspace_w:
                mb_h += 1
                cur_w = 0
            cur_w += ch_w
        self.textspace_h = self.screen_h - mb_h - 1
        for i in range(self.textspace_w + self.linum_w):
            self.screen.change(self.textspace_h, i, " ", "\033[47m")
        for i in range(mb_h):
            for j in range(self.textspace_w):
                self.screen.change(self.textspace_h + 1 + i, j, " ", "")
        # modeline = self.mode + \
        #     f"   ln: {self.y[0] + 1} + {self.y[1]}, col: {self.x + 1}"
        modeline = "%s  " % (self.mode)
        total_w = self.textspace_w + self.linum_w
        modelinum = "  ln: %d + %d, col: %d " % (*self.y, self.x)
        spare_w = total_w - len(modeline + modelinum)
        if self.file:
            file = self.file
        else:
            file = ""
        if self.undo_version != self.saved_version:
            if len(file) > spare_w:
                fname = file[: spare_w - 5] + "..[+]"
            else:
                fname = file + "[+]" + " " * (spare_w - len(file) - 3)
        else:
            if len(file) > spare_w:
                fname = file[: spare_w - 2] + ".."
            else:
                fname = file + " " * (spare_w - len(file))
        modeline = modeline + fname + modelinum
        shift = 0
        for ch in modeline:
            self.screen.change(self.textspace_h, shift + 1, ch, "\033[47;30m")
            shift += get_width(ch)
        cur_h = 1
        cur_w = 0
        for ch in self.cmd_input:
            ch_w = get_width(ch)
            if cur_w + ch_w > self.textspace_w:
                cur_h += 1
                cur_w = 0
            self.screen.change(self.textspace_h + cur_h, cur_w, ch, "")
            cur_w += ch_w

    def draw_textspace(self):
        cur = copy.deepcopy(self.scroll_begin)
        isend = False
        # if self.file and os.path.splitext(os.path.split(self.file)[1])[1] in renderers:
            # rendered = renderers[os.path.splitext(os.path.split(self.file)[1])[1]](
                # self.get_all(), self.textspace_w, self.colorscheme
            # )
        # else:
            # rendered = self.text
        for i in range(self.textspace_h):
            shift = 0
            if not isend and self.show_linum and cur[1] == 0:
                linum = f"%{self.linum_w - 1}d " % (cur[0] + 1)
                for ch in linum:
                    self.screen.change(i, shift, ch, "\033[1;33m")
                    shift += get_width(ch)
            if self.show_linum and not isend:
                shift = self.linum_w
            else:
                shift = 0
            if not isend:
                for j in range(len(self.text[cur[0]][cur[1]])):
                    if self.mode == "SELECT" and self.in_select(cur, j):
                        self.screen.change(
                            i,
                            shift,
                            self.text[cur[0]][cur[1]][j],
                            self.colorscheme[self.tok2hl[self.lexer.get(cur, j)]] + "\033[1;47m",
                        )
                    else:
                        # print(self.lexer_cls, self.lexer, self.lexer.get(cur, j), cur, j)
                        self.screen.change(
                            i,
                            shift,
                            self.text[cur[0]][cur[1]][j],
                            self.colorscheme[self.tok2hl[self.lexer.get(cur, j)]],
                        )
                    for x in range(1, get_width(self.text[cur[0]][cur[1]][j])):
                        self.screen.change(i, shift + x, "", "")
                    shift += get_width(self.text[cur[0]][cur[1]][j])
                isend = not self.y_inc(cur)
            for j in range(shift, self.textspace_w):
                self.screen.change(i, j, " ", "")
        # self.echo('|'.join(map(lambda a: str(a), self.screen.data[:1])))
        # self.echo(str(self.text))
        self.screen.refresh()

    def draw_cursor(self):
        if self.mode != "COMMAND":
            ln = 0
            cur_y = copy.deepcopy(self.scroll_begin)
            while self.y_cmp(cur_y, self.y) < 0:
                ln += 1
                if not self.y_inc(cur_y):
                    break
            col = sum(map(get_width, self.text[self.y[0]][self.y[1]][: self.x]))
            gotoxy(ln + 1, col + 1 + self.linum_w)
        else:
            y = 0
            x = 0
            for ch in self.cmd_input[: self.cmd_x]:
                ch_w = get_width(ch)
                if x + ch_w > self.textspace_w:
                    y += 1
                    x = 0
                x += ch_w
            gotoxy(self.textspace_h + 2 + y, x + 1)
        flush()

    def update(self):
        self.draw_modeline_minibuffer()
        self.scroll()
        self.lexer.lex(min(len(self.text), self.scroll_begin[0] + self.textspace_h))
        self.draw_textspace()
        self.draw_cursor()

    def mainloop(self):
        while not self.exit:
            # self.echo(f'{self.undo_history} {self.undo_version}')
            self.update()
            key = getch()
            num = 0
            if self.mode != "INSERT" and ord("1") <= ord(key) <= ord("9"):
                while key.isdigit():
                    num *= 10
                    num += ord(key) - ord("0")
                    key = getch()
            else:
                num = 1
            for i in range(num):
                if key in self.keymaps[self.mode]:
                    x = self.keymaps[self.mode][key]
                    while isinstance(x, dict):
                        key = getch()
                        if key not in x:
                            break
                        x = x[key]
                    if callable(x):
                        x()
                    elif isinstance(x, Func):
                        call_Func(x, self.config_code, [])
                elif self.mode == "INSERT" and key.isprintable():
                    self.insert([key])
                elif self.mode == "INSERT" and key in "\n\r":
                    self.insert_enter()
                elif self.mode == "COMMAND" and key.isprintable():
                    self.cmd_insert(key)


clear()
if len(sys.argv) == 2:
    file = sys.argv[1]
else:
    file = None
editor = Editor(os.get_terminal_size().lines, os.get_terminal_size().columns, file)
editor.mainloop()
