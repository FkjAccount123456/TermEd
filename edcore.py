from msvcrt import getwch as getch, kbhit
import copy
import pyperclip
import os
import sys


def clear():
    os.system("cls")


def gotoxy(y, x):
    print(f"\033[{y};{x}f", end="")


def flush():
    print(flush=True, end='')


Pos = tuple[int, int]
widths = [
    (126, 1), (159, 0), (687, 1), (710, 0), (711, 1),
    (727, 0), (733, 1), (879, 0), (1154, 1), (1161, 0),
    (4347, 1), (4447, 2), (7467, 1), (7521, 0), (8369, 1),
    (8426, 0), (9000, 1), (9002, 2), (11021, 1), (12350, 2),
    (12351, 1), (12438, 2), (12442, 0), (19893, 2), (19967, 1),
    (55203, 2), (63743, 1), (64106, 2), (65039, 1), (65059, 0),
    (65131, 2), (65279, 1), (65376, 2), (65500, 1), (65510, 2),
    (120831, 1), (262141, 2), (1114109, 1),
]


def get_width(o):
    # 参考了https://wenku.baidu.com/view/da48663551d380eb6294dd88d0d233d4b14e3f18.html?
    """Return the screen column width for unicode ordinal o."""
    global widths
    o = ord(o)
    if o == 0xe or o == 0xf:
        return 0
    if o == 0x9:
        return 8
    for num, wid in widths:
        if o <= num:
            return wid
    return 1


class Screen:
    def __init__(self, h: int, w: int):
        self.h, self.w = h, w
        self.data = [[' ' for i in range(w)] for j in range(h)]
        self.color = [['' for i in range(w)] for j in range(h)]
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
        for y, x in self.changed:
            gotoxy(y + 1, x + 1)
            print(self.color[y][x] + self.data[y][x], end='\033[0m')
        self.changed = set()


class Editor:
    def __init__(self, h: int, w: int, file: str | None = None):
        self.screen = Screen(h, w)
        self.y, self.x = [0, 0], 0
        self.selecty, self.selectx = [0, 0], 0
        self.ideal_x = 0
        self.screen_h = h
        self.textspace_h = self.screen_h - 2
        self.screen_w = w - 1
        self.textspace_w = self.screen_w
        self.linum_w = 0

        self.scroll_begin = [0, 0]
        self.text: list[list[list[str]]] = [[[]]]

        self.exit = False
        self.mode = 'NORMAL'

        self.cmd_input = ""
        self.cmd_x = 0

        self.file = file

        self.show_linum = True

        if self.show_linum:
            self.textspace_w -= 6
            self.linum_w = 6

        self.keymaps = {
            'NORMAL': {
                'i': lambda: self.__setattr__("mode", "INSERT"),
                'v': self.setmode_select,
                ':': self.setmode_command,
                # 'q': lambda: self.__setattr__("exit", True),
                'h': lambda: self.move_cursor('left'),
                'j': lambda: self.move_cursor('down'),
                'k': lambda: self.move_cursor('up'),
                'l': lambda: self.move_cursor('right'),
                '0': lambda: self.move_cursor('home'),
                '$': lambda: self.move_cursor('end'),
                'p': lambda: self.insert_any(pyperclip.paste()),
                'g': {
                    'g': lambda: self.move_cursor('start'),
                },
                'G': lambda: self.move_cursor('final'),
                '\xe0': {
                    'K': lambda: self.move_cursor('left'),
                    'P': lambda: self.move_cursor('down'),
                    'H': lambda: self.move_cursor('up'),
                    'M': lambda: self.move_cursor('right'),
                    'G': lambda: self.move_cursor('home'),
                    'O': lambda: self.move_cursor('end'),
                    'I': lambda: self.move_cursor('pageup'),
                    'Q': lambda: self.move_cursor('pagedown'),
                },
            },
            'INSERT': {
                '\x1b': lambda: self.__setattr__("mode", "NORMAL"),
                '\x08': self.del_before_cursor,
                '\t': lambda: self.insert([' ', ' ', ' ', ' ']),
                '\xe0': {
                    'K': lambda: self.move_cursor('left'),
                    'P': lambda: self.move_cursor('down'),
                    'H': lambda: self.move_cursor('up'),
                    'M': lambda: self.move_cursor('right'),
                    'G': lambda: self.move_cursor('home'),
                    'O': lambda: self.move_cursor('end'),
                    'I': lambda: self.move_cursor('pageup'),
                    'Q': lambda: self.move_cursor('pagedown'),
                },
            },
            'SELECT': {
                '\x1b': lambda: self.__setattr__("mode", "NORMAL"),
                'h': lambda: self.move_cursor('left'),
                'j': lambda: self.move_cursor('down'),
                'k': lambda: self.move_cursor('up'),
                'l': lambda: self.move_cursor('right'),
                '0': lambda: self.move_cursor('home'),
                '$': lambda: self.move_cursor('end'),
                'd': self.del_selected,
                'y': lambda: pyperclip.copy(self.get_selected()),
                'g': {
                    'g': lambda: self.move_cursor('start'),
                },
                'G': lambda: self.move_cursor('final'),
                '\xe0': {
                    'K': lambda: self.move_cursor('left'),
                    'P': lambda: self.move_cursor('down'),
                    'H': lambda: self.move_cursor('up'),
                    'M': lambda: self.move_cursor('right'),
                    'G': lambda: self.move_cursor('home'),
                    'O': lambda: self.move_cursor('end'),
                    'I': lambda: self.move_cursor('pageup'),
                    'Q': lambda: self.move_cursor('pagedown'),
                },
            },
            'COMMAND': {
                '\x1b': self.esc_cmdmode,
                '\x08': self.cmd_del_before_cursor,
                '\x19': lambda: self.cmd_paste(pyperclip.paste()),
                '\n': self.accept_cmd,
                '\r': self.accept_cmd,
                '\xe0': {
                    'K': lambda: self.cmd_move_cursor('left'),
                    'P': lambda: self.cmd_move_cursor('down'),
                    'H': lambda: self.cmd_move_cursor('up'),
                    'M': lambda: self.cmd_move_cursor('right'),
                    'G': lambda: self.cmd_move_cursor('home'),
                    'O': lambda: self.cmd_move_cursor('end'),
                    'I': lambda: self.cmd_move_cursor('pageup'),
                    'Q': lambda: self.cmd_move_cursor('pagedown'),
                },
            }
        }

        self.open_file()

    def open_file(self):
        if self.file:
            try:
                with open(self.file, 'r', encoding='utf-8') as f:
                    self.text = [[[]]]
                    self.y = [0, 0]
                    self.ideal_x = self.x = 0
                    self.scroll_begin = [0, 0]
                    self.mode = 'NORMAL'
                    self.insert_any(f.read())
            except:
                pass

    def write_file(self):
        if self.file:
            try:
                with open(self.file, 'w', encoding='utf-8') as f:
                    text = '\n'.join(map(lambda a: ''.join(map(''.join, a)), self.text))
                    f.write(text)
            except:
                pass

    def setmode_select(self):
        self.mode = "SELECT"
        self.selecty, self.selectx = copy.deepcopy(self.y), self.x

    def setmode_command(self):
        self.mode = "COMMAND"
        self.cmd_x = 1
        self.cmd_input = ':'

    def esc_cmdmode(self):
        self.mode = "NORMAL"
        self.cmd_x = 0
        self.cmd_input = ''

    def cmp_2D(self, y1, x1, y2, x2):
        if y1[0] != y2[0]:
            return y1[0] - y2[0]
        elif y1[1] != y2[1]:
            return y1[1] - y2[1]
        else:
            return x1 - x2

    def in_select(self, y, x):
        if self.mode != 'SELECT':
            return False
        if self.cmp_2D(self.y, self.x, self.selecty, self.selectx) <= 0:
            return self.cmp_2D(self.y, self.x, y, x) <= 0 and \
                self.cmp_2D(y, x, self.selecty, self.selectx) <= 0
        else:
            return self.cmp_2D(self.selecty, self.selectx, y, x) <= 0 and \
                self.cmp_2D(y, x, self.y, self.x) <= 0

    def del_selected(self):
        if self.cmp_2D(self.y, self.x, self.selecty, self.selectx) <= 0:
            beginy, beginx, endy, endx = self.y, self.x, self.selecty, self.selectx
        else:
            endy, endx, beginy, beginx = self.y, self.x, self.selecty, self.selectx
            self.y, self.x = copy.deepcopy(self.selecty), self.selectx
        if beginy[0] == endy[0]:
            if endy[1] == len(self.text[endy[0]]) - 1 and\
                    endx == len(self.text[endy[0]][endy[1]]) and\
                    endy[0] < len(self.text) - 1:
                self.text[endy[0]].extend(self.text[endy[0] + 1])
                del self.text[endy[0] + 1]
            if beginy[1] == endy[1]:
                del self.text[beginy[0]][beginy[1]][beginx: endx + 1]
            else:
                del self.text[beginy[0]][beginy[1]][beginx:]
                del self.text[beginy[0]][endy[1]][:endx + 1]
                del self.text[beginy[0]][beginy[1] + 1: endy[1]]
            self.correct_line(endy[0])
        else:
            if endy[1] == len(self.text[endy[0]]) - 1 and\
                    endx == len(self.text[endy[0]][endy[1]]) and\
                    endy[0] < len(self.text) - 1:
                self.text[endy[0]].extend(self.text[endy[0] + 1])
                del self.text[endy[0] + 1]
            del self.text[beginy[0]][beginy[1]][beginx:]
            del self.text[beginy[0]][beginy[1] + 1:]
            del self.text[endy[0]][endy[1]][:endx + 1]
            del self.text[endy[0]][:endy[1]]
            self.text[beginy[0]].extend(self.text[endy[0]])
            del self.text[endy[0]]
            self.correct_line(beginy[0])
            del self.text[beginy[0] + 1: endy[0]]
        self.mode = "NORMAL"
        self.ideal_x = self.x

    def get_selected(self):
        if self.cmp_2D(self.y, self.x, self.selecty, self.selectx) <= 0:
            beginy, beginx, endy, endx = self.y, self.x, self.selecty, self.selectx
        else:
            endy, endx, beginy, beginx = self.y, self.x, self.selecty, self.selectx
        if beginy[0] == endy[0]:
            if beginy[1] == endy[1]:
                res = "".join(self.text[beginy[0]]
                              [beginy[1]][beginx: endx + 1])
            else:
                res = "".join(self.text[beginy[0]][beginy[1]][beginx:])
                res += "".join(map("".join,
                               self.text[beginy[0]][beginy[1] + 1: endy[1]]))
                res += "".join(self.text[beginy[0]][endy[1]][:endx + 1])
            if endy[1] == len(self.text[endy[0]]) - 1 and\
                    endx == len(self.text[endy[0]][endy[1]]) and\
                    endy[0] < len(self.text) - 1:
                res += '\n'
        else:
            res = "".join(self.text[beginy[0]][beginy[1]][beginx:])
            res += "".join(map("".join,
                           self.text[beginy[0]][beginy[1] + 1:])) + '\n'
            res += '\n'.join(map(lambda a: ''.join(map(''.join, a)),
                             self.text[beginy[0] + 1: endy[0]])) + '\n'
            res += "".join(map("".join,
                           self.text[endy[0]][endy[1] + 1:]))
            res += "".join(self.text[endy[0]][endy[1]][:endx + 1])
            if endy[1] == len(self.text[endy[0]]) - 1 and\
                    endx == len(self.text[endy[0]][endy[1]]) and\
                    endy[0] < len(self.text) - 1:
                res += '\n'
        return res

    def y_cmp(self, y1, y2):
        if y1[0] != y2[0]:
            return y1[0] - y2[0]
        else:
            return y1[1] - y2[1]

    def y_inc(self, y):
        if y[1] < len(self.text[y[0]]) - 1:
            y[1] += 1
        elif y[0] < len(self.text) - 1:
            y[0] += 1
            y[1] = 0
        else:
            return False
        return True

    def y_dec(self, y):
        if y[1] > 0:
            y[1] -= 1
        elif y[0] > 0:
            y[0] -= 1
            y[1] = len(self.text[y[0]]) - 1
        else:
            return False
        return True

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

    def correct_line(self, linum: int):
        data = "".join(map("".join, self.text[linum]))
        self.text[linum] = [[]]
        cur_w = 0
        for ch in data:
            ch_w = get_width(ch)
            if cur_w + ch_w > self.textspace_w:
                self.text[linum].append([])
                cur_w = 0
            cur_w += ch_w
            self.text[linum][-1].append(ch)

    # 不愧是Python，轻易就做到了我们无法做到的事
    # 终于知道为什么要在代码里写Fuck了（
    # 有一种依托答辩的美感（
    # 这要用i33绝对没这事
    def insert(self, text: list[str]):
        next_pos = sum(
            map(len, self.text[self.y[0]][:self.y[1]])) + self.x + len(text)
        self.text[self.y[0]][self.y[1]] = (self.text[self.y[0]][self.y[1]][:self.x]
                                           + text
                                           + self.text[self.y[0]][self.y[1]][self.x:])
        self.correct_line(self.y[0])
        self.y[1] = 0
        while self.y[1] < len(self.text[self.y[0]]) - 1 and \
                next_pos - len(self.text[self.y[0]][self.y[1]]) >= 0:
            next_pos -= len(self.text[self.y[0]][self.y[1]])
            self.y[1] += 1
        self.x = next_pos
        self.ideal_x = self.x

    def insert_enter(self):
        self.text[self.y[0]].insert(
            self.y[1] + 1, self.text[self.y[0]][self.y[1]][self.x:])
        self.text[self.y[0]][self.y[1]
                             ] = self.text[self.y[0]][self.y[1]][:self.x]
        self.text.insert(self.y[0] + 1, self.text[self.y[0]][self.y[1] + 1:])
        self.text[self.y[0]] = self.text[self.y[0]][:self.y[1] + 1]
        self.correct_line(self.y[0] + 1)
        self.y[0] += 1
        self.y[1] = 0
        self.x = 0
        self.ideal_x = self.x

    def del_before_cursor(self):
        if self.x == 0 and self.y[1] == 0:  # 删换行
            if self.y[0] == 0:
                return
            self.y_dec(self.y)
            self.x = len(self.text[self.y[0]][self.y[1]])
            self.text[self.y[0]].extend(self.text[self.y[0] + 1])
            self.correct_line(self.y[0])
        elif self.x == 0:  # 删字符
            self.y[1] -= 1
            self.x = len(self.text[self.y[0]][self.y[1]]) - 1
            del self.text[self.y[0]][self.y[1]][self.x]
            self.correct_line(self.y[0])
        else:
            self.x -= 1
            del self.text[self.y[0]][self.y[1]][self.x]
            self.correct_line(self.y[0])
            if self.x == 0 and self.y[1] != 0:
                self.y[1] -= 1
                self.x = len(self.text[self.y[0]][self.y[1]])
        self.ideal_x = self.x

    def insert_any(self, text: str):
        tmp = []
        for ch in text:
            if ch == '\n':
                self.text[self.y[0]].insert(
                    self.y[1] + 1, self.text[self.y[0]][self.y[1]][self.x:])
                del self.text[self.y[0]][self.y[1]][self.x:]
                self.text.insert(
                    self.y[0] + 1, self.text[self.y[0]][self.y[1] + 1:])
                del self.text[self.y[0]][self.y[1] + 1:]
                self.text[self.y[0]].append(tmp)
                self.correct_line(self.y[0])
                self.y[0] += 1
                self.y[1] = 0
                self.x = 0
                tmp = []
            elif ch == '\r':
                pass
            else:
                tmp.append(ch)
        if tmp:
            self.text[self.y[0]][self.y[1]] = \
                self.text[self.y[0]][self.y[1]][:self.x] + \
                tmp + self.text[self.y[0]][self.y[1]][self.x:]
            self.correct_line(self.y[0])
            self.x += len(tmp)
        self.ideal_x = self.x

    def move_cursor(self, dir: str):
        if dir == 'up':
            if self.y_dec(self.y):
                self.x = min(self.ideal_x, len(
                    self.text[self.y[0]][self.y[1]]))
        elif dir == 'down':
            if self.y_inc(self.y):
                self.x = min(self.ideal_x, len(
                    self.text[self.y[0]][self.y[1]]))
        elif dir == 'left':
            if self.x:
                self.x = self.x - 1
                if self.x == 0 and self.y[1] != 0:
                    self.y[1] -= 1
                    self.x = len(self.text[self.y[0]][self.y[1]])
            else:
                self.y_dec(self.y)
                self.x = len(self.text[self.y[0]][self.y[1]])
            self.ideal_x = self.x
        elif dir == 'right':
            if self.x < len(self.text[self.y[0]][self.y[1]]):
                self.x += 1
            else:
                self.y_inc(self.y)
                self.x = 0
            self.ideal_x = self.x
        elif dir == 'home':
            self.ideal_x = self.x = 0
        elif dir == 'end':
            self.ideal_x = self.x = len(self.text[self.y[0]][self.y[1]])
        elif dir == 'pageup':
            for i in range(self.textspace_h):
                if not self.y_dec(self.y):
                    break
            self.x = min(self.ideal_x, len(self.text[self.y[0]][self.y[1]]))
        elif dir == 'pagedown':
            for i in range(self.textspace_h):
                if not self.y_inc(self.y):
                    break
            self.x = min(self.ideal_x, len(self.text[self.y[0]][self.y[1]]))
        elif dir == 'start':
            self.y = [0, 0]
            self.x = 0
        elif dir == 'final':
            self.y = [len(self.text) - 1, len(self.text[-1]) - 1]
            self.x = len(self.text[self.y[0]][self.y[1]])

    def cmd_insert(self, ch: str):
        self.cmd_input = self.cmd_input[:self.cmd_x] + \
            ch + self.cmd_input[self.cmd_x:]
        self.cmd_x += 1

    def cmd_paste(self, text: str):
        if '\n' in text or '\r' in text:
            return
        self.cmd_input += self.cmd_input[:self.cmd_x] + \
            text + self.cmd_input[self.cmd_x:]
        self.cmd_x += len(text)

    def cmd_move_cursor(self, dir: str):
        if dir == 'left':
            self.cmd_x -= 1
        elif dir == 'right':
            self.cmd_x += 1
        elif dir == 'begin' or dir == 'up' or dir == 'pageup' or dir == 'start':
            self.cmd_x = 0
        elif dir == 'end' or dir == 'down' or dir == 'pageup' or dir == 'final':
            self.cmd_x = len(self.cmd_input)

    def cmd_del_before_cursor(self):
        self.cmd_x -= 1
        self.cmd_input = self.cmd_input[:self.cmd_x] + \
            self.cmd_input[self.cmd_x + 1:]

    def run_cmd(self, cmd: str):
        cmda = cmd[1:].split(' ', 1)
        while len(cmda) < 2:
            cmda.append('')
        head, arg = cmda
        head = head.strip()
        arg = arg.strip()
        if head == 'q':
            self.exit = True
        elif head == 'o':
            if arg:
                self.file = arg
            self.open_file()
        elif head == 'w':
            if arg:
                self.file = arg
            self.write_file()

    def accept_cmd(self):
        self.run_cmd(self.cmd_input)
        self.cmd_x = 0
        self.cmd_input = ''
        self.mode = "NORMAL"

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
        for i in range(self.textspace_w):
            self.screen.change(self.textspace_h, i, ' ', '\033[47m')
        for i in range(mb_h):
            for j in range(self.textspace_w):
                self.screen.change(self.textspace_h + 1 + i, j, ' ', '')
        modeline = self.mode + \
            f"   ln: {self.y[0] + 1} + {self.y[1]}, col: {self.x + 1}"
        shift = 0
        for ch in modeline:
            self.screen.change(self.textspace_h, shift + 1, ch, '\033[47;30m')
            shift += get_width(ch)
        cur_h = 1
        cur_w = 0
        for ch in self.cmd_input:
            ch_w = get_width(ch)
            if cur_w + ch_w > self.textspace_w:
                cur_h += 1
                cur_w = 0
            self.screen.change(self.textspace_h + cur_h, cur_w, ch, '')
            cur_w += ch_w

    def draw_textspace(self):
        cur = copy.deepcopy(self.scroll_begin)
        isend = False
        for i in range(self.textspace_h):
            shift = 0
            if not isend and self.show_linum and cur[1] == 0:
                linum = "%5d "%(cur[0] + 1)
                for ch in linum:
                    self.screen.change(i, shift, ch, '\033[1;33m')
                    shift += get_width(ch)
            if not isend:
                for j in range(len(self.text[cur[0]][cur[1]])):
                    if self.mode == "SELECT" and self.in_select(cur, j):
                        self.screen.change(
                            i, shift, self.text[cur[0]][cur[1]][j], '\033[47;30m')
                    else:
                        self.screen.change(
                            i, shift, self.text[cur[0]][cur[1]][j], '')
                    for x in range(1, get_width(self.text[cur[0]][cur[1]][j])):
                        self.screen.change(i, shift + x, '', '')
                    shift += get_width(self.text[cur[0]][cur[1]][j])
                isend = not self.y_inc(cur)
            for j in range(shift, self.textspace_w):
                self.screen.change(i, j, ' ', '')
        self.screen.refresh()

    def draw_cursor(self):
        if self.mode != "COMMAND":
            ln = 0
            cur_y = copy.deepcopy(self.scroll_begin)
            while self.y_cmp(cur_y, self.y) < 0:
                ln += 1
                if not self.y_inc(cur_y):
                    break
            col = sum(map(get_width, self.text[self.y[0]][self.y[1]][:self.x]))
            gotoxy(ln + 1, col + 1 + self.linum_w)
        else:
            y = 0
            x = 0
            for ch in self.cmd_input[:self.cmd_x]:
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
        self.draw_textspace()
        self.draw_cursor()

    def mainloop(self):
        while not self.exit:
            self.update()
            key = getch()
            num = 0
            if self.mode != 'INSERT' and ord('1') <= ord(key) <= ord('9'):
                while key.isdigit():
                    num *= 10
                    num += ord(key) - ord('0')
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
                elif self.mode == "INSERT" and key.isprintable():
                    self.insert([key])
                elif self.mode == "INSERT" and key in '\n\r':
                    self.insert_enter()
                elif self.mode == "COMMAND" and key.isprintable():
                    self.cmd_insert(key)


if len(sys.argv) == 2:
    file = sys.argv[1]
else:
    file = None
editor = Editor(os.get_terminal_size().lines, os.get_terminal_size().columns, file)
editor.mainloop()
