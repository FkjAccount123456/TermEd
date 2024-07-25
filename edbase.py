from utils import get_width


class TextSpace:
    def __init__(self, textspace_w):
        self.y, self.x = [0, 0], 0
        self.ideal_x = 0
        self.textspace_w = textspace_w
        self.text = [[[]]]

    def cmp_2D(self, y1, x1, y2, x2):
        if y1[0] != y2[0]:
            return y1[0] - y2[0]
        elif y1[1] != y2[1]:
            return y1[1] - y2[1]
        else:
            return x1 - x2

    def del_inrange(self, beginy, beginx, endy, endx):
        if beginy[0] == endy[0]:
            if (
                endy[1] == len(self.text[endy[0]]) - 1
                and endx == len(self.text[endy[0]][endy[1]])
                and endy[0] < len(self.text) - 1
            ):
                self.text[endy[0]].extend(self.text[endy[0] + 1])
                del self.text[endy[0] + 1]
            if beginy[1] == endy[1]:
                del self.text[beginy[0]][beginy[1]][beginx : endx + 1]
            else:
                del self.text[beginy[0]][beginy[1]][beginx:]
                del self.text[beginy[0]][endy[1]][: endx + 1]
                del self.text[beginy[0]][beginy[1] + 1 : endy[1]]
            self.correct_line(endy[0])
        else:
            if (
                endy[1] == len(self.text[endy[0]]) - 1
                and endx == len(self.text[endy[0]][endy[1]])
                and endy[0] < len(self.text) - 1
            ):
                self.text[endy[0]].extend(self.text[endy[0] + 1])
                del self.text[endy[0] + 1]
            del self.text[beginy[0]][beginy[1]][beginx:]
            del self.text[beginy[0]][beginy[1] + 1 :]
            del self.text[endy[0]][endy[1]][: endx + 1]
            del self.text[endy[0]][: endy[1]]
            self.text[beginy[0]].extend(self.text[endy[0]])
            del self.text[endy[0]]
            self.correct_line(beginy[0])
            del self.text[beginy[0] + 1 : endy[0]]

    def get_inrange(self, beginy, beginx, endy, endx):
        if beginy[0] == endy[0]:
            if beginy[1] == endy[1]:
                res = "".join(self.text[beginy[0]][beginy[1]][beginx : endx + 1])
            else:
                res = "".join(self.text[beginy[0]][beginy[1]][beginx:])
                res += "".join(
                    map("".join, self.text[beginy[0]][beginy[1] + 1 : endy[1]])
                )
                res += "".join(self.text[beginy[0]][endy[1]][: endx + 1])
            if (
                endy[1] == len(self.text[endy[0]]) - 1
                and endx == len(self.text[endy[0]][endy[1]])
                and endy[0] < len(self.text) - 1
            ):
                res += "\n"
        else:
            res = "".join(self.text[beginy[0]][beginy[1]][beginx:])
            if beginy[1] + 1 < len(self.text[beginy[0]]):
                res += "".join(map("".join, self.text[beginy[0]][beginy[1] + 1 :]))
            res += "\n"
            # echo = [list(res)]
            if endy[0] > beginy[0] + 1:
                res += (
                    "\n".join(
                        map(
                            lambda a: "".join(map("".join, a)),
                            self.text[beginy[0] + 1 : endy[0]],
                        )
                    )
                    + "\n"
                )
            # echo += [list(res)]
            res += "".join(map("".join, self.text[endy[0]][endy[1] + 1 :]))
            res += "".join(self.text[endy[0]][endy[1]][: endx + 1])
            # echo += [list(res)]
            if (
                endy[1] == len(self.text[endy[0]]) - 1
                and endx == len(self.text[endy[0]][endy[1]])
                and endy[0] < len(self.text) - 1
            ):
                res += "\n"
            # self.echo(str(echo))
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

    def dec_pos(self, y, x):
        if x > 0:
            x -= 1
            if x == 0 and y[1] > 0:
                y[1] -= 1
                x = len(self.text[y[0]][y[1]])
        elif y[1] > 0:
            y[1] -= 1
            x = len(self.text[y[0]][y[1]])
        elif y[0] > 0:
            y[0] -= 1
            y[1] = len(self.text[y[0]]) - 1
            x = len(self.text[y[0]][y[1]])
        return y, x

    def inc_pos(self, y, x):
        if x < len(self.text[y[0]][y[1]]):
            x += 1
        elif y[1] < len(self.text[y[0]]) - 1:
            y[1] += 1
            x = 0
            if x < len(self.text[y[0]][y[1]]):
                x += 1
        elif y[0] < len(self.text) - 1:
            y[0] += 1
            y[1] = 0
            x = 0
        return y, x

    def get_cur_char(self, y, x):
        if x < len(self.text[y[0]][y[1]]):
            return self.text[y[0]][y[1]][x]
        elif y[1] < len(self.text[y[0]]) - 1:
            return self.text[y[0]][y[1] + 1][0]
        elif y[0] < len(self.text) - 1:
            return "\n"
        else:
            return None

    def insert_any(self, text):
        tmp = []
        for ch in text:
            if ch == "\n" or ch == None:
                self.text[self.y[0]].insert(
                    self.y[1] + 1, self.text[self.y[0]][self.y[1]][self.x :]
                )
                del self.text[self.y[0]][self.y[1]][self.x :]
                self.text.insert(self.y[0] + 1, self.text[self.y[0]][self.y[1] + 1 :])
                del self.text[self.y[0]][self.y[1] + 1 :]
                self.text[self.y[0]].append(tmp)
                self.correct_line(self.y[0])
                self.y[0] += 1
                self.y[1] = 0
                self.x = 0
                tmp = []
            elif ch == "\r":
                pass
            else:
                tmp.append(ch)
        if tmp:
            self.text[self.y[0]][self.y[1]] = (
                self.text[self.y[0]][self.y[1]][: self.x]
                + tmp
                + self.text[self.y[0]][self.y[1]][self.x :]
            )
            self.correct_line(self.y[0])
            for i in range(len(tmp)):
                self.y, self.x = self.inc_pos(self.y, self.x)
        self.ideal_x = self.x
