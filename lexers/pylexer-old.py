"""
复杂的代码，难度不亚于写edcore
（付-2021：getvarinhz）（
但是效率是真的高
现在用不上tree-sitter，这个就是我手里最先进的高亮程序
（TermEd-ng：反正以后就换tree-sitter了）（

专治各种不服——TermEd Python增量词法分析源码
什么？看起来不复杂？你读去啊！（
感受OIer的力量！（

没事，时间复杂度不大
O(|s| + log2n) # s 插入/删除字符数，n代码总行数
什么？log2n可以忽略？那可不行，这可是算法复杂度分析！（
-date 240725 注：应为O(|s| + n)，没事，n再大能超一百万不成？

如果翻一下Vim源码或许可以copy它的（但是我也看不懂啊）（

-date 240724 算了，明天再接着改吧

写注释最多的一次，大概我就是忘了这段代码也能读懂
（但是忘了逻辑就真读不懂了）

-date 240725 得了，这份代码报废了吧，去写一套类似Vim-syntax的分析系统吧
"""

from enum import Enum, unique
from keyword import kwlist as PY_KWS
from edbase import TextSpace
import copy
import bisect


@unique
class PyTokenType(Enum):
    Un = -1
    Kw = 0
    Id = 1
    Num = 2
    Op = 4
    Const = 5
    StrEsc = 12
    ConLine = 13
    ConStr = 14
    White = 15
    Err = 16

    # -lnk PyLexer.__init__.text
    # 光字符串就搞出12个状态（
    Str = 6  # '
    LongStr = 7
    StrBegin = 8
    LongStrBegin = 9
    StrEnd = 10
    LongStrEnd = 11
    Str1 = 17  # "
    LongStr1 = 18
    StrBegin1 = 19
    LongStrBegin1 = 20
    StrEnd1 = 21
    LongStrEnd1 = 22


PY_OPS = "+-*/%><=!~^&@:;()[]{},."


class PyLexer(TextSpace):
    def __init__(self, text: list[list[list[str]]], w):
        super().__init__(w)
        self.base = text
        # 解析时同时解析当前和翻转两种状态下的结果
        # 所以这里要有5种状态
        # -no/-longstr/-longstr1/-str/-str1
        self.text: list[
            list[
                list[
                    tuple[
                        PyTokenType,
                        PyTokenType,
                        PyTokenType,
                        PyTokenType,
                        PyTokenType,
                    ]
                ]
            ]
        ] = []
        self.status_cvts = []
        for i in self.base:
            self.text.append([])
            for j in i:
                self.text[-1].append([])
                for k in j:
                    self.text[-1][-1].append(
                        (
                            PyTokenType.Un,
                            PyTokenType.Un,
                            PyTokenType.Un,
                            PyTokenType.Un,
                            PyTokenType.Un,
                            0,
                        )
                    )

    def get_cur_base(self, y, x):
        if x < len(self.base[y[0]][y[1]]):
            return self.base[y[0]][y[1]][x]
        elif y[1] < len(self.base[y[0]]) - 1:
            return self.base[y[0]][y[1] + 1][0]
        elif y[0] < len(self.base) - 1:
            return "\n"
        else:
            return None

    # 你说得对，但是最复杂的还不是这个
    def identifier(self):
        s = ""
        begin = copy.deepcopy(self.y), self.x
        cur = self.get_cur_base(self.y, self.x)
        while cur and (cur.isalnum() or cur == "_"):
            self.y, self.x = self.dec_pos(self.y, self.x)
            s += cur
            cur = self.get_cur_base(self.y, self.x)
        if s[0].isdigit():
            if not s.isdecimal():
                tp = PyTokenType.Err
            else:
                tp = PyTokenType.Num
        elif s[0] in PY_KWS:
            tp = PyTokenType.Kw
        else:
            tp = PyTokenType.Id
        while self.cmp_2D(*begin, self.y, self.x) < 0:
            self.text[begin[0][0]][begin[0][1]][begin[1]] = (
                tp,
                PyTokenType.LongStr,
                PyTokenType.LongStr1,
                PyTokenType.Str,
                PyTokenType.Str1,
            )
            begin = self.inc_pos(*begin)

    def whitespace(self):
        cur = self.get_cur_base(self.y, self.x)
        while cur and cur in " \t":
            self.text[self.y[0]][self.y[1]][self.x] = (
                PyTokenType.White,
                PyTokenType.LongStr,
                PyTokenType.LongStr1,
                PyTokenType.Str,
                PyTokenType.Str1,
            )
            self.y, self.x = self.inc_pos(self.y, self.x)
            cur = self.get_cur_base(self.y, self.x)

    def quote(self):
        begin = copy.deepcopy(self.y), self.x
        cur = self.get_cur_base(self.y, self.x)
        q_tp = self.get_cur_base(self.y, self.x)
        q_cnt = 1
        while q_cnt < 3 and cur and cur == q_tp:
            q_cnt += 1
            self.y, self.x = self.inc_pos(self.y, self.x)
            cur = self.get_cur_base(self.y, self.x)
        if q_cnt == 1:
            if q_tp == '\'':
                self.text[begin[0][0]][begin[0][1]][begin[1]] = (
                    PyTokenType.StrBegin,
                    PyTokenType.LongStr,
                    PyTokenType.LongStr1,
                    PyTokenType.StrEnd,
                    PyTokenType.Str1,
                )
            else:
                self.text[begin[0][0]][begin[0][1]][begin[1]] = (
                    PyTokenType.StrBegin1,
                    PyTokenType.LongStr,
                    PyTokenType.LongStr1,
                    PyTokenType.Str,
                    PyTokenType.StrEnd1,
                )
        elif q_cnt == 2:
            ...
        else:
            ...

    def symbol(self):
        self.y, self.x = self.dec_pos(self.y, self.x)
        status = self.get_cur_char(self.y, self.x)[3]
        self.y, self.x = self.inc_pos(self.y, self.x)
        cur = self.get_cur_base(self.y, self.x)

    def change(self, do):
        if do[4] == "i":
            self.y, self.x = do[:2]
            ins = []
            for i in do[5]:
                if i == "\n":
                    ins.append(None)
                elif i == "\r":
                    pass
                else:
                    ins.append((PyTokenType.Un, PyTokenType.Un, PyTokenType.Un, 0))
            self.insert_any(ins)
            text = do[5]
            self.y, self.x = do[:2]

            # 先到上一个单词
            if text[0].isalnum() or text[0] == "_":
                old = copy.deepcopy(self.y), self.x
                self.y, self.x = self.dec_pos(self.y, self.x)
                cur = self.get_cur_base(self.y, self.x)
                while cur and (cur.isalnum() or cur == "_"):
                    old = copy.deepcopy(self.y), self.x
                    self.y, self.x = self.dec_pos(self.y, self.x)
                    if (self.y, self.x) == old:
                        break
                    cur = self.get_cur_base(self.y, self.x)
                if (self.y, self.x) != old:
                    self.y, self.x = self.inc_pos(self.y, self.x)
            else:
                old = copy.deepcopy(self.y), self.x
                self.y, self.x = self.dec_pos(self.y, self.x)
                cur = self.get_cur_base(self.y, self.x)
                while cur and not (cur.isalnum() or cur == "_" or cur.isspace()):
                    old = copy.deepcopy(self.y), self.x
                    self.y, self.x = self.dec_pos(self.y, self.x)
                    if (self.y, self.x) == old:
                        break
                    cur = self.get_cur_base(self.y, self.x)
                if (self.y, self.x) != old:
                    self.y, self.x = self.inc_pos(self.y, self.x)

        else:
            # 最复杂的是删掉一段代码！！！
            self.del_inrange(*do[:4])
