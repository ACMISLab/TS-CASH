

import os.path
import re
import sys

import numpy as np
import pandas as pd

from pylibs.exp_ana_common.ExcelOutKeys import PaperFormat, EK
from pylibs.utils.util_common import UC
from pylibs.utils.util_log import get_logger

log = get_logger()


r"""
语法:

多列:
\multicolumn{4}{|c|}{Country List}

多行:
\usepackage{multirow}
\multirow{3}{4em}{Multiple row}
\multirow{3}{*}{Multiple row}

表格线:
\cline{1-2}
"""


def _save_text_to_file(filename, latex_txt, home):
    filename = os.path.join(home, filename)
    if not filename.endswith(".tex"):
        filename = filename + ".tex"
    log.info(f"saving latex to file: {os.path.abspath(filename)}")
    with open(filename, "w") as f:
        f.write(latex_txt)


class LatexClass:

    def __init__(self, latex: str):
        self.ori_latex = latex
        self.body = None
        self.header = None
        self.parse_latex_txt()

    def parse_latex_txt(self):
        """
        将一个 latex 表格分成如下几个部分
        \begin{table} [h!]

        \caption{Average accuracy and speedup under different $\alpha$.}
        \label{tab:effect_of_stop_alpha}
        \begin{tabular}{ccccc}
        oprule
        \multirow{2}{*}{$\alpha$} & \multicolumn{2}{|c}{VUS ROC} &  \multicolumn{2}{|c}{VUS PR} \\
        & \multicolumn{1}{|c} {Value}& Speedup & \multicolumn{1}{|c} {Value} & Speedup \\
        \midrule
        extbf{0.001} & extbf{44.16}(±13.75) & 3.60 & extbf{78.63}(±7.26) & 3.60 \\
        0.01 & 43.71(±13.68) & 6.13 & 78.23(±7.38) & 6.13 \\
        0.1 & 42.75(±13.61) & 16.92 & 77.51(±7.52) & 16.92 \\
        0.5 & 42.38(±13.55) & 28.86 & 77.25(±7.59) & 28.86 \\
        \bottomrule
        \end{tabular}
        \end{table}[h!]


        :return:
        :rtype:
        """
        self.header = self.ori_latex[:self.ori_latex.find(r"oprule")]
        pass


class LatexTable:

    def __init__(self, df, caption=None, label=None, float_format="%.2f",
                 home=PaperFormat.HOME_LATEX_HOME_IET):
        self.df = df
        self.caption = caption
        self.label = label
        self.float_format = float_format
        self.home = home

        if isinstance(df, pd.DataFrame):
            # 显示所有列
            pd.set_option('display.max_columns', None)
            # 显示所有行
            pd.set_option('display.max_rows', None)
            # 设置value的显示长度为100，默认为50
            pd.set_option('max_colwidth', 100)
            self.latex_txt = self.df.to_latex(index=False, label=self.label, caption=self.caption,
                                              float_format=self.float_format, escape=False)
            self.latex_txt = self.latex_txt.replace("\\n", "\n")
        else:
            self.latex_txt = df
        self.funcs = []

    def to_latex(self, filename=None):
        for _fun in self.funcs:
            self.latex_txt = _fun(self.latex_txt)

        if filename is None:
            return self.latex_txt
        else:
            self._save_to_file(filename, self.latex_txt)

    def _save_to_file(self, filename, latex_txt):
        _save_text_to_file(filename, latex_txt, self.home)
        # filename = os.path.join(self.home, filename)
        # if not filename.endswith(".tex"):
        #     filename = filename + ".tex"
        # log.info(f"saving latex to file: {os.path.abspath(filename)}")
        # with open(filename, "w") as f:
        #     f.write(latex_txt)

    def format_model_and_dataset(self):
        # 格式化模型名称
        for _model_name_ori in PaperFormat.model_name_maps.keys():
            self.latex_txt = self.latex_txt.replace(_model_name_ori, PaperFormat.model_name_maps.get(_model_name_ori))
        self.latex_txt = self.latex_txt.replace(EK.MODEL_NAME, "Methods")

    def wide_table(self):
        self.latex_txt = self.latex_txt.replace("table}", "table*}")

    def enable_ht(self):
        self.latex_txt = self.latex_txt.replace(r"\begin{table}", r"\begin{table}[h!]")
        self.latex_txt = self.latex_txt.replace(r"\begin{table*}", r"\begin{table*}[h!]")
        return self

    def set_ht(self):
        self.enable_ht()
        return self

    def set_h(self):
        self.latex_txt = self.latex_txt.replace(r"\begin{table}", r"\begin{table}[h]")
        self.latex_txt = self.latex_txt.replace(r"\begin{table*}", r"\begin{table*}[h]")
        return self

    def replace(self, pattern, repl):
        self.latex_txt = self.latex_txt.replace(pattern, repl)
        return self

    def set_header(self, header: str):
        """

        :param header:
        :type header:
        :return:
        :rtype:
        """
        # re.sub(pattern, repl, string, count=0, flags=0) 函数参数讲解：repl 替换掉 string 中被 pattern 匹配的字符，count 表示最大替换次数，flags 表示正则表达式的常量。

        header_start = self.latex_txt.find("\oprule")
        header_end = self.latex_txt.find("\\midrule")

        assert header_start>-1, "toprule not found"
        assert header_end > -1, "midrule rule not found"
        self.latex_txt=self.latex_txt[:header_start]+"\oprule"+header+self.latex_txt[header_end:]

        return self

    def bold_text(self, param):
        """
        Bold the specified param

        Parameters
        ----------
        param :

        Returns
        -------

        """
        self.latex_txt = self.latex_txt.replace(param, f"\extbf{{{param}}}")

    def highlight_text_less_than(self, significant_level=0.01, decimals=2):
        for _v in _get_highlight_values(self.df, significant_level, decimals):
            _v = float(_v)
            self.bold_text(f"{_v:.{decimals}f}")

    def replace_measure_name(self):
        self.replace("VUS_ROC", "VUS ROC")
        self.replace("VUS_PR", "VUS PR")
        self.replace("VUS_PR", "VUS PR")


def _get_highlight_values(_data, significant_level=0.01, decimals=3):
    _t = _data.iloc[:, 1:].values.reshape(-1)
    _t = np.round(_t, decimals)
    return _t[np.argwhere(_t < significant_level)]


class LatexSubTable(LatexTable):
    """
    \begin{tabular}[t]{@{} l c *{2}{d{1.7}} @{}}
    oprule
     Band &    L & \mc{$\mathcal{O}_{JK}^2$} & \mc{$\mathcal{O}_{EP}^2$} \\
    \midrule
       1 &  2.5 &      0.97230(3) &      0.9866(6) \\
    \midrule
         &  1.5 &      0.96685(6) &      0.980(3)  \\
         &  2.5 &      0.92390(9) &      0.956(4)  \\
    \bottomrule
    \end{tabular}
    """

    def __init__(self, df, caption, label):
        super().__init__(df, caption, label)

    def get_sub_content(self):
        # header_start = self.latex_txt.find("\\begin{tabular}")
        # header_end = self.latex_txt.find("\\end{tabular}")
        # print(self.latex_txt)
        # sys.exit()
        # return self.latex_txt[header_start:header_end + len("\end{tabular}")]
        self.latex_txt = self.latex_txt.replace("begin{table}", "begin{subtable}[t]{\extwidth}")
        self.latex_txt = self.latex_txt.replace("end{table}", "end{subtable}")
        return self.latex_txt


class MultLatexTable:
    """
    \begin{table}
    \begin{subtable}[t]{0.48extwidth}
    """

    def __init__(self, caption, label, home=PaperFormat.HOME_LATEX_HOME_IET):
        self._ht = ""
        self.table_begin = r"\begin{table}"
        self.table_end = r"\end{table}"
        self._vspace = r"\vspace{10pt}"
        self.caption = caption
        self.label = label
        self.home = home
        self.tables = []

    def append_subtable(self, table: LatexSubTable):
        self.tables.append(table.get_sub_content())

    def enable_ht(self):
        self._ht = "ht!"

    def wide_table(self):
        self.table_begin = r"\begin{table*}"
        self.table_end = r"\end{table*}"

    def normal_table(self):
        self.table_begin = r"\begin{table}"
        self.table_end = r"\end{table}"

    def to_latex(self, filename: str = UC.get_filename_entry()):
        texts = []
        texts.append(self.table_begin)
        texts.append(f"\\caption{{{self.caption}}}")
        texts.append(f"\\label{{{self.label}}}")

        for _tex in self.tables:
            # texts.append("\\begin{subtable}[t]{\extwidth}")
            texts.append(self._vspace)
            texts.append(_tex)
            # texts.append("\\end{subtable}")
        texts.append(self.table_end)
        text = "\n".join(texts)

        _save_text_to_file(filename, text, self.home)


if __name__ == '__main__':
    #     lc = LatexClass(r"""
    #     \begin{table}[h!]
    # \caption{Average accuracy and speedup under different $\alpha$.}
    # \label{tab:effect_of_stop_alpha}
    # \begin{tabular}{ccccc}
    # oprule
    # \multirow{2}{*}{$\alpha$} & \multicolumn{2}{|c}{VUS ROC} &  \multicolumn{2}{|c}{VUS PR} \\
    # & \multicolumn{1}{|c} {Value}& Speedup & \multicolumn{1}{|c} {Value} & Speedup \\
    # \midrule
    # extbf{0.001} & extbf{44.16}(±13.75) & 3.60 & extbf{78.63}(±7.26) & 3.60 \\
    # 0.01 & 43.71(±13.68) & 6.13 & 78.23(±7.38) & 6.13 \\
    # 0.1 & 42.75(±13.61) & 16.92 & 77.51(±7.52) & 16.92 \\
    # 0.5 & 42.38(±13.55) & 28.86 & 77.25(±7.59) & 28.86 \\
    # \bottomrule
    # \end{tabular}
    # \end{table}[h!]
    #
    #     """)
    pass
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [1, 2, 3]
    })
    lt = LatexSubTable(df, caption="aa", label="sdf")
    lt.get_sub_content()

    mul = MultLatexTable(caption="a0", label="sdf")
    mul.append_subtable(lt)
    mul.append_subtable(lt)
    print(mul.to_latex())
