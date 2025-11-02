import subprocess
import time
import unittest

from pylibs.utils.util_gnuplot import Gnuplot
from pylibs.utils.util_log import get_logger

log = get_logger()


class TestDatasetLoader(unittest.TestCase):

    def test_demo(self):
        # 实例化一个gnuplot
        gnuplot = subprocess.Popen(['gnuplot', '-p'], shell=True, stdin=subprocess.PIPE)

        # 获取输入终端, 并输入会话命令
        gnuplot.stdin.write("plot sin(x),cos(x)\n".encode("utf-8"))
        gnuplot.stdin.flush()

        time.sleep(3)
        # 退出程序
        gnuplot.stdin.write("quit\n".encode("utf-8"))  # close the gnuplot window
        gnuplot.stdin.close()

        # 等待进程结束
        gnuplot.wait()
    def test_multi_line(self):
        gp=Gnuplot()
        gp.sets("""
$df <<EOF
1
2
3
4
1.1
-1
6
EOF
set style fill solid 0.5 border -1
set style data boxplot 
set style boxplot outliers pointtype 4
plot $df using (1):($1*5) fc "#f8f8f8", "" using (2):($1*10) fc "#ff0000"
        """)
        gp.write_to_file()
        gp.show()