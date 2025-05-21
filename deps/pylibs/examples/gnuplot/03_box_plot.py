#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/4/15 15:41
# @Author  : gsunwu@163.com
# @File    : 03_box_plot.py
# @Description:
from pylibs.utils.util_gnuplot import Gnuplot
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

# set boxwidth  0.5
set pointsize 1
unset key 
# 只保留坐标的刻度小，上右下的刻度线取消
set border 2
# 这只x=1处显示A，x=2处显示B
set xtics ("A" 1, "B" 2) scale 0.0

# 隐藏x和y轴上的刻度
set xtics nomirror
set ytics nomirror

plot $df using (1):($1*5) fc "#f8f8f8", "" using (2):($1*10) fc "#ff0000"
""")
gp.write_to_file()
gp.show()