#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/4/28 12:31
# @Author  : gsunwu@163.com
# @File    : util_progress.py
# @Description:
import time
import sys


def progress_bar(per: float, msg: str = "", total_length: int = 20):
    """
    per: 当前进度百分比,取值范围0-100
    total_length: 进度条总长度,默认20个字符
    """

    # 计算已完成的进度条长度
    bar_length = int(per / 100 * total_length)

    # 生成进度条字符串
    bar = '[' + '=' * bar_length + ' ' * (total_length - bar_length - 1) + ']'

    # 计算进度百分比
    percent = f'{per:.0f}%'

    # 将进度条和百分比拼接起来
    progress_str = f'\r {msg}  {bar} {percent}'

    # 使用 \r 和 flush 在单行动态刷新进度条
    sys.stdout.write(progress_str)
    sys.stdout.flush()

    # 进度100%后换行
    if per == 100:
        sys.stdout.write('\n')


if __name__ == '__main__':
    # 测试
    for i in range(101):
        progress_bar(i)
        time.sleep(0.05)
