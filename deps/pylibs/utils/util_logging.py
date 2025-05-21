#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/4/21 07:29
# @Author  : gsunwu@163.com
# @File    : util_logging.py
# @Description:
import os.path
import logging
import traceback


class UtilLogger:
    logger = None

    @staticmethod
    def get_logger(level=logging.INFO):
        if UtilLogger.logger is None:
            UtilLogger.logger = _get_logger(level)
        return UtilLogger.logger


def _get_logger(level):
    logger = logging.getLogger('test_logger')
    # 设置日志等级
    logger.setLevel(level)

    # 追加写入文件a ，设置utf-8编码防止中文写入乱码
    traceback.extract_stack()
    test_log = logging.FileHandler(os.path.join(os.path.dirname(__file__), "debug.log"), 'a', encoding='utf-8')

    # 向文件输出的日志级别
    test_log.setLevel(logging.DEBUG)

    # 向文件输出的日志信息格式
    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s -%(process)s')

    test_log.setFormatter(formatter)

    # 加载文件到logger对象中
    logger.addHandler(test_log)

    return logger
