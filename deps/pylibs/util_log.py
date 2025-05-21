import logging
import sys

log = None
# 确保log只初始化一次
if log is None:
    print("Init logging ...")
    log = logging.getLogger("main")
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(logging.Formatter(
        '%(filename)s:%(lineno)d %(levelname)s - %(message)s'))
    log.addHandler(stream_handler)
    log.setLevel(logging.INFO)


def get_log():
    return log


def get_logger():
    return get_log()


def get_logs():
    return get_log()


def getlogs():
    return get_log()


def getlog():
    return get_log()


def info(msg):
    log.info(msg)


def debug(msg):
    log.debug(msg)
