# 创建logger对象
import logging
import os.path
import sys
from pylibs.utils.util_system import UtilSys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ConstHandler:
    FILE_HANDLER = "FileHandler"
    STREAM_HANDLER = "FileHandler"
    MYSQL_HANDLER = "FileHandler"


class LoggingHelper:
    _console_logger = None

    @classmethod
    def get_console_log(cls, level=logging.WARNING):
        if cls._console_logger is None:
            cls._init(level)
        return cls._console_logger

    @classmethod
    def _init(cls, level):

        if cls._console_logger is not None and cls._console_logger.hasHandlers():
            cls._remove_all_handlers()

        # init
        cls._console_logger = logging.getLogger("util_log")
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.set_name(ConstHandler.STREAM_HANDLER)
        stream_handler.setFormatter(logging.Formatter('%(filename)s:%(lineno)d %(levelname)s - %(message)s'))
        cls._console_logger.addHandler(stream_handler)
        cls._console_logger.setLevel(level)

    @classmethod
    def _remove_all_handlers(cls):
        for handler in cls._console_logger.handlers:
            cls._console_logger.removeHandler(handler)


def get_logger(level=None):
    """
    如何设置日志级别？
    第一次调用是，给定 level 即可。后续调用时，会自动设置 level

    可通过 export LOG_LEVEL=ERROR 设置，这种设置方法优先级最高

    home_path = os.environ.get('HOME')
    """
    if UtilSys.is_debug_mode():
        log = LoggingHelper.get_console_log(logging.INFO)
    else:
        log = LoggingHelper.get_console_log(logging.DEBUG)
    return log


log = get_logger()


def logconf(msg):
    log.info(f"🔧 {msg}")


def logwarn(msg):
    log.info(f"⚠️ {msg}")


def loginfo(msg):
    log.info(f"{msg}")


if __name__ == '__main__':
    log = get_logger(level=logging.DEBUG)
    print("Debug 模式下应该输出3类信息：")
    UtilSys.is_debug_mode() and log.info("Debug message")
    UtilSys.is_debug_mode() and log.info("Info message")
    log.warning("Warning message")

    print("--------------------------------")
    log = get_logger(level=logging.WARNING)
    print("Waring 模式下应该输出1类信息：")
    UtilSys.is_debug_mode() and log.info("Debug message")
    UtilSys.is_debug_mode() and log.info("Info message")
    log.warning("Warning message")
    log = get_logger(level=logging.INFO)
