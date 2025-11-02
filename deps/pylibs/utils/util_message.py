from pylibs.common import Emjoi
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_system import UtilSys
log = get_logger()


def log_setting_msg(msg):
    UtilSys.is_debug_mode() and log.info(f"{Emjoi.SETTING} {msg}")


def debug_msg(msg):
    UtilSys.is_debug_mode() and log.info(f"{Emjoi.INFO} {msg}")


def log_metric_msg(msg):
    UtilSys.is_debug_mode() and log.info(f"{Emjoi.METRIC} {msg}")


def log_warning_msg(msg):
    log.warning(f"{Emjoi.WARNING} {msg}")


def log_error_msg(msg):
    log.error(f"{Emjoi.ERROR} {msg}")


def log_warn_msg(msg):
    log_warning_msg(msg)


def log_start_msg(msg):
    UtilSys.is_debug_mode() and log.info(f"{Emjoi.START} {msg}")


def log_finished_msg(msg):
    UtilSys.is_debug_mode() and log.info(f"{Emjoi.FINISHED} {msg}")


def log_success_msg(msg):
    log_finished_msg(msg)


def logi(msg):
    UtilSys.is_debug_mode() and log.info_msg(msg)


def loge(msg):
    log_error_msg(msg)


def logw(msg):
    log_warning_msg(msg)


def logs(msg):
    log_setting_msg(msg)
