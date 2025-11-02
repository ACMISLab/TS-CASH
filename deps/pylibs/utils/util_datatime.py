import datetime


class DateTimeFormat:
    # 年份_今年第几天 https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
    # 如2023年的第40天表示为:  2023_040
    YEAR_DAY_INDEX = "%Y_%j"

    # 4位年_2位月_2位日, 如2023_01_12
    YEAR_MONTH_DAY = "%Y_%m_%d"

    # 4位年_2位月_2位日_2位时_2位分钟_2位秒,
    # 如2023年1月12日12点10分11秒: 如 2023_01_12_12_10_11
    YEAR_MONTH_DAY_HOUR_MIN_SECOND = "%Y_%m_%d_%H_%M_%S"
    # 20230112_1210
    YEAR_MONTH_DAY_HOUR_MIN = "%Y%m%d_%H%M"

    # 4位年-2位月-2位日 2位时:2位分钟:2位秒,
    # 如2023年1月12日12点10分11秒: 如 2023-01-12 12:10:11
    YEAR_MONTH_DAY_HOUR_MIN_SECOND_NORMAL = "%Y-%m-%d %H:%M:%S"


def get_datatime():
    return datetime.datetime.now().strftime(DateTimeFormat.YEAR_MONTH_DAY_HOUR_MIN_SECOND_NORMAL)


def get_str_datetime(format: str = DateTimeFormat.YEAR_MONTH_DAY_HOUR_MIN):
    return datetime.datetime.now().strftime(format)
