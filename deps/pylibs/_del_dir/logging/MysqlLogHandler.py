# coding: utf-8
import logging
import pandas as pd

from pylibs.util_mysql import get_mysql_connection


class MysqlLogHandler(logging.Handler):

    def __init__(self, uri="mysql+pymysql://root:your_password@your_server_ip:9201/nni_experiments?charset=utf8"):
        self.table = "app_logs"
        self.db = get_mysql_connection()
        logging.Handler.__init__(self)

    def emit(self, record):
        target = {}
        for key, val in record.__dict__.items():
            target[key] = str(val)
        df = pd.DataFrame([list(target.values())], columns=list(target.keys()))
        df.to_sql(self.table, self.db, if_exists="append", index=True)
