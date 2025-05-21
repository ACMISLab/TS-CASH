#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/14 09:10
# @Author  : gsunwu@163.com
# @File    : util_open_qu.py
# @Description: 日志存储，log saving
import json
import logging
import os
import platform
import sys
import traceback

import distributed
import requests
from requests.auth import HTTPBasicAuth
from dataclasses import dataclass

from pylibs.config import Debug


# def get_logger():
#
#
# logging.basicConfig(
#     stream=sys.stdout,
#     level=logging.DEBUG
# )
# log = logging.getLogger(__name__)
# if sys.platform != "darwin":
#     log_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs.log"))
#     print(f"Log file: {log_file}")
#     file_handler = logging.FileHandler(log_file, 'a', encoding='utf-8')
#     file_handler.setLevel(logging.DEBUG)
#     log.addHandler(file_handler)


class ZincSearch:
    @staticmethod
    def get_logger(name=None):
        #   # curl -u sunwu:your_password. -X POST "http://your_server_ip:13291/api/aa/_doc" -H "Content-Type: application/json" -d '{ "title": "23Sharma","content":"sdfslxxjkljldsf" }'
        if name is None:
            name = "default"

        return HTTPLogs(
            host=f"http://your_server_ip:13291/api/{name}/_doc",
            username="sunwu",
            password="your_server_password",
            log_type="zincsearch"
        )


class OpenObserve:
    @staticmethod
    def get_logger(name=None, level=logging.INFO, console=False):
        # curl -u 1228746736@qq.com:W6XUEAEw46SEfJRv -k http://your_server_ip:13292/api/default/default/_json -d '[{"level":"info","job":"test","log":"test message for openobserve"}]'
        if name is None:
            name = "default"

        # http://your_server_ip:13292
        # "http://your_server_ip:14001/api/default/logging/_json"
        return HTTPLogs(

            host=f"http://your_server_ip:13292/api/default/{name}/_json",
            username="1228746736@qq.com",
            password="your_server_password",
            log_type="openobserve",
            level=level
        )


class zs(ZincSearch):
    pass


@dataclass
class HTTPLogs:
    # host: str = "http://your_server_ip:14001"
    host: str = "http://your_server_ip:13292"
    # url = "http://your_server_ip:14001/api/default/logging/_json"
    username: str = "1228746736@qq.com"
    password: str = "SKFufp5hNAZUvOuj"
    os_name = platform.node()
    name: str = "logging"
    level: int = logging.INFO
    log_type: str = None
    auth = None

    #  logging level
    #  0 SHOW ALL
    #  10 DEBUG
    #  20 INFO
    #  30 WARNING
    #  40 ERROR
    #  50 CRITICAL
    def __post_init__(self):
        if self.auth is None:
            self.auth = HTTPBasicAuth(self.username, self.password)

    def _send_msg(self, data: str, level="INFO"):
        """


        Parameters
        ----------
        data : str
            Json string, e.g.: [{"level":"INFO","msg":"JLKSJDFLJLSDF","identifier":"util_vector.py:85"}]

        Returns
        -------

        """
        # 设置请求的URL和认证信息

        # 准备请求体数据

        # 发送POST请求
        try:
            data = self.process_data(data)
            stack = traceback.format_stack()
            _t = {
                'level': level,
                "msg": data,
                "node": HTTPLogs.os_name,
                "loc": stack[-3],

            }
            json_data = json.dumps(_t)
            response = requests.post(self.host,
                                     data=json_data,
                                     auth=self.auth,
                                     verify=False,
                                     timeout=1)
            # 输出响应内容
            if response.status_code == 200:
                # distributed.print(data)
                # print(data)
                return
            else:
                print(data)
                Debug.errmsg(f"❌❌❌ Logging message failed since {response.status_code}:{response.text}")
            return response.text
        except Exception as e:
            Debug.errmsg(f"❌❌❌ Failed to dump data {data} since {e}")

    def info(self, data: str):
        if self.level <= logging.INFO:
            return self._send_msg(data, level="INFO")
        else:
            print("info", data)

    def conf(self, data: str):
        if self.level <= logging.INFO:
            return self._send_msg("🔧🔧🔧 " + data, level="INFO")

    def warning(self, data: str):
        if self.level <= logging.WARN:
            return self._send_msg(data, level="WARNING")

    def error(self, data: str):
        if self.level <= logging.ERROR:
            return self._send_msg(data, level="ERROR")

    def debug(self, data: str):
        if self.level <= logging.DEBUG:
            return self._send_msg(data, level="DEBUG")

    @staticmethod
    def get_logger(name=""):
        return HTTPLogs(name=name)

    def process_data(self, data):
        return str(data)


"""
mkdir -p /docker/appdata/openobserve/data
chmod 777 /docker/appdata/openobserve/data
docker rm -f openobserve
docker run -d --restart always \
      --name openobserve \
      -v /docker/appdata/openobserve/data:/data \
      -p 13292:5080 \
      -e RUST_BACKTRACE=1 \
      -e ZO_ROOT_USER_EMAIL="1228746736@qq.com" \
      -e ZO_ROOT_USER_password="your_server_password" \
      public.ecr.aws/zinclabs/openobserve:latest
"""


class oo(OpenObserve):
    pass


if __name__ == '__main__':
    for i in range(10):
        log = oo.get_logger(11)
        log.error(f"new search {i}")
