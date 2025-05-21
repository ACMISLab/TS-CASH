#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/4/1 16:13
# @Author  : gsunwu@163.com
# @File    : test_ssh.py
# @Description:

import unittest
import logging

from pylibs.utils.util_servers import Servers

log = logging.getLogger(__name__)


class TestDatasetLoader(unittest.TestCase):

    def test_demo(self):
        server=Servers.S164
        assert server.is_port_listing(6006)==True
        assert server.is_port_listing(6007)==False
    def test_upload(self):
        server=Servers.S164
        server.upload_file_by_ssh(__file__,"/tmp/abc.py")