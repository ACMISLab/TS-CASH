#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/10 14:03
# @Author  : gsunwu@163.com
# @File    : util_uuid.py
# @Description:
import uuid
class UtilUUID:
    @staticmethod
    def get_uuid():

        return str(uuid.uuid4())
if __name__ == '__main__':
    print(UtilUUID.get_uuid())