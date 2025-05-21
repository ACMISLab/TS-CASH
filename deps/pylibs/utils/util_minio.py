#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/17 16:27
# @Author  : gsunwu@163.com
# @File    : util_minio.py
# @Description:
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  minio==7.2.5
import os.path
from pylibs.utils.util_datetime import get_year_month
import sys
from dataclasses import dataclass

from minio import Minio

# 使用endpoint、access key和secret key来初始化minioClient对象。
# AK: M4PqZ5g1hZwqD0QHM5Bk
# SK: HHdSPQ7leEgwHOD7yyoQyQlJnupF2WE2YxakyeV7
# MINIO_ACCESS_KEY = "NHFpJPeqcw3efb5Mx4FJ"
# MINIO_SECRET_KEY = "1TgeCivhGOvhJSABryHfT0tstlxqU8Os1Nw661D3"
MINIO_ACCESS_KEY = "lPZgbitXE3XXKt35O6Au"
MINIO_SECRET_KEY = "PeIRkb4ANle22wXeW4fWoEHhaeS58B1iAC5XTngd"
MINIO_ENDPOINT = "your_server_ip:9000"


@dataclass
class UtilMinio:
    access_key: str = MINIO_ACCESS_KEY
    secret_key: str = MINIO_SECRET_KEY
    endpoint: str = MINIO_ENDPOINT
    bucket_name: str = "paper-exp-backup"

    def __post_init__(self):
        self.client = Minio(self.endpoint,
                            access_key=self.access_key,
                            secret_key=self.secret_key,
                            secure=False,
                            cert_check=False)
        print(self.client)

    def upload_file(self, local_file, remote_path=None):
        # Make the bucket if it doesn't exist.
        found = self.client.bucket_exists(self.bucket_name)
        if not found:
            self.client.make_bucket(self.bucket_name)
            print("Created bucket", self.bucket_name)
        else:
            print("Bucket", self.bucket_name, "already exists")

        if remote_path is None:
            remote_path = os.path.join(get_year_month(), os.path.basename(local_file))

        # Upload the file, renaming it in the process
        self.client.fput_object(
            self.bucket_name, remote_path, local_file,
        )
        print(
            local_file, "successfully uploaded as object",
            remote_path, "to bucket", self.bucket_name,
        )

    def upload_directory(self, local_path, remote_path=None):
        # Make the bucket if it doesn't exist.
        found = self.client.bucket_exists(self.bucket_name)
        if not found:
            self.client.make_bucket(self.bucket_name)
            print("Created bucket", self.bucket_name)
        else:
            print("Bucket", self.bucket_name, "already exists")

        if remote_path is None:
            remote_path = os.path.join(get_year_month(), os.path.basename(local_path))

        # Upload the file, renaming it in the process
        self.client.fput_object(
            self.bucket_name, remote_path, local_path,
        )
        print(
            local_path, "successfully uploaded as object",
            remote_path, "to bucket", self.bucket_name,
        )


if __name__ == '__main__':
    um = UtilMinio()
    # um.upload_file(sys.argv[0])
    um.upload_directory("./")
