#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/10 15:36
# @Author  : gsunwu@163.com
# @File    : util_qiniu.py
# @Description:
import logging
import os
import sys
from dataclasses import dataclass
from qiniu import Auth, put_file
from pylibs.utils.util_datetime import get_year_month, get_year

log = logging.getLogger(__name__)


@dataclass
class QiniuUtil:
    # 七牛云账户的Access Key和Secret Key
    # https://img.lessonplan.cn/IMG/HOME/tests/Snipaste_2024-03-10_16-02-01.jpg
    access_key: str = 'BZVp9geQcbz18jqHVimxCcN4qWKU9-KiKlq9F1wn'
    secret_key: str = 'wZGA1mYfgmLH1lay9f7rQRu2MutPvFcwa2MWJbMJ'
    home: str = "IMG/HOME"
    server_host: str = 'https//img.lessonplan.cn'
    IMAGE_EXT = ['png', 'jpg', 'jpeg', '.pdf']
    bucket_name: str = 'lessonplan-img'

    # IMG/HOME/{word2lessonplan}/{user_name}/{uuid}
    target_folder: str = 'IMG/HOME/tests'

    def _get_token(self, remote_path):
        q = Auth(self.access_key, self.secret_key)
        return q.upload_token(self.bucket_name, remote_path)

    # 初始化认证对象
    def upload(self, local_folder='/sunwu/aaa'):
        """
        上传图片
        https:<self.server_host>/<self.target_folder><image_name>
        e.g.:
        https://img.lessonplan.cn/IMG/HOME/tests/Snipaste_2024-03-10_16-02-01.jpg

        Parameters
        ----------
        local_folder :

        Returns
        -------

        """

        # q = Auth(self.access_key, self.secret_key)
        ret_arr = []
        q = Auth(self.access_key, self.secret_key)
        # 获取文件列表
        for root, dirs, files in os.walk(local_folder):
            for file in files:
                local_file_path = os.path.join(root, file)
                # 相对路径，用于创建在七牛云中相同的文件结构
                relative_path = os.path.relpath(local_file_path, local_folder)
                cloud_path = os.path.join(self.target_folder, relative_path)
                # 生成上传 Token
                # cloud_path: 'IMG/HOME/tests/util_class.py'
                token = q.upload_token(self.bucket_name, cloud_path)

                # 上传文件
                if local_file_path.split(".")[-1] in self.IMAGE_EXT:
                    # cloud_path='IMG/HOME/tests/simple331.png'
                    # local_file_path='./simple331.png'
                    ret, info = put_file(token, cloud_path, local_file_path)
                    if ret is not None:
                        log.info(f"上传成功: {file} 到 {cloud_path}")
                        ret_arr.append(cloud_path)
                    else:
                        log.info(f"上传失败: {file}")

        # 注意：请根据你的需要调整脚本，比如处理上传结果和异常等。
        return ret_arr

    def upload_file(self, local_file, remote_file=None):
        # 相对路径，用于创建在七牛云中相同的文件结构

        # 生成上传 Token

        if remote_file is None:
            remote_file = os.path.join("tshpo", os.path.basename(local_file))
        token = self._get_token(remote_file)
        # 上传文件
        # cloud_path='IMG/HOME/tests/simple331.png'

        # local_file_path='./simple331.png'
        # put_file(token, cloud_path, local_file_path)
        # qiniu-py-upload-gwusun-top-idvoll1.qiniudns.com
        # qiniu-py-upload.gwusun.top
        # cname: qiniu-py-upload-gwusun-top-idvoll1.qiniudns.com
        # http://qiniu-py-upload.gwusun.top/2024-08/output.xlsx&token=JU6C1VVuXfqe3zSmkc8uGCQoIyGhMpcm7omuYXHk:NgH9EqPyQgFzbh72cFsSnWqbrvc=:eyJzY29wZSI6InB5LXVwbG9hZHM6MjAyNC0wOC9vdXRwdXQueGxzeCIsImRlYWRsaW5lIjoxNzIzNzc4OTEwfQ==
        ret, info = put_file(token, remote_file, local_file)

        q = Auth(self.access_key, self.secret_key)
        original_download_url = f"{self.server_host}/{remote_file}"
        private_url = q.private_download_url(original_download_url, 3600)
        if info.status_code == 200:
            print(f"✅✅✅ File {local_file} upload successfully to qiniu: {remote_file},download_url: {private_url}")
        else:
            print("❌❌❌ File upload failed to qiniu", ret, info)


@dataclass
class MyQiniu:
    access_key: str = "JU6C1VVuXfqe3zSmkc8uGCQoIyGhMpcm7omuYXHk"
    secret_key: str = "VvlczRnTkZ58sfTCw9KEN6qqblMXDSAnYMEi4VJr"
    bucket_name: str = "py-uploads"
    server_host: str = 'http://qiniu-py-upload.gwusun.top'

    def __post_init__(self):
        self.q = QiniuUtil(
            access_key=self.access_key,
            secret_key=self.secret_key,
            bucket_name=self.bucket_name,
            server_host=self.server_host
        )

    def upload_file(self, local_file, remote_file=None):
        self.q.upload_file(local_file=local_file, remote_file=remote_file)


if __name__ == '__main__':
    q = MyQiniu()
    q.upload_file(sys.argv[0])
