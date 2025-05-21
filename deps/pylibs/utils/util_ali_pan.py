#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/10 14:53
# @Author  : gsunwu@163.com
# @File    : util_ali_pan.py
# @Description:
# !/root/miniconda3/bin/python
# _*_ coding: utf-8 _*_
# @Time    : 2023/12/2 18:22
# @Author  : gsunwu@163.com
# @File    : gitlab_to_ali.py
# @Description:
import traceback
from typing import List

from aligo import Aligo

PROGRAM_DESCRIPTION = """
上传多个文件到阿里网盘：  python ali_up.py -f a.txt b.txt /home/c.txt

"""


class AliPanUtil:
    @staticmethod
    def upload_files(local_dir: List[str] | str, target_dir: str = "auto_backup/backup"):
        """

        Parameters
        ----------
        local_dir :  本地路径
        target_dir :  阿里网盘路径

        Returns
        -------

        """
        try:
            if type(local_dir) == str:
                local_dir = [local_dir]
            ali = Aligo()  # 第一次使用，会弹出二维码，供扫描登录

            user = ali.get_user()  # 获取用户信息
            print("用户信息", user.user_name, user.nick_name, user.phone)  # 打印用户信息
            print(f"从 [{local_dir}] 备份到阿里云网盘的 [{target_dir}] ")
            target_dir_obj = ali.get_folder_by_path(target_dir)
            if target_dir_obj is None:
                print(f"找不到阿里云盘的文件夹: {target_dir}")
            else:
                # res_list = ali.upload_folder(local_dir,
                #                              parent_file_id=target_dir_obj.file_id,
                #                              check_name_mode="overwrite")
                res_list = ali.upload_files(local_dir,
                                            parent_file_id=target_dir_obj.file_id,
                                            check_name_mode="auto_rename")
                if len(res_list[0].file_id) == 40:
                    print("✅ 备份成功 ", res_list)
                else:
                    print("❌ 备份失败！", res_list)
                return res_list
        except Exception as e:
            traceback.print_exc()


if __name__ == '__main__':
    """
    python ali_up.py -f /Users/sunwu/Nextcloud/App/tailscale/tailscale_1.60.0_amd64/tailscaled /Users/sunwu/Nextcloud/App/tailscale/tailscale_1.60.0_amd64/tailscale
    """
    import argparse

    parser = argparse.ArgumentParser(description=PROGRAM_DESCRIPTION)
    parser.add_argument("-f", "--file", type=str, required=True, nargs='+', help="The file(s) to upload.")
    parser.add_argument("--target", type=str, help="The ali directory.", default="auto_backup/backup")
    args = parser.parse_args()
    AliPanUtil.upload_files(args.file, args.target)
