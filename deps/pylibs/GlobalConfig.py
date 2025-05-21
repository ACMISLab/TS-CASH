#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/7/1 11:29
# @Author  : gsunwu@163.com
# @File    : GlobalConfig.py
# @Description: 全局配置类，配置文件保存在 ~/configs.yaml
import os

import yaml


class GlobalConfig:

    @staticmethod
    def get_perplexity_api_keys():
        with open(os.path.join(os.path.expanduser('~') ,'configs.yaml'), 'r') as file:
            config = yaml.safe_load(file)
        api_keys = config['perplexity']['apiKeys']
        return api_keys

if __name__ == '__main__':
    print(GlobalConfig.get_perplexity_api_keys())
