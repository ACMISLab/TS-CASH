#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/8/7 17:08
# @Author  : gsunwu@163.com
# @File    : util_pickle.py
# @Description:
import pickle

class PickleHandler:
    """
    包含两个函数：
    1. 将python 中的任何对象dump pickle 到文件
    2. load 之前dump 的文件
    """
    def __init__(self, filename):
        self.filename = filename

    def dump(self, obj):
        """将对象序列化并保存到文件"""
        with open(self.filename, 'wb') as file:
            pickle.dump(obj, file)
        print(f"对象已成功保存到 {self.filename}")

    def load(self):
        """从文件加载之前序列化的对象"""
        with open(self.filename, 'rb') as file:
            obj = pickle.load(file)
        print(f"对象已成功加载自 {self.filename}")
        return obj

# 示例用法
if __name__ == "__main__":
    handler = PickleHandler('data.pkl')

    # 示例对象
    data = {'key': 'value', 'number': 42}

    # 保存对象
    handler.dump(data)

    # 加载对象
    loaded_data = handler.load()
    print(loaded_data)
