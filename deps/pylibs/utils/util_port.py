#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/15 22:31
# @Author  : gsunwu@163.com
# @File    : util_port.py
# @Description:
import socket
import subprocess
import time

from pylibs.utils.util_network import is_port_listing


def is_port_in_use(port):
    return is_port_listing(port)
    # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #     try:
    #         s.bind(('localhost', port))
    #         return False
    #     except:
    #         return True


def wait_kill_process_on_port(port=6006):
    kill_process_on_port(port)
    while is_port_in_use(port):
        print(f"Waiting for kill process on port={port}...")
        time.sleep(0.1)


def wait_port_listing(port=6006):
    while not is_port_in_use(port):
        print(f"Waiting for listing port={port}...")
        time.sleep(0.1)


def kill_process_on_port(port):
    # 使用lsof命令找到占用指定端口的进程PID
    try:
        result = subprocess.run(['lsof', '-ti', f':{port}'], capture_output=True, text=True)
        pids = result.stdout.strip().split('\n')
        for pid in pids:
            if pid:
                # print(f"Killing process {pid} on port {port}")
                # 使用kill命令结束进程
                subprocess.run(['kill', '-9', pid])
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    kill_process_on_port(6006)
