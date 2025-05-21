#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/4/3 08:32
# @Author  : gsunwu@163.com
# @File    : util_ssh_tunnel.py
# @Description:
import time

#
from sshtunnel import SSHTunnelForwarder


class SSHTunnel:
    @staticmethod
    def port_mapping(*, server_ip, server_port, username, password, port_mappings=None):
        # SSH服务器的IP地址、端口、用户名和密码
        print("SSH server ip", server_ip)
        if port_mappings is None:
            port_mappings = {
                8888: 8888,  # 本地端口9090到服务器端口9090
                5701: 5701,  # 本地端口9091到服务器端口9091
                # 添加更多端口映射
            }
        for _port in port_mappings.keys():
            print(f"http://localhost:{_port}")

        ssh_server = server_ip
        ssh_port = server_port
        ssh_user = username
        ssh_password = password
        remote_bind_address = [("your_server_ip", server_port) for _, server_port in port_mappings.items()]
        local_bind_address = [('your_server_ip', local_port) for local_port, _ in port_mappings.items()]
        # 使用SSHTunnelForwarder设置端口映射
        with SSHTunnelForwarder(
                (ssh_server, ssh_port),
                ssh_username=ssh_user,
                ssh_password=ssh_password,
                remote_bind_addresses=remote_bind_address,
                local_bind_addresses=local_bind_address
        ) as tunnel:
            print("Port forwarding is setup.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Port forwarding has been stopped.")
