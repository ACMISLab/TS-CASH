#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/19 00:05
# @Author  : gsunwu@163.com
# @File    : util_frp.py
# @Description:
# frpc 3.71
"""
[common]
server_addr = your_server_ip
server_port = 12000
token = oynqu9qntchvhxff7mpkxfyqdzmiov9xk19v40dj4kqnpxvaav2553hyv26h2wlv
tls_enable = true
log_level = trace

[nextcloud]
type = tcp
local_ip = your_server_ip
local_port = 11123
remote_port = 17004
use_encryption = true
use_compression = true
tls_enable = true

"""
# http://your_server_ip:7500
from dataclasses import dataclass
from pathlib import Path

from pylibs.utils.util_process import kill_process_by_name
from pylibs.config import Env
from pylibs.utils.util_bash import BashUtil
from pylibs.utils.util_log_server import zs
from pylibs.utils.util_servers import Servers, Server
from pylibs.utils.util_supervisor import Supervisord

log = zs.get_logger("dask")


@dataclass
class UtilFrps:
    confs = []
    conf_home = "/etc/"
    frpc_exe_path = "/usr/local/bin/frps"
    frpc_ini_path = "/etc/frps.ini"
    # /usr/local/bin/frpc -c /etc/frpc.ini
    frps_server: int = Env.FRP_SERVER_LOCAL

    def __post_init__(self):
        # [common]
        # server_addr = your_server_ip
        # server_port = 12000
        # token = oynqu9qntchvhxff7mpkxfyqdzmiov9xk19v40dj4kqnpxvaav2553hyv26h2wlv
        # tls_enable = true
        # log_level = trace
        self._add_cmd("[common]")
        self._add_cmd(f"server_addr = {self.frps_server}")
        self._add_cmd("server_port = 12000")
        self._add_cmd("token = oynqu9qntchvhxff7mpkxfyqdzmiov9xk19v40dj4kqnpxvaav2553hyv26h2wlv")
        self._add_cmd("tls_enable = true")
        self._add_cmd("log_level = trace")

    def _add_cmd(self, cmd):
        self.confs.append(cmd + "\n")

    def add_channel(self, name, local_port, remote_port, local_ip="your_server_ip"):
        # [nextcloud]
        # type = tcp
        # local_ip = your_server_ip
        # local_port = 11123
        # remote_port = 17004
        # use_encryption = true
        # use_compression = true
        # tls_enable = true
        self._add_cmd(f"[{name}]")
        self._add_cmd(f"type = tcp")
        self._add_cmd(f"local_ip = {local_ip}")
        self._add_cmd(f"local_port = {local_port}")
        self._add_cmd(f"remote_port = {remote_port}")
        self._add_cmd(f"use_encryption = true")
        self._add_cmd(f"use_compression = true")
        self._add_cmd(f"tls_enable = true")

    def save(self):
        p = Path(Env.get_runtime_home(), "frpc.ini")
        if not p.parent.exists():
            p.parent.mkdir(parents=True)
        with open(p, "w+") as f:
            f.writelines(self.confs)
        return str(p.absolute())

    def start_frpc(self):
        pass

    def upload_frpc_ini(self, server: Server):
        f = self.save()
        print(f"Uploading {f} to {self.frpc_ini_path}")
        server.upload_file(f, self.frpc_ini_path)

    def start(self, server: Server):
        self.upload_frpc_ini(server)
        self.upload_frpc_exe(server)
        ssh = server.get_ssh()
        ssh.exec(f"nohup {self.frpc_exe_path} -c {self.frpc_ini_path} >>/tmp/frpc.log 2>&1 &")

    def upload_frpc_exe(self, server: Server):
        ssh = server.get_ssh()

        server.upload_file(
            Env.FRPC_EXE_HOME,
            self.frpc_exe_path
        )
        ssh.send("chmod +x /usr/local/bin/frpc")


@dataclass
class UtilFrpc:
    confs = []
    conf_home = "/etc/"
    frpc_exe_path = "/usr/local/bin/frpc"
    frpc_ini_path = "/etc/frpc.ini"
    # /usr/local/bin/frpc -c /etc/frpc.ini
    frps_server: int = Env.FRP_SERVER_LOCAL

    def __post_init__(self):
        # [common]
        # server_addr = your_server_ip
        # server_port = 12000
        # token = oynqu9qntchvhxff7mpkxfyqdzmiov9xk19v40dj4kqnpxvaav2553hyv26h2wlv
        # tls_enable = true
        # log_level = trace
        self._add_cmd("[common]")
        self._add_cmd(f"server_addr = {self.frps_server}")
        self._add_cmd("server_port = 12000")
        self._add_cmd("token = oynqu9qntchvhxff7mpkxfyqdzmiov9xk19v40dj4kqnpxvaav2553hyv26h2wlv")
        self._add_cmd("tls_enable = true")
        self._add_cmd("log_level = trace")

    def _add_cmd(self, cmd):
        self.confs.append(cmd + "\n")

    def add_channel(self, name, local_port, remote_port, local_ip="your_server_ip"):
        # [nextcloud]
        # type = tcp
        # local_ip = your_server_ip
        # local_port = 11123
        # remote_port = 17004
        # use_encryption = true
        # use_compression = true
        # tls_enable = true
        self._add_cmd(f"[{name}]")
        self._add_cmd(f"type = tcp")
        self._add_cmd(f"local_ip = {local_ip}")
        self._add_cmd(f"local_port = {local_port}")
        self._add_cmd(f"remote_port = {remote_port}")
        self._add_cmd(f"use_encryption = true")
        self._add_cmd(f"use_compression = true")
        self._add_cmd(f"tls_enable = true")

    def save(self):
        p = Path(Env.get_runtime_home(), "frpc.ini")
        if not p.parent.exists():
            p.parent.mkdir(parents=True)
        with open(p, "w+") as f:
            f.writelines(self.confs)
        return str(p.absolute())

    def start_frpc(self):
        pass

    def upload_frpc_ini(self, server: Server):
        f = self.save()
        print(f"Uploading {f} to {self.frpc_ini_path}")
        server.upload_file(f, self.frpc_ini_path)

    def remote_start(self, server: Server):
        self.upload_frpc_ini(server)
        self.upload_frpc_exe(server)
        ssh = server.get_ssh()
        ssh.exec("pkill frpc")
        ssh.exec(f"nohup {self.frpc_exe_path} -c {self.frpc_ini_path} >/tmp/frpc.log 2>&1 &")

    def local_start(self):
        frpc_conf = self.save()
        self.kill_local_frpc()
        # super = Supervisord(out_file=Path("/tmp/frpc_supervisord.conf"))
        log.info(BashUtil.exe_cmd(f"nohup {self.frpc_exe_path} -c {frpc_conf} >/tmp/frpc.log 2>&1 &"))

    def upload_frpc_exe(self, server: Server):
        ssh = server.get_ssh()

        server.upload_file(
            Env.get_frpc_home(),
            self.frpc_exe_path
        )
        ssh.send("chmod +x /usr/local/bin/frpc")

    def kill_local_frpc(self):
        kill_process_by_name(self.frpc_exe_path)


if __name__ == '__main__':
    s1 = Servers.S100_9
    uf = UtilFrpc()
    uf.add_channel("s1", local_ip=Env.FRP_SERVER_LOCAL, local_port=3333, remote_port=3333)
    uf.remote_start(s1)
