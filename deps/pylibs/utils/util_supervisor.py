#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/20 14:06
# @Author  : gsunwu@163.com
# @File    : util_supervisor.py
# @Description:
"""
pip install supervisor
"""
import os.path
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pylibs.config import Env, Debug
from pylibs.utils.util_bash import BashUtil
from pylibs.utils.util_network import is_port_listing
from pylibs.utils.util_port import wait_kill_process_on_port
from typing import Union

"""
# http://supervisord.org/configuration.html#program-x-section-settings
[program:cat]
command=/bin/cat
process_name=%(program_name)s
numprocs=1
directory=/tmp
umask=022
priority=999
autostart=true
autorestart=unexpected
startsecs=10
startretries=3
exitcodes=0
stopsignal=TERM
stopwaitsecs=10
stopasgroup=false
killasgroup=false
user=chrism
redirect_stderr=false
stdout_logfile=/a/path
stdout_logfile_maxbytes=1MB
stdout_logfile_backups=10
stdout_capture_maxbytes=1MB
stdout_events_enabled=false
stderr_logfile=/a/path
stderr_logfile_maxbytes=1MB
stderr_logfile_backups=10
stderr_capture_maxbytes=1MB
stderr_events_enabled=false
environment=A="1",B="2"
serverurl=AUTO
"""


@dataclass
class Supervisord:
    cmds: List[str] = None
    out_file: Union[Path, str] = Path("/tmp/supervisord.conf")
    http_port: int = 65534
    # KEY="val",KEY2="val2"
    environment: str = 'MALLOC_TRIM_THRESHOLD_="65536"'

    def _add_cmd(self, cmd):
        self.cmds.append(f"{cmd}")

    def __post_init__(self):
        self.cmds = []
        if type(self.out_file) == "str":
            self.out_file = Path(self.out_file)
        SUPERVISORD_COMMONS = f"""
[supervisord] 
logfile=/tmp/supervisord.log 
logfile_maxbytes=50MB  
logfile_backups=10 
loglevel=info 
pidfile=/tmp/supervisord_{self.http_port}.pid  
nodaemon=false  

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface
# supervisorctl  -s http://localhost:6005/ status

[inet_http_server]
port=your_server_ip:{self.http_port}
username=sunwu
password=gzudb@604
        """
        for i in SUPERVISORD_COMMONS.split("\n"):
            self.cmds.append(i)

    def write_to_file(self):
        if not self.out_file.parent.exists():
            self.out_file.parent.mkdir()
        print(f"Write file to {self.out_file.absolute()}")
        with open(self.out_file, "w+") as f:
            f.write("\n".join(self.cmds))

        return self.out_file.absolute()

    def add_program(self, name, command, autostart="true", startretries=9999):
        # print(f" program name: {name} ")
        # print(f" program command: {command}")
        self._add_cmd(f"[program:{name}]")
        self._add_cmd(f"command={command}")
        self._add_cmd(f"environment={self.environment}")
        self._add_cmd(f"autostart={autostart}")
        self._add_cmd(f"startretries={startretries}")
        self._add_cmd(f"stderr_logfile=/tmp/{name}_err.log")
        self._add_cmd(f"stdout_logfile=/tmp/{name}_out.log")
        self._add_cmd(f"redirect_stderr=true")
        self._add_cmd(f"exitcodes=9,8")
        self._add_cmd(f"")

    def start(self):
        conf_file = self.write_to_file()
        if is_port_listing(self.http_port):
            wait_kill_process_on_port(self.http_port)

        res = BashUtil.exe_cmd(f"{Env.get_supervisord_exe()} -c {conf_file}")
        """
        Error: Another program is already listening on a port that one of our HTTP servers is configured to use.  Shut this program down first before starting supervisord.
        For help, use /Users/sunwu/miniforge3/envs/amltk/bin/supervisord -h
        """
        if res.find("Error") > -1:
            print(f"❌❌❌  Start supervisor failed since {res}")
        else:
            print(f"Supervisor is running at:http://localhost:{self.http_port}")

    def start_remote_worker(self):
        conf_file = self.write_to_file()
        if is_port_listing(self.http_port):
            wait_kill_process_on_port(self.http_port)

        res = BashUtil.exe_cmd(f"{Env.get_supervisord_exe()} -c {conf_file}")
        """
        Error: Another program is already listening on a port that one of our HTTP servers is configured to use.  Shut this program down first before starting supervisord.
        For help, use /Users/sunwu/miniforge3/envs/amltk/bin/supervisord -h
        """
        if res.find("Error") > -1:
            print(f"❌❌❌  Start supervisor failed since {res}")
        else:
            print(f"Supervisor is running at:http://localhost:{self.http_port}")

    def remote_start_dask_scheduler(self, dask_scheduler):
        """

        Parameters
        ----------
        dask_scheduler :  pylibs.utils.util_servers.Server

        Returns
        -------

        """
        conf_name = f"dask_scheduler_on_{dask_scheduler.name}.conf"
        remote_config_name = Path(dask_scheduler.work_home, conf_name).as_posix()
        self.out_file = Path(Env.get_runtime_home(), conf_name)
        file_name = self.write_to_file()
        dask_scheduler.upload_file_by_ssh(str(file_name), remote_config_name)
        assert dask_scheduler.supervisord_exec is not None, "Supervisord can't be None"
        dask_scheduler.remote_start_supervisord(dask_scheduler, remote_config_name)
        while not dask_scheduler.is_dask_scheduler_running(dask_scheduler):
            print("Waiting scheduler to run ...")
            time.sleep(1)

    def remote_start_dask_workers(self, dask_worker, dask_scheduler):
        """

        Parameters
        ----------
        dask_worker : pylibs.utils.util_servers.Server
        dask_scheduler :  pylibs.utils.util_servers.Server

        Returns
        -------

        """
        conf_name = f"dask_workers_on_{dask_worker.name}.conf"
        remote_config_name = Path(dask_worker.work_home, conf_name).as_posix()
        self.out_file = Path(Env.get_runtime_home(), conf_name)
        file_name = self.write_to_file()
        dask_worker.upload_file_by_ssh(str(file_name), remote_config_name)
        assert dask_worker.supervisord_exec is not None, "Supervisord can't be None"
        dask_scheduler.remote_start_supervisord(dask_worker, remote_config_name)

    def remote_start_dask_workers_ssh_channel(self, dask_worker, dask_scheduler, port_mappings: dict):
        """

        Parameters
        ----------
        dask_worker : pylibs.utils.util_servers.Server
        dask_scheduler :  pylibs.utils.util_servers.Server

        Returns
        -------

        """

        if dask_scheduler.name == dask_worker.name:
            print("Skip for ssh tunnel since dask_scheduler.name == dask_worker.name")
            return False
        else:
            conf_name = f"ssh_tunel_from_{dask_scheduler.name}_to_{dask_worker.name}.conf"
            self.out_file = Path(Env.get_runtime_home(), conf_name)
            file_name = self.write_to_file()
            remote_conf = Path(f"{dask_scheduler.work_home}/{conf_name}").as_posix()
            # /root/miniconda3/bin/supervisord -f /tmp/ssh_tunel_S219.conf
            dask_scheduler.upload_file_by_ssh(str(file_name), remote_conf)
            dask_scheduler.remote_start_supervisord(dask_scheduler, remote_conf)

            # Ensure that the one of ssh channel is available
            # (base) $curl your_server_ip:30001
            # curl: (1) Received HTTP/0.9 when not allowed
            _first_listing_port = list(port_mappings)[0]

            def is_ssh_available():
                return dask_scheduler.get_ssh().exec(f"curl your_server_ip:{_first_listing_port}").find(
                    "Received HTTP/0.9 when not allowed") > -1

            while not is_ssh_available():
                print("Waiting for ssh tunnel is working...")
                time.sleep(1)

            return True

    def remote_start_service(self, ser):
        file_name = self.write_to_file()
        remote_file_name = os.path.join("/etc", os.path.basename(self.out_file))
        ser.upload_file(str(file_name), remote_file_name)
        ser.get_ssh().exec(f"{ser.supervisord_exec} -c {remote_file_name}")


if __name__ == '__main__':
    s = Supervisord()
    s.add_program("bb", "tail -f /dev/null")
    s.add_program("bb", "tail -f /dev/null")
    s.start()
