import warnings

warnings.filterwarnings("ignore")
import ctypes
import logging
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from distributed import Client, PipInstall
from pylibs.config import Env, CONF, Debug
from pylibs.utils.util_bash import BashUtil, UtilBash
from pylibs.utils.util_gpu import automatic_memory_decition_by_gpu_total_memory
from pylibs.utils.util_ssh import SSHClient
from pylibs.utils.util_ssh_tunnel import SSHTunnel
from pylibs.utils.util_supervisor import Supervisord
from pylibs.utils.util_system import UtilSys

log = logging.getLogger(__name__)


class DaskRunnerConfig:
    DASK_WORKER_RUNNER_RELATIVE_PATH = "pylibs/dask/distributed/worker_runner.py"
    DASK_SCHEDULER_RUNNER_RELATIVE_PATH = "pylibs/dask/distributed/scheduler_runner.py"


@dataclass
class ENV(Env):
    pass


class GPUType:
    RTX_A100 = "RTX_A100"
    RTX_A6000 = "RTX_A6000"
    RTX_2080TI = "RTX_2080TI"
    # 不是GPU
    EMPTY = None

    @staticmethod
    def get_total_memory_by_type(gpu_type: str):
        #    # 2080TI: 11019
        #     # A6000: 49140
        #     # A100:  40536

        # No GPU for given
        if gpu_type == "None" or gpu_type is None:
            return 0

        if gpu_type == GPUType.RTX_2080TI:
            return 11019
        elif gpu_type == GPUType.RTX_A6000:
            return 49140
        elif gpu_type == GPUType.RTX_A100:
            return 40536
        else:
            raise ValueError(f"Unknown GPU type {gpu_type}")


from typing import Union, Optional


@dataclass
class Server:
    ip: str
    port: Union[int, None]
    username: Optional[str] = None
    password: Optional[str] = None
    dask_port: Optional[int] = None
    name: Optional[str] = None
    dask_worker_port_begin: Optional[int] = 50000
    gpu_type: Optional[str] = GPUType.EMPTY
    desc: str = None
    # [remote_start,end]
    dask_worker_port_count: int = 190
    python_exec: str = "/root/miniconda3/bin/python"
    python_sshtunnel_exe = "/root/miniconda3/bin/sshtunnel"
    n_gpu_card: int = 0
    n_cpu_core: int = 32
    pip_exec = python_exec[:-6] + "pip"
    work_home: str = "/remote-home/cs_acmis_sunwu/"
    # 存储数据用的地方
    data_home: str = "/home/data"
    RSYNC_PYTHON_FILTER = ' -f"+ */" -f"- ***/.git/***" -f"+ *.py" -f"+ *.sh"  -f"+ **/Dockerfile" -f"+ **/Makefile"  -f"- *" '
    RSYNC_FULL = " "
    PYLIB_UPLOAD_FLAG = False

    """
    import options:
    -m, --prune-empty-dirs      prune empty directory chains from the file-list

    """
    LIB_PATH = "/tmp/pylibs-1.0.0-py3-none-any.whl"
    LIB_WHEEL_PATH = "/tmp/pylibs-1.0.0-py3-none-any.whl"

    dask_client: Optional[Client] = None
    _sc: SSHClient = None
    supervisord_exec: str = "/root/miniconda3/bin/supervisord"
    debug: bool = False
    pylibs_home: str = "/remote-home/cs_acmis_sunwu/2024/pylibs"

    def get_ssh(self) -> SSHClient:
        if self._sc is None:
            self._sc = SSHClient(self.ip, self.port, self.username, self.password)
        return self._sc

    def get_scheduler_address(self):
        return f"tcp://{self.ip}:{self.dask_port}"

    def upload_dir(self, local_path: Union[Path, str], remote_path: Union[Path, str],
                   rsync_option: str = '-f"- ***/.git/***" -f"- ***/.idea/***" -f"- ***cache/***" -f"- ***cachedir/***" -f"- ***/__pycache__/***" -f"- *.npz" -f"- *.xlsx" -f"- *.csv" -f"- *.arff" -f"- *.png" -f"- *.html"  -f"- ***/results/***"'):
        """
        Upload a dir.
         -f"- *.json" -f"- *.pkl"
        rsync_option="" upload all file in the directory

        Parameters
        ----------
        local_path : str
        remote_path : str
        rsync_option : str
            文件过滤表达式， 如 -f"- *.py", -f"+ *.py"


        Returns
        -------

        """
        if type(local_path) == str:
            local_path = Path(local_path)

        if type(remote_path) == str:
            remote_path = Path(remote_path)

        cmd = f'sshpass -p \'{self.password}\' rsync  -h -e "ssh -p {self.port} -o PubkeyAuthentication=yes -o stricthostkeychecking=no" -avi --copy-links -m  {rsync_option} -L {local_path.absolute()}/ {self.username}@{self.ip}:{remote_path.absolute()}/'
        log.info(f"Upload {local_path.absolute()} to {remote_path.absolute()}")
        # log.debug(cmd)
        self.exe_local_cmd(cmd)

    def deploy_dask(self):
        DaskDeploy.deploy_server(self)

    def upload_tailscale(self,
                         tailscale_path: Path = Path("/Users/sunwu/Nextcloud/App/tailscale/tailscale_1.60.0_amd64/")):
        """

        Parameters
        ----------
        tailscale_path : Path
            Tailscale and tailscaled are under directory.

        Returns
        -------

        """
        ssh = self.get_ssh()
        # ssh.send('rm -rf /usr/local/bin/tailscale')
        # ssh.send('rm -rf /usr/local/bin/tailscaled')

        self.upload_dir(
            tailscale_path,
            Path("/usr/local/bin/"), rsync_option=""
        )
        ssh.send("chmod +x /usr/local/bin/tailscale")
        ssh.send("chmod +x /usr/local/bin/tailscaled")
        ssh.send(
            "nohup /usr/local/bin/tailscaled --tun=userspace-networking --socks5-server=localhost:1055 --outbound-http-proxy-listen=localhost:1055 -verbose 1 > /tmp/tailscaled.log 2>&1 &")
        ssh.send(
            f"tailscale up --login-server=http://your_server_ip:8280 --accept-routes=true --accept-dns=false --authkey acb9133f8558bc364ad8397bff9d9bba6cc246c798ec9a0b --reset --hostname server-{self.ip.replace('.', '-')}")

        ssh.send("cat /tmp/tailscaled.log")

    def upload_frpc(self):
        ssh = self.get_ssh()
        self.upload_file(
            f"{Env.get_frpc_home()}/frpc",
            "/usr/local/bin/frpc"
        )
        ssh.send("chmod +x /usr/local/bin/frpc")

    def upload_file(self, source_file: str, remote_file: str):
        """

        Parameters
        ----------


        Returns
        -------

        """
        rsync = True
        assert Path(source_file).exists(), f"Source file is not found: {source_file}"
        cmd = f'sshpass -p \'{self.password}\' rsync  -h -e "ssh -p {self.port} -o PubkeyAuthentication=yes -o stricthostkeychecking=no" -avi -m  {source_file} {self.username}@{self.ip}:{remote_file}'
        log.info(f"Upload {source_file} to {remote_file}")
        res = BashUtil.exe_cmd(cmd)
        # 'building file list ... done
        # <f..t.... ssh_tunel_from_S164_to_S153.conf
        #
        # sent 157 bytes  received 54 bytes  140.67 bytes/sec
        # total size is 1.27K  speedup is 6.03'
        if str(res).find("total size is") == -1:
            print(f"Failed to upload file {source_file}")
            Debug.err_found()

    def upload_file_by_ssh(self, source_file: str, remote_file: str):
        """

        Parameters
        ----------


        Returns
        -------

        """
        if type(source_file) == str:
            source_file = Path(source_file)
        if type(remote_file) == str:
            remote_file = Path(remote_file)

        res = self.get_ssh().upload_file_with_ftp(source_file.as_posix(), remote_file.as_posix())
        target_size = res.st_size
        local_size = os.path.getsize(source_file.as_posix())
        assert local_size == target_size, "Failed to upload file."
        print(f"Upload {source_file} to {remote_file} successfully.")

    def get_ftp(self):
        return self.get_ssh().ftp

    def prepare_env(self):
        """
        Prepare environment for running experiments

        Returns
        -------

        """
        if self.ip == "localhost" or self.ip == "your_server_ip":
            # pass
            pass
        else:
            return DEVHelper.prepare_env(self)

    def prepare_pylibs(self):
        return self.upload_pylibs()

    def upload_pylibs(self):
        """
        Prepare environment for running experiments

        Returns
        -------

        """
        if self.ip == "localhost" or self.ip == "your_server_ip":
            # pass
            pass
        else:
            if self.PYLIB_UPLOAD_FLAG:
                return
            else:
                ssh = self.get_ssh()
                ssh.mkdir(ENV.REMOTE_PYLIBS_HOME.absolute())
                self.upload_dir(Env.get_pylibs_home(),
                                ENV.REMOTE_PYLIBS_HOME, rsync_option=Server.RSYNC_PYTHON_FILTER)
                ssh.exec(f"{self.pip_exec} install -e {Env.REMOTE_PYLIBS_HOME}")
                self.PYLIB_UPLOAD_FLAG = True

    def get_dask_client_from_server(self) -> Client:
        if self.ip is None:
            return None
        assert self.dask_port is not None, "Dask port must be specified."
        client = Client(
            address=f"{self.ip}:{self.dask_port}",
            direct_to_workers=True,
            timeout=30)

        # try to fix Unmanaged (Old) memory hanging
        # https://github.com/dask/distributed/issues/6232
        client.amm.start()
        if platform.system() == "Linux":
            # fix:  high unmanaged memory usage or “memory leak” warnings on workers can be misleading, seeing https://distributed.dask.org/en/latest/worker-memory.html#memory-not-released-back-to-the-os
            def trim_memory() -> int:
                libc = ctypes.CDLL("libc.so.6")
                return libc.malloc_trim(0)

            client.run(trim_memory)
        return client

    def get_dask_client(self) -> Client:
        """
        usage: sumbit -> gather

        Examples:
        from pylibs.utils.util_servers import Servers
        def f(a=1):
            return a ** 2

        client = Servers.S219.get_dask_client()
        features_results= []
        for i in range(100):
            features_results.append(client.submit(f, a=i))
        r = client.gather(features_results)
        print(r)


        Returns
        -------

        """
        return self.get_dask_client_from_server()

    def get_scheduler_worker(self) -> Client:
        client = Client(
            address=f"{Servers.DASK_SCHEDULER.ip}:{Servers.DASK_SCHEDULER.dask_port}",
            direct_to_workers=False,
            timeout=3)

        if platform.system() == "Linux":
            # fix:  high unmanaged memory usage or “memory leak” warnings on workers can be misleading, seeing https://distributed.dask.org/en/latest/worker-memory.html#memory-not-released-back-to-the-os
            def trim_memory() -> int:
                libc = ctypes.CDLL("libc.so.6")
                return libc.malloc_trim(0)

            client.run(trim_memory)
        return client

        # noinspection PyUnreachableCode
        def trim_memory() -> int:
            libc = ctypes.CDLL("libc.so.6")
            return libc.malloc_trim(0)

        client.run(trim_memory)

    def __str__(self):
        return self.name

    def get_dask_client_self(self) -> Client:
        if UtilSys.is_macos() or self.name == Servers.DASK_SCHEDULER.name:
            # 如果是本地，那么就获取远程地址
            log.info(f"Getting dask client: {self.ip}:{self.dask_port}")
            client = Client(
                address=f"{self.ip}:{self.dask_port}",
                direct_to_workers=False,
                timeout=3)

        else:
            # 如果不是本地，那就使用本地运行
            log.info(f"Getting dask client: your_server_ip:6006")
            client = Client(
                address=f"your_server_ip:6006",
                direct_to_workers=False)

        # self.install_pylibs_by_pip(client)
        return client

    def get_dask_client_one_instance(self) -> Client:

        if self.dask_client is None:
            if UtilSys.is_macos() or self.name == Servers.DASK_SCHEDULER.name:
                # 如果是本地，那么就获取远程地址
                log.info(f"Getting dask client: {self.ip}:{self.dask_port}")
                self.dask_client = Client(
                    address=f"{self.ip}:{self.dask_port}",
                    direct_to_workers=False,
                    timeout=3)

            else:
                # 如果不是本地，那就使用本地运行
                log.info(f"Getting dask client: your_server_ip:6006")
                self.dask_client = Client(
                    address=f"your_server_ip:6006",
                    direct_to_workers=False)

        return self.dask_client

    @DeprecationWarning
    def prepare_dask_worker(self):
        self.clear_env()
        self.upload_pylibs()
        self.run_dask_worker()

    def prepare_dask_workers(self, scheduler, dask_worker_port_begin=30000):
        """
        You must start the scheduler before running this
        Parameters
        ----------
        scheduler :
        dask_worker_port_begin :

        Returns
        -------

        """
        #   check to start the scheduler before running this
        assert self.is_dask_scheduler_running(scheduler)
        self.dask_worker_port_begin = dask_worker_port_begin
        # self.clear_env()
        # self.upload_pylibs()
        self.start_dask_workers(scheduler)

    def prepare_dask_workers_cpu(self, scheduler, dask_worker_port_begin=30000):
        """
        You must start the scheduler before running this
        Parameters
        ----------
        scheduler :
        dask_worker_port_begin :

        Returns
        -------

        """
        #   check to start the scheduler before running this
        assert self.is_dask_scheduler_running(scheduler)
        self.dask_worker_port_begin = dask_worker_port_begin
        self.start_dask_workers_cpu(scheduler)

    def prepare_dataset(self):
        _local_home = Path("/Users/sunwu/Documents/uts_benchmark_dataset/benchmark")
        _remote_home = Path("/remote-home/cs_acmis_sunwu/uts_benchmark_dataset")
        self.upload_dir(_local_home, _remote_home, rsync_option="")

    @DeprecationWarning
    def prepare_egg(self):
        """
        Note: .egg has a lot of problems, disable to use it
        Returns
        -------

        """
        print("🚕 Preparing .egg...")
        # Build and upload .egg
        self.exe_local_cmd(
            f"cd {Env.get_pylibs_home()};make egg"
        )

        self.upload_file(Env.LOCAL_PYLIBS_EGG, Server.LIB_PATH)
        print("✅ Preparing .egg mark_as_finished")

    def prepare_wheel(self):
        """
        Note: .egg has a lot of problems, disable to use it
        Returns
        -------

        """
        print("🚕 Preparing .wheel...")
        self.exe_local_cmd(f"cd {Env.get_pylibs_home()};make wheel")
        self.upload_file(Env.get_wheel_path(), Env.LIB_WHEEL_PATH_REMOTE)
        self.install_wheel_remote()
        print("✅ Preparing .whell mark_as_finished")

    def build_wheel(self):
        """
        Note: .egg has a lot of problems, disable to use it
        Returns
        -------

        """
        print("🚕 Preparing .egg...")
        # Build and upload .egg

        self.exe_local_cmd(f"cd {Env.get_pylibs_home()};make wheel")
        return ENV.get_wheel_path()

    def build_egg(self):
        """
        Note: .egg has a lot of problems, disable to use it
        Returns
        -------

        """
        print("🚕 Preparing .egg...")
        # Build and upload .egg

        self.exe_local_cmd(f"cd {Env.get_pylibs_home()};make egg")
        return Env.LOCAL_PYLIBS_EGG

    def prepare_tailscale(self, tailscale_path: Path = None):

        """

        Parameters
        ----------
        tailscale_path : Path
            Tailscale and tailscaled are under directory.

        Returns
        -------

        """
        if tailscale_path is None:
            tailscale_path = Path(Env.TAILSCALE_SAVE_PATH)
        ssh = self.get_ssh()
        # ssh.send('rm -rf /usr/local/bin/tailscale')
        # ssh.send('rm -rf /usr/local/bin/tailscaled')

        self.upload_dir(
            tailscale_path,
            Path("/usr/local/bin/"), rsync_option=""
        )
        ssh.send("chmod +x /usr/local/bin/tailscale")
        ssh.send("chmod +x /usr/local/bin/tailscaled")

        def get_host_name():
            return f"auto_login_{self.name}".replace("_", "-")

        ssh.send(
            "nohup /usr/local/bin/tailscaled --tun=userspace-networking --socks5-server=localhost:1055 --outbound-http-proxy-listen=localhost:1055 -verbose 1 > /tmp/tailscaled.log 2>&1 &")
        # ssh.send(
        # f"tailscale up --login-server=http://your_server_ip:8280 --accept-routes=true --accept-dns=false --authkey acb9133f8558bc364ad8397bff9d9bba6cc246c798ec9a0b --reset --hostname {get_host_name()}")
        # tailscale up --login-server=http://your_server_ip:8280 --force-reauth --accept-routes=true --accept-dns=false --authkey d02ac23560d7ccf5e47637d5d92f4ef993d0c794553d99eb --reset
        ssh.send(
            f"tailscale up --login-server=http://your_server_ip:8280 --force-reauth --accept-routes=true --accept-dns=false --authkey d02ac23560d7ccf5e47637d5d92f4ef993d0c794553d99eb --reset --hostname {get_host_name()}")

        # ssh.send("cat /tmp/tailscaled.log")

    def run_dask_worker(self):
        ssh = self.get_ssh()
        print("Run dask worker...")
        res = ssh.exec(
            f"{self.python_exec} {Env.REMOTE_PYLIBS_HOME}/pylibs/dask/distributed/start_dask_worker.py --port-start {self.dask_worker_port_begin} --connect-address your_server_ip --server-name {self.name}"
        )
        # check worker is running
        while True:
            supervised_str = ssh.exec("pgrep supervisord")
            ports = re.findall("\d{3,}", supervised_str)
            if len(ports) > 0:
                break
            else:
                print("Waiting for supervisord to start")
            time.sleep(1)

        if Env.is_process_success(res):
            print("✅✅✅ Deploy successfully")

        else:
            print("❌❌❌ Deploy failed since ", res)

    def start_frpc_remote(self):
        from pylibs.utils.util_frp import UtilFrpc
        frp = UtilFrpc()
        for i in range(self.dask_worker_port_begin):
            frp.add_channel(name=f"{self.name}-{self.ip}-{i}",
                            local_port=self.dask_worker_port_begin + i,
                            remote_port=self.dask_worker_port_begin + i,
                            )
            if i >= self.dask_worker_port_count:
                break
        frp.remote_start()

    def start_frpc_local(self):
        from pylibs.utils.util_frp import UtilFrpc
        frp = UtilFrpc()
        for i in range(self.dask_worker_port_begin):
            frp.add_channel(name=f"{self.name}-{self.ip}-{i}",
                            local_port=self.dask_worker_port_begin + i,
                            remote_port=self.dask_worker_port_begin + i,
                            )
            if i >= self.dask_worker_port_count:
                break
        frp.local_start()

    def get_frpc_conf_path(self):
        from pylibs.utils.util_frp import UtilFrpc
        frp = UtilFrpc()
        for i in range(self.dask_worker_port_begin):
            frp.add_channel(name=f"{self.name}-{self.ip}-{i}",
                            local_port=self.dask_worker_port_begin + i,
                            remote_port=self.dask_worker_port_begin + i,
                            )
            if i >= self.dask_worker_port_count:
                break
        return frp.save()

    def prepare_scheduler(self):
        if self.is_dask_scheduler_running(self):
            print("Scheduler is already running")
            return

        su = Supervisord(
            out_file=Path(self.work_home, f"dask_scheduler_supervisord.conf").as_posix(),
            http_port=5901
        )

        su.add_program("dask_scheduler",
                       f"{self.python_exec} {self.pylibs_home}/pylibs/dask/distributed/scheduler_runner.py")

        # 自动重启worker，如果内存大于给定阈值的话
        # # try to fix Unmanaged (Old) memory hanging
        # https://github.com/dask/distributed/issues/6232
        su.add_program("dask_worker_auto_restart",
                       f"{self.python_exec} {self.pylibs_home}/pylibs/dask/distributed/auto_restart_worker.py --threshold 10")

    def prepare_scheduler_cpu(self):
        if self.is_dask_scheduler_running(self):
            print("Scheduler is already running")
            return

        su = Supervisord(
            out_file=Path(self.work_home, f"dask_scheduler_supervisord.conf").as_posix(),
            http_port=5901
        )

        su.add_program("dask_scheduler",
                       f"/root/miniconda3/bin/dask  scheduler --host your_server_ip --port 6006 --dashboard-address your_server_ip:8888")

        # 自动重启worker，如果内存大于给定阈值的话
        # # try to fix Unmanaged (Old) memory hanging
        # https://github.com/dask/distributed/issues/6232
        # --contact-address
        # su.add_program("dask_worker_auto_restart",
        #                f"/root/miniconda3/bin/dask worker --nthreads 1  --memory-limit 6G  --no-nanny  --scheduler your_server_ip:8888")
        su.remote_start_dask_scheduler(self)

    def set_num_frpc_ports(self, n_port):
        self.dask_worker_port_count = n_port

    def clear_env(self):
        # ssh = self.get_ssh()
        # ssh.kill_program_by_name("python ")
        # ssh.exec(f"{self.pip_exec} uninstall pylibs -y")
        pass

    def install_wheel_remote(self):
        ssh = self.get_ssh()
        ssh.exec(f"{self.pip_exec} install {Env.LIB_WHEEL_PATH_REMOTE}")

    def exe_local_cmd(self, cmd):
        print(f"exec cmd: {cmd}")
        os.system(cmd)
        # res = UtilBash.run_command_print_progress(cmd)
        # print(res)

    def push_pylibs_to_gitea(self):
        print(BashUtil.exe_cmd("cd /Users/sunwu/SW-Research/pylibs;make p"))

    def install_pylibs_by_pip(self, client):
        self.push_pylibs_to_gitea()
        print("Update pylibs...")
        plugin = PipInstall(packages=
        [
            "git+http://aa9d08e0f02e1c17f2a5e4d744549b45396e9117@your_server_ip:15000/sunwu/pylibs.git@master"
        ],
            pip_options=["--upgrade"])
        client.register_plugin(plugin)

    def get_gpu_memory(self):
        return GPUType.get_total_memory_by_type(self.gpu_type)

    def get_number_of_gpus(self):
        return self.n_gpu_card

    def get_cpu_count(self):
        return self.n_cpu_core

    def kill_all(self):
        ssh = self.get_ssh()
        ssh.exec("pgrep supervisord|xargs  -r  kill  -9")
        ssh.exec("pgrep python|xargs  -r  kill  -9")
        ssh.exec("pgrep -f StrictHostKeyChecking|xargs  -r  kill  -9")
        ssh.exec("pkill screen")
        ssh.exec("pkill frpc")
        ssh.exec("pkill python")
        ssh.exec(f"pgrep -f zzzzzzzzzzzzz|xargs  -r  kill  -9 ")
        ssh.kill_program_by_substr("bin/dask")
        while ssh.exec("ps aux").find("python") > -1:
            ssh.exec("pgrep supervisord|xargs  -r  kill  -9")
            ssh.exec("pgrep python|xargs  -r  kill  -9")
            time.sleep(1)

    def port_mappings(self, port_maps):
        """

        Parameters
        ----------
        port_maps : dict
            {
                3000:3000
            }
        Returns
        -------

        """
        SSHTunnel.port_mapping(self.name, port_maps)

    def kill_headscale(self):
        # pgrep headscale|xargs -r kill -8
        self.get_ssh().exec("pgrep supervisord|xargs  -r  kill  -9")
        # self.get_ssh().exec("pgrep python|xargs  -r  kill  -9")
        self.get_ssh().exec("pgrep headscale|xargs  -r  kill  -9")
        self.get_ssh().exec("pgrep derper|xargs  -r  kill  -9")
        self.get_ssh().exec("pgrep caddy|xargs  -r  kill  -9")

    def prepare_python_packages(self):
        # pip install  psutil==5.7.2
        self.get_ssh().exec(
            f"{self.pip_exec} install sshtunnel==0.4.0 qiniu mysql-connector-python==8.3.0 supervisor==4.2.5  typeguard==4.2.1 sqlalchemy==2.0.29 pymysql==1.1.0 xgboost==2.0.3 arch==7.0.0 matplotlib==3.8 keras==2.9.0 seaborn==0.13.2 openpyxl==3.1.2 openai==1.35.10 bayesian-optimization==1.5.1 ConfigSpace==1.1.1")
        # self.upload_pylibs()

    def prepare_ubuntu_packages(self):
        self.get_ssh().exec("apt-get install -y git sshpass iftop iproute2 ")

    def kill_python(self):
        self.get_ssh().kill_program_by_name("main")
        self.get_ssh().kill_program_by_name("screen")

    def upload_runfiles(self):
        self.get_ssh().exec("rm -rf /tmp/main.py;rm -rf /tmp/search_pace.py")
        self.upload_file_by_ssh(CONF.get_main_path(), "/tmp/main.py")
        self.upload_file_by_ssh(CONF.get_search_space_path(), "/tmp/search_pace.py")

    def remote_start_supervisord(self, server, remote_conf: str):
        """

        Parameters
        ----------
        server :  pylibs.utils.util_servers.Server
        remote_conf :

        Returns
        -------

        """
        _ssh = server.get_ssh()
        _ssh.kill_program_by_substr(os.path.basename(remote_conf))
        res = _ssh.exec(f"{server.supervisord_exec} -c {remote_conf}")
        if res.find("Error") > -1:
            Debug.err_found()
            sys.exit(-1)

    def is_port_listing(self, port):
        res = self.get_ssh().exec(f"lsof -i :{port}")
        return res.find("LISTEN") > -1

    def generate_sshtunnel_cmd(self, port_mappings):
        """
        # _port_maps_sshtunnel = " ".join([f"your_server_ip:{_port}" for _port in port_mappings.keys()])
        # _tunnel_cmd = f"{self.dask_worker.python_sshtunnel_exe}  " \
        #               f"--username '{self.dask_worker.username}' " \
        #               f"--server_port {self.dask_worker.port} " \
        #               f"--password {self.dask_worker.password} " \
        #               f"--remote_bind_address {_port_maps_sshtunnel} " \
        #               f"--local_bind_address {_port_maps_sshtunnel} " \
        #               f"--threaded --verbose " \
        #               f" {self.dask_worker.ip} "

        Parameters
        ----------
        port_mappings :

        Returns
        -------

        """
        # _tunnel_cmd = self.generate_sshtunnel_cmd(port_mappings)

        _port_maps_sshtunnel = " ".join([f"your_server_ip:{_port}" for _port in port_mappings.keys()])
        _tunnel_cmd = f"{self.python_sshtunnel_exe}  " \
                      f"--username '{self.username}' " \
                      f"--server_port {self.port} " \
                      f"--password {self.password} " \
                      f"--remote_bind_address {_port_maps_sshtunnel} " \
                      f"--local_bind_address {_port_maps_sshtunnel} " \
                      f"--threaded --verbose " \
                      f" {self.ip} "
        return _tunnel_cmd

    def start_dask_workers(self, scheduler):
        self.dask_worker = self
        self.dask_scheduler = scheduler
        _gpu_total_memory = self.dask_worker.get_gpu_memory()
        if _gpu_total_memory > 0:
            _n_gpu = self.dask_worker.get_number_of_gpus()
            device_indexes = [i for i in range(_n_gpu)]
            n_worker_for_each_device = int(
                _gpu_total_memory / automatic_memory_decition_by_gpu_total_memory(_gpu_total_memory))
        else:
            n_worker_for_each_device = 1
            device_indexes = [-1 for i in range(int(self.dask_worker.get_cpu_count() / 2))]

        log.debug(f"device_indexes：{device_indexes}")
        log.debug(f"n_worker_for_each_device：{n_worker_for_each_device}")

        # sshpass -p X5bjN8UNyji2F5PaH2hbxNKoa ssh -CNg -L 3000:your_server_ip:12344 root@your_server_ip -p 12031
        dask_worker_manager = Supervisord(http_port=self.dask_worker.dask_worker_port_begin - 1)
        ssh_tunnel_manager = Supervisord(http_port=self.dask_worker.dask_worker_port_begin - 2)

        index = 0
        _worker_listing_port = self.dask_worker.dask_worker_port_begin - 1
        port_mappings = {}
        for _device_index in device_indexes:
            for _thread_index in range(n_worker_for_each_device):
                # increase the dask port
                _worker_listing_port = _worker_listing_port + 1

                # process name
                process_name = f"{self.dask_worker.name}-{_worker_listing_port}-g{_device_index}t{_thread_index}i{index}".replace(
                    "/", "").replace("-", "")

                # get_the ssh tunnel
                tunnel_port = {
                    _worker_listing_port: _worker_listing_port
                }
                port_mappings.update(tunnel_port)

                _contact_address = f"tcp://your_server_ip:{_worker_listing_port}"
                _worker_runner = os.path.join(self.dask_worker.pylibs_home,
                                              DaskRunnerConfig.DASK_WORKER_RUNNER_RELATIVE_PATH)
                _worker_cmd = f"{self.dask_worker.python_exec} {_worker_runner} --port {_worker_listing_port}  --scheduler {self.dask_scheduler.get_scheduler_address()} --process_name {process_name} --cuda_visible_devices {_device_index} --contact_address {_contact_address}"
                dask_worker_manager.add_program(process_name, _worker_cmd)
                index = index + 1

        # build ssh tunnel. open the tunnel in the scheduler, to the worker
        _tunnel_cmd = self.dask_worker.generate_sshtunnel_cmd(port_mappings)
        ssh_tunnel_manager.add_program(f"ssh_channel_for_{self.dask_worker.name}", _tunnel_cmd)

        # worker manager 在 worker 上执行
        dask_worker_manager.remote_start_dask_workers(self.dask_worker, self.dask_scheduler)

        # SSH tunnel 在 scheduler 上执行,
        # Notice: the worker has already started before run the  ssh channel
        _flag = ssh_tunnel_manager.remote_start_dask_workers_ssh_channel(self.dask_worker,
                                                                         self.dask_scheduler,
                                                                         port_mappings)

    def start_dask_workers_cpu(self, scheduler):
        self.dask_worker = self
        self.dask_scheduler = scheduler
        n_worker_for_each_device = 1
        device_indexes = [-1 for i in range(int(self.dask_worker.get_cpu_count() / 2))]

        log.debug(f"device_indexes：{device_indexes}")
        log.debug(f"n_worker_for_each_device：{n_worker_for_each_device}")

        # sshpass -p X5bjN8UNyji2F5PaH2hbxNKoa ssh -CNg -L 3000:your_server_ip:12344 root@your_server_ip -p 12031
        dask_worker_manager = Supervisord(http_port=self.dask_worker.dask_worker_port_begin - 1)

        index = 0
        _worker_listing_port = self.dask_worker.dask_worker_port_begin - 1
        port_mappings = {}
        for _device_index in device_indexes:
            for _thread_index in range(n_worker_for_each_device):
                # increase the dask port
                _worker_listing_port = _worker_listing_port + 1

                # process name
                process_name = f"{self.dask_worker.name}-{_worker_listing_port}-g{_device_index}t{_thread_index}i{index}".replace(
                    "/", "").replace("-", "")

                # get_the ssh tunnel
                tunnel_port = {
                    _worker_listing_port: _worker_listing_port
                }
                port_mappings.update(tunnel_port)

                _contact_address = f"tcp://your_server_ip:{_worker_listing_port}"
                _worker_runner = os.path.join(self.dask_worker.pylibs_home,
                                              DaskRunnerConfig.DASK_WORKER_RUNNER_RELATIVE_PATH)
                _worker_cmd = f"{self.dask_worker.python_exec} {_worker_runner} --port {_worker_listing_port}  --scheduler {self.dask_scheduler.get_scheduler_address()} --process_name {process_name}  --contact_address {_contact_address}"
                dask_worker_manager.add_program(process_name, _worker_cmd)
                index += 1

        # build ssh tunnel. open the tunnel in the scheduler, to the worker
        _tunnel_cmd = self.dask_worker.generate_sshtunnel_cmd(port_mappings)

        # worker manager 在 worker 上执行
        dask_worker_manager.remote_start_dask_workers(self.dask_worker, self.dask_scheduler)

    def is_dask_scheduler_running(self, scheduler) -> bool:
        """
        Check  whether the scheduler is running

        Parameters
        ----------
        scheduler : pylibs.utils.util_servers.Server

        Returns
        -------

        """
        scheduler_addr = scheduler.get_scheduler_address()
        # 'curl your_server_ip:20232
        # curl: (1) Received HTTP/0.9 when not allowed
        # [__S7W__]'
        res = scheduler.get_ssh().exec(f"curl {scheduler_addr.replace('tcp://', '')}")
        return res.find("Received HTTP/0.9 when not allowed") > -1

    def restart_dask(self):
        self.get_dask_client().restart(wait_for_workers=False)

    def kill_this(self):
        self.get_ssh().kill_program_by_substr(os.path.dirname(sys.argv[0]))


"""
sshtunnel [-h] [-U SSH_USERNAME] [-p SSH_PORT] [-P SSH_PASSWORD] -R
                 IP:PORT [IP:PORT ...] [-L [IP:PORT ...]] [-k SSH_HOST_KEY]
                 [-K KEY_FILE] [-S KEY_PASSWORD] [-t] [-v] [-V] [-x IP:PORT]
                 [-c SSH_CONFIG_FILE] [-z] [-n] [-d [FOLDER ...]]
                 ssh_address
    S219: Server = Server(ip="your_server_ip",
                          port=20157,
                          username='root',
                          password="your_server_password",
https://sshtunnel.readthedocs.io/en/latest/
sshtunnel your_server_ip -U "root" -p  20157 -P "your_password." -R your_server_ip:55200 -L your_server_ip:55200 -t -v -n
sshpass -p "your_password." ssh -CNg -L 55200:your_server_ip:55200 root@your_server_ip -p 20157

curl  --output - --http0.9 your_server_ip:55200/metrics
"""


@dataclass
class Servers:
    BENCHMARK_SERVER = Server(ip="your_server_ip",
                              port=22,
                              username='xx',
                              password="your_server_password",
                              dask_port=6006,
                              name="LOCAL",
                              dask_worker_port_begin=54800,
                              gpu_type=GPUType.EMPTY,
                              n_gpu_card=0,
                              n_cpu_core=4,
                              python_exec="/usr/local/bin/python"
                              )


class DEVHelper:
    @staticmethod
    def prepare_environment():
        DEVHelper.prepare_100_9()

    @staticmethod
    def prepare_100_9():
        DEVHelper.prepare_env(Server(ip="your_server_ip"))

    @staticmethod
    def prepare_env(server: Server):
        _server = Server(ip=server.ip, port=server.port, username=server.username, password=server.password)
        ssh = _server.get_ssh()

        # todo
        ssh.mkdir(ENV.REMOTE_WORK_HOME.absolute())
        _server.upload_dir(ENV.LOCAL_WORK_HOME, ENV.REMOTE_WORK_HOME,
                           Server.RSYNC_PYTHON_FILTER)
        _server.upload_dir(ENV.LOCAL_PYLIBS_HOME,
                           ENV.REMOTE_PYLIBS_HOME, Server.RSYNC_PYTHON_FILTER)
        ssh.send("bash /root/bin/pylibs_install.sh")

    @staticmethod
    def get_server_1009():
        return Server(ip="your_server_ip")

    @staticmethod
    def get_server_220():
        return Server(ip="your_server_ip")


class DaskDeploy:

    @staticmethod
    def deploy_153():
        scheduler = Servers.S153
        DaskDeploy.deploy_server(scheduler)

    @staticmethod
    def deploy_164():
        scheduler = Servers.S164
        DaskDeploy.deploy_server(scheduler)

    @staticmethod
    def deploy_219():
        scheduler = Servers.S219
        DaskDeploy.deploy_server(scheduler)

    @staticmethod
    def deploy_220():
        scheduler = Servers.S220
        DaskDeploy.deploy_server(scheduler)

    @classmethod
    def deploy_100_9(cls):
        scheduler = Servers.S100_9
        DaskDeploy.deploy_server(scheduler)

    @classmethod
    def deploy_215(cls):
        scheduler = Servers.S215
        DaskDeploy.deploy_server(scheduler)

    @classmethod
    def deploy_server(cls, job_runner: Server):
        """
        Deploy the dask with GPU environment.

        Parameters
        ----------
        job_runner : Server

        Returns
        -------

        """
        # job_runner.prepare_python_packages()
        job_runner.prepare_python_packages()
        job_runner.upload_pylibs()
        job_runner.kill_all()
        job_runner.prepare_scheduler()
        job_runner.prepare_dask_workers_cpu(job_runner, 30000)
        Debug.end()

    @classmethod
    def deploy_server_cpu(cls, job_runner: Server):
        """
        Deploy the dask with CPU environment.

        Parameters
        ----------
        job_runner : Server

        Returns
        -------

        """
        # job_runner.prepare_python_packages()
        job_runner.prepare_python_packages()
        job_runner.upload_pylibs()
        job_runner.kill_all()
        job_runner.prepare_scheduler_cpu()
        job_runner.prepare_dask_workers_cpu(job_runner, dask_worker_port_begin=30000)
        Debug.end()
