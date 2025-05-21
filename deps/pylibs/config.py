import json
import os.path
import platform
import pprint
import sys
from pathlib import Path

import distributed


def is_linux():
    return platform.system() == "Linux"


class Env:
    EXP_HOME = "/Users/sunwu/SW-Research/pylibs/exps"
    # this directory c"ontains tailscaled and tailscale
    LIB_WHEEL_PATH_REMOTE = "/tmp/pylibs-1.0.0-py3-none-any.whl"
    FRP_EXE_HOME = "/Users/sunwu/Nextcloud/App/frp/frp_0.37.1_linux_amd64/"

    # http://your_server_ip:7500
    # http://your_server_ip:7500/
    # FRP_SERVER_LOCAL = "your_server_ip"
    TAILSCALE_SAVE_PATH = "/Users/sunwu/Nextcloud/App/tailscale/tailscale_1.60.0_amd64/"
    MACOS_RUNTIME_HOME = "/Users/sunwu/runtime"
    SERVER_RUNTIME_HOME = "/remote-home/cs_acmis_sunwu/runtime"
    LOCAL_PYLIBS_EGG = "/Users/sunwu/SW-Research/pylibs/dist/pylibs-1.0.0-py3.10.egg"
    LOCAL_PYLIBS_WHEEL = "/Users/sunwu/SW-Research/pylibs/dist/pylibs-1.0.0-py3-none-any.whl"
    REMOTE_PYLIBS_HOME: Path = Path("/remote-home/cs_acmis_sunwu/automl/deps/pylibs")
    DEV_PYLIBS_WHEEL = "/remote-home/cs_acmis_sunwu/2024/pylibs/dist/pylibs-1.0.0-py3-none-any.whl"
    DEV_PYLIBS_HOME: str = "/remote-home/cs_acmis_sunwu/2024/p2/pylibs"
    LOCAL_PYLIBS_HOME: Path = Path("/Users/sunwu/SW-OpenSourceCode/AutoML-Benchmark/deps/pylibs/")
    REMOTE_WORK_HOME: Path = Path("/remote-home/cs_acmis_sunwu/2024/p2/")
    LOCAL_WORK_HOME: Path = Path("/Users/sunwu/SW-Research/AutoML")

    @staticmethod
    def get_runtime_home():
        if sys.platform == "darwin":
            # MacOS X
            return Env.MACOS_RUNTIME_HOME
        else:
            # other

            return Env.SERVER_RUNTIME_HOME

    @classmethod
    def mark_as_finished(cls):
        """
        用于统一程序运行的的输出结果，通常用于多程序交互
        Returns
        -------

        """
        msg = "Process is end"
        print(msg)
        return msg

    @classmethod
    def is_process_success(cls, res):
        return str(res).find(cls.mark_as_finished()) > -1

    @classmethod
    def get_frpc_home(cls):
        if platform.system() == "Linux":
            return "/remote-home/cs_acmis_sunwu/app/frp_0.37.1_linux_amd64"
        else:
            return Env.FRP_EXE_HOME

    @classmethod
    def get_pylibs_home(cls):
        if platform.system() == "Linux":
            return Env.REMOTE_PYLIBS_HOME
        else:
            return Env.LOCAL_PYLIBS_HOME

    @staticmethod
    def get_wheel_path():
        if platform.system() == "Linux":
            return Env.DEV_PYLIBS_WHEEL
        else:
            return Env.LOCAL_PYLIBS_WHEEL

    @staticmethod
    def get_egg_path():
        if platform.system() == "Linux":
            raise NotImplementedError("Not get_egg_path")
        else:
            return Env.LOCAL_PYLIBS_EGG

    @classmethod
    def get_supervisord_exe(cls):
        if is_linux():
            return "/root/miniconda3/bin/supervisord"
        else:

            return "/Users/sunwu/miniforge3/envs/autosklearn/bin/supervisord"

    @staticmethod
    def get_python_exe():
        if is_linux():
            # return "/root/miniconda3/bin/python"
            return "/root/miniconda3/envs/autosklearn/bin/python"
        else:
            # return "/Users/sunwu/miniforge3/envs/amltk/bin/python"
            return "/Users/sunwu/miniforge3/envs/autosklearn/bin/python"

    @staticmethod
    def get_exp_results_save_home():
        if is_linux():
            return "/remote-home/cs_acmis_sunwu/experiment_results"
        else:
            return "/Users/sunwu/Documents/experiment_results"

    @staticmethod
    def get_frpc_exe_path():
        if is_linux():
            return "/usr/local/bin/frpc"
        else:
            return "/Users/sunwu/Nextcloud/App/npc/frp_0.37.1_darwin_amd64/frpc"

    @classmethod
    def get_script_home(cls):
        if is_linux():
            return Path("/remote-home/cs_acmis_sunwu/2024/pylibs/exps").as_posix()
        else:
            return Path("/Users/sunwu/SW-Research/pylibs/exps").as_posix()


class ExpServerConf:
    def __init__(self, ip, port, username, password, runtime=os.path.join("~", "runtime"), work_home=None):
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.runtime = runtime
        if work_home is None:
            work_home = runtime.split("runtime")[0]
        self.work_home = work_home

    def __str__(self):
        return pprint.pformat(self.__dict__)


class ServerName:
    LOCAL = "LOCAL"

    GPU_153_WAN = "GPU_153_WAN"
    GPU_153_LAN = "GPU_153_LAN"

    GPU_100_9_WAN = "GPU_100_9_WAN"
    GPU_100_9_LAN = "GPU_100_9_LAN"

    GPU_219_WAN = "GPU_219_WAN"
    GPU_219_LAN = "GPU_219_LAN"

    GPU_100_9 = "gpu_100_9"
    MYSQL_WAN = "MYSQL_WAN"
    CYBERDB_LAN = "CYBERDB_LAN"


class ServerNameV2:
    REDIS_219 = "redis"
    BACKUP_SEVER = "backup_server"
    REDIS_LOCAL = "redis_local"


class FRPType:
    CT2 = "ct2"
    MYSELF = "ali"


class GlobalConfig:
    pass


class GCF(GlobalConfig):
    # FRP_TYPE = FRPType.CT2
    FRP_TYPE = FRPType.MYSELF

    @staticmethod
    def get_server_conf(name=ServerName.GPU_219_WAN) -> ExpServerConf:
        return GCFV2.get_server_conf(name)


class GCFV2(GlobalConfig):
    # FRP_TYPE = FRPType.CT2
    FRP_TYPE = FRPType.MYSELF

    _CONF = None

    @staticmethod
    def get_server_conf(name: str = ServerName.GPU_219_WAN) -> ExpServerConf:
        if GCFV2._CONF is None:
            with open(os.path.join(os.path.dirname(__file__), "server_config.json"), "r") as f:
                GCFV2._CONF = json.load(f)

        _key = None
        if name.lower().find("219") > -1:
            _key = "10_219"

        if name.lower().find("153") > -1:
            _key = "10_153"

        if name.lower().find("100_9") > -1:
            _key = "100_9"
        if name.lower().find("local") > -1:
            _key = "local"

        assert _key is not None, f"Unknown type [{name}]!"

        _type = "wan"
        if name.lower().endswith("lan"):
            _type = "lan"

        _conf = GCFV2._CONF[_key]
        _ip_conf = _conf[_type]

        return ExpServerConf(ip=_ip_conf['ip'],
                             port=_ip_conf['port'],
                             username=_conf['username'],
                             password=_conf['password'],
                             runtime=_conf['runtime'],
                             work_home=_conf['work_home']
                             )


class GCFV3:
    # FRP_TYPE = FRPType.CT2
    FRP_TYPE = FRPType.MYSELF

    _CONF = None

    @staticmethod
    def get_server_conf(name: str = ServerNameV2.REDIS_219, net_type="lan") -> ExpServerConf:
        if GCFV2._CONF is None:
            with open(os.path.join(os.path.dirname(__file__), "server_config.json"), "r") as f:
                GCFV2._CONF = json.load(f)
        try:
            _conf = GCFV2._CONF[name]
            _ip_conf = _conf[net_type]

            return ExpServerConf(ip=_ip_conf['ip'],
                                 port=_ip_conf['port'],
                                 username=_conf['username'],
                                 password=_conf['password'],
                                 runtime=_conf['runtime'],
                                 work_home=_conf['work_home']
                                 )
        except:
            raise RuntimeError(f"server [{name}] with type [{net_type}] is not existed")


class CONFIG_DB_100_111:
    CONFIG_DB_HOST = "your_server_ip"
    CONFIG_DB_PORT = 9201
    CONFIG_DB_USER = "root"
    CONFIG_DB_PASSWORD = "your_password."
    CONFIG_DB_DATABASE = 'nni_experiments'


config_feishu_webhook_url = "https://open.feishu.cn/open-apis/bot/v2/hook/2190ea80-fb70-483d-b10e-7bf9ac29f37b"


def config_webhook_of_feishu():
    """
    Get the webhook url of feishu.

    Returns
    -------
    str

    """
    return config_feishu_webhook_url


class CONF:
    DATASET_NAME = 'dataset_name'
    DATA_ID = 'data_id'
    SCRIPT_HOME = Env.get_script_home()

    @staticmethod
    def get_main_path():
        return Path(CONF.SCRIPT_HOME) / "main.py"

    @staticmethod
    def get_search_space_path():
        return Path(CONF.SCRIPT_HOME) / "e_cash_libs.py"


class Debug:
    @staticmethod
    def err_found():
        distributed.print("❌❌❌❌❌Error found❌❌❌❌❌")

    @classmethod
    def end(cls):
        distributed.print("✅" * 30)

    @classmethod
    def errmsg(cls, msg):
        distributed.print("❌" * 10)
        distributed.print(msg)
        distributed.print("❌" * 10)


if __name__ == '__main__':
    # print(GCFV2.get_server_conf(ServerName.GPU_219_LAN))
    # print(GCFV3.get_server_conf())
    print(Debug.errmsg("jlsjdf"))
