import os.path
import time
import traceback

from pylibs._del_dir.experiments.exp_helper import parallel_run_cmd, JobConfV1
from pylibs.config import ExpServerConf, GCF, ServerName
from pylibs.utils.util_bash import BashUtil
from pylibs.utils.util_directory import make_dirs
from pylibs.utils.util_ssh import SSHClient


def get_metrics_name_from_exp_home(exp_home):
    return os.path.basename(exp_home + ".bz2")


class ESC:
    GPU_100_9_LAN = ExpServerConf(ip='your_server_ip',
                                  port="22",
                                  username="root",
                                  password="your_server_password",
                                  runtime="/remote-home/sunwu/cs_acmis_sunwu/runtime/")

    GPU_100_9_WAN = ExpServerConf(
        # ip='your_server_ip',
        # port="12210",
        ip='ct2.frp.vip',
        port="30722",
        username="root",
        password="your_server_password",
        runtime="/remote-home/sunwu/cs_acmis_sunwu/runtime/"
    )

    GPU_10_153_WAN = ExpServerConf(
        # ip='your_server_ip',
        # port="12208",
        ip='ct2.frp.vip',
        port="30720",
        username="root",
        password="your_server_password",
        runtime="/remote-home/acmis_fdd/sw_research_code/runtime/")

    GPU_10_153_LAN = ExpServerConf(
        ip='your_server_ip',
        port="20045",
        username="root",
        password="your_server_password",
        runtime="/remote-home/acmis_fdd/sw_research_code/runtime/"
    )

    CPU_10_219_LAN = ExpServerConf(ip='your_server_ip',
                                   port="20300",
                                   username="root",
                                   password="your_server_password",
                                   runtime="/remote-home/cs_acmis_sunwu/sw_research_code/runtime/")
    CPU_10_219_WAN = ExpServerConf(
        # ip='your_server_ip',
        # port="12202",
        ip='ct2.frp.vip',
        port="30721",
        username="root",
        password="your_server_password",
        runtime="/remote-home/cs_acmis_sunwu/sw_research_code/runtime/"
    )

    LOCAL_HOST = ExpServerConf(ip=None,
                               port=None,
                               username=None,
                               password=None,
                               runtime="/Users/sunwu/Downloads/download_metrics/")


class HOMEConf:
    GPU_10_153 = "/remote-home/acmis_fdd/sw_research_code/runtime/"
    GPU_100_9 = "/remote-home/sunwu/cs_acmis_sunwu/runtime/"
    LOCAL_HOME = "/Users/sunwu/Downloads/download_metrics/"


class ServerConfig:

    def __init__(self, proxy_ip, proxy_port, username, password, real_ip, work_home):
        self.proxy_ip = proxy_ip
        self.proxy_port = proxy_port
        self.username = username
        self.password = password
        self.real_ip = real_ip
        self.work_home = work_home


class RS:

    @staticmethod
    def upload_to_100_9(local, remote):
        # filter = JobConfV1.get_exp_version()[:-1]
        # --debug=FILTER
        obj = RC.DOCKER_GPU_100_9
        cmd = f'sshpass -p "{obj.password}" rsync  -e "ssh -p {obj.proxy_port}" -avi {local}  root@{obj.proxy_ip}:{remote}  '
        BashUtil.run_command_print_progress(cmd)


"""

"""


class RC:
    GPU_10_153 = ServerConfig(**{
        "proxy_ip": "your_server_ip",
        "proxy_port": 12208,
        "username": 'root',
        "password": 'Fdd19950518*',
        "real_ip": "your_server_ip",
        "work_home": "/remote-home/acmis_fdd/sw_research_code/runtime/"
    })
    REMOTE_HOME = "/remote-home/sunwu/cs_acmis_sunwu/sw_research_code/"
    REMOTE_HOME_ENV = "/remote-home/sunwu/cs_acmis_sunwu/sw_research_code/environments/"

    @staticmethod
    def get_ssh_client(conf: ExpServerConf) -> SSHClient:
        sc = SSHClient(conf.ip, conf.port, conf.username, conf.password)
        return sc

    @staticmethod
    def get_client_host_100_9():
        conf = RC.GPU_100_9
        sc = SSHClient(conf.proxy_ip, conf.proxy_port, conf.username, conf.password)
        return sc

    @staticmethod
    def get_client_docker_100_9():
        conf = RC.DOCKER_GPU_100_9
        sc = SSHClient(conf.proxy_ip, conf.proxy_port, conf.username, conf.password)
        return sc

    GPU_100_9 = ServerConfig(**{
        "proxy_ip": "your_server_ip",
        "proxy_port": 22,
        "username": 'root',
        "password": 'acmis',
        "real_ip": "your_server_ip",
        "work_home": "/remote-home/sunwu/cs_acmis_sunwu/sw_research_code/A01_paper_exp"
    })

    DOCKER_GPU_100_9 = ServerConfig(**{
        "proxy_ip": "your_server_ip",
        "proxy_port": 12210,
        "username": 'root',
        "password": 'acmis',
        "real_ip": "your_server_ip",
        "work_home": "/remote-home/sunwu/cs_acmis_sunwu/sw_research_code/A01_paper_exp"
    })
    GPU_100_9 = ServerConfig(**{
        "proxy_ip": "your_server_ip",
        "proxy_port": 22,
        "username": 'root',
        "password": 'your_password',
        "real_ip": "your_server_ip",
        "work_home": "/remote-home/sunwu/cs_acmis_sunwu/runtime/"
    })
    SERVER_GYXY = [ServerConfig(**{
        "proxy_ip": "your_server_ip",
        "proxy_port": 7540,
        "username": 'root',
        "password": 'acmis',
        "real_ip": "your_server_ip",
        "work_home": "/remote-home/sunwu/cs_acmis_sunwu/runtime/"
    }),
                   ServerConfig(**{
                       "proxy_ip": "your_server_ip",
                       "proxy_port": 7541,
                       "username": 'root',
                       "password": 'acmis',
                       "real_ip": "your_server_ip",
                       "work_home": "/remote-home/sunwu/cs_acmis_sunwu/runtime/"
                   }),
                   ServerConfig(**{
                       "proxy_ip": "your_server_ip",
                       "proxy_port": 7542,
                       "username": 'root',
                       "password": 'acmis',
                       "real_ip": "your_server_ip",
                       "work_home": "/remote-home/sunwu/cs_acmis_sunwu/runtime/"
                   }),
                   ServerConfig(**{
                       "proxy_ip": "your_server_ip",
                       "proxy_port": 7543,
                       "username": 'root',
                       "password": 'acmis',
                       "real_ip": "your_server_ip",
                       "work_home": "/remote-home/sunwu/cs_acmis_sunwu/runtime/"
                   }),
                   ServerConfig(**{
                       "proxy_ip": "your_server_ip",
                       "proxy_port": 7544,
                       "username": 'root',
                       "password": 'acmis',
                       "real_ip": "your_server_ip",
                       "work_home": "/remote-home/sunwu/cs_acmis_sunwu/runtime/"
                   }), ]
    SERVER_ACMIS = [
        ServerConfig(**{
            "proxy_ip": "your_server_ip",
            "proxy_port": 12208,
            "username": 'root',
            "password": 'Fdd19950518*',
            "real_ip": "your_server_ip",
            "work_home": "/remote-home/acmis_fdd/sw_research_code/runtime/"
        }),
        ServerConfig(**{
            "proxy_ip": "your_server_ip",
            "proxy_port": 12210,
            "username": 'root',
            "password": 'acmis',
            "real_ip": "your_server_ip",
            "work_home": "/remote-home/sunwu/cs_acmis_sunwu/runtime/"
        }),
        # ServerConfig(**{
        #     "proxy_ip": "your_server_ip",
        #     "proxy_port": 22,
        #     "username": 'root',
        #     "password": 'your_password',
        #     "real_ip": "your_server_ip",
        #     "work_home": "/remote-home/sunwu/cs_acmis_sunwu/runtime/"
        # }),
    ]
    # 代理IP,port, user, pwd, real ip
    GYXY_ALL = [
        ("your_server_ip", 7540, "root", "acmis", "your_server_ip"),
        ("your_server_ip", 7541, "root", "acmis", "your_server_ip"),
        ("your_server_ip", 7542, "root", "acmis", "your_server_ip"),
        ("your_server_ip", 7543, "root", "acmis", "your_server_ip"),
        ("your_server_ip", 7544, "root", "acmis", "your_server_ip"),
    ]
    IP_PORT_USER_PWD_DESC = [
        ("your_server_ip", 7541, "root", "acmis", "your_server_ip"),
        ("your_server_ip", 7542, "root", "acmis", "your_server_ip"),
        ("your_server_ip", 7543, "root", "acmis", "your_server_ip"),
    ]

    IP_PORT_USER_PWD_DESC_54_58 = [
        ("your_server_ip", 7540, "root", "acmis", "your_server_ip"),
        ("your_server_ip", 7544, "root", "acmis", "your_server_ip"),
    ]


def restart_all():
    """
    从59机器上从启所有docker

    Returns
    -------

    """
    sc = SSHClient("your_server_ip", 7510, "root", "acmis@624624624")
    out = []
    for ip, port, user, pwd, real_id in RC.GYXY_ALL:
        out.append(parallel_run_cmd(sc, f"ssh root@{real_id} docker restart tsb_uad"))
    for o in out:
        o.join()


def restart_docker(real_ip):
    sc = SSHClient("your_server_ip", 7510, "root", "acmis@624624624")
    sc.exec(f"ssh root@{real_ip} docker restart tsb_uad")


def restart_52():
    """
    从59机器上从启所有docker

    Returns
    -------

    """
    sc = SSHClient("your_server_ip", 7510, "root", "acmis@624624624")
    sc.exec("ssh root@your_server_ip docker restart tsb_uad")


def restart_58():
    sc = SSHClient("your_server_ip", 7510, "root", "acmis@624624624")
    sc.exec("ssh root@your_server_ip docker restart tsb_uad")


def pre_prepare():
    for ip, port, user, pwd, description in RC.IP_PORT_USER_PWD_DESC:
        print(f"----------------------------{description}------------------------------------")
        sc = SSHClient(ip, port, user, pwd)
        sc.exec(
            "pkill bash && pkill python && pkill screen && pgrep -f 'python' | xargs  -r  kill  && pgrep -f 'bash' | xargs  -r  kill ")
        sc.exec(
            "rm -rf /nohup.log")
        print("-------------------------------end---------------------------------")


def list_running_process():
    for ip, port, user, pwd, real_ip in RC.GYXY_ALL:
        sc = SSHClient(ip, port, user, pwd)
        parallel_run_cmd(sc, "cat ~/fast_progress.txt", end="\n")


def list_usage():
    while True:
        try:
            sc = SSHClient("your_server_ip", 7510, "root", "acmis@624624624")
            for ip, port, user, pwd, real_id in RC.GYXY_ALL:
                cmd = "ssh root@%s docker stats tsb_uad --no-stream --format \"{{.CPUPerc}}\"" % real_id
                parallel_run_cmd(sc, cmd)

        except KeyboardInterrupt:
            pass
        except:
            traceback.print_exc()
        finally:
            time.sleep(5)
            print("", end="\r")


def copy_from_153_to_100_9(exp_name):
    source = os.path.join(ESC.GPU_10_153_LAN.home, exp_name)
    target = os.path.join(ESC.GPU_100_9_LAN.home, exp_name)

    cmd = f'sshpass -p {ESC.GPU_10_153_LAN.password} rsync -avi  -e "ssh -p {ESC.GPU_10_153_LAN.port}" root@{ESC.GPU_10_153_LAN.ip}:{source}/ {target}/'
    ssh = RC.get_client_docker_100_9()
    ssh.mkdir(target)
    ssh.exec(cmd)


def copy_file_on_lan(
        _from=ESC.GPU_10_153_LAN,
        _to=ESC.CPU_10_219_LAN,
        _ssh=GCF.get_server_conf(ServerName.GPU_153_LAN)):
    print("Copy from 153 to 219")
    source = os.path.join(_from.runtime)
    target = os.path.join(_to.runtime)
    # -avi
    cmd = f'sshpass -p {_to.password} rsync -avih --stats  -e "ssh -p {_to.port} -o PubkeyAuthentication=yes   -o stricthostkeychecking=no" -f"+ {JobConfV1.EXP_VERSION}**/***" -f"- *"  {source}  root@{_to.ip}:{target} '
    ssh = RC.get_ssh_client(_ssh)
    ssh.exec(cmd)
    print("Done copy from 153 to 219")


def copy_from_153_to_219(exp_name="",
                         _from=ESC.GPU_10_153_LAN,
                         _to=ESC.CPU_10_219_LAN,
                         _ssh=GCF.get_server_conf(ServerName.GPU_153_LAN)):
    print("Copy from 153 to 219")
    source = os.path.join(_from.runtime)
    target = os.path.join(_to.runtime)
    cmd = f'sshpass -p {_to.password} rsync -avi   -e "ssh -p {_to.port} -o PubkeyAuthentication=yes   -o stricthostkeychecking=no" -f"+ {exp_name}" -f"- *"  {source}  root@{_to.ip}:{target} '
    ssh = RC.get_ssh_client(_ssh)
    ssh.exec(cmd)
    print("Done copy from 153 to 219")


def remote_run_exp(args):
    from pylibs.config import GCF, ServerName
    from pylibs.experiments.exp_helper import JobConfV1
    from pylibs.servers import upload_to_153, upload_to_100_9
    from pylibs.utils.util_log import logconf
    from pylibs.utils.util_thread import UtilThreads

    logconf(f"Entry {args.entry}")
    ut = UtilThreads()

    ut.append_without_args(upload_to_153)
    ut.append_without_args(upload_to_100_9)
    ut.start()

    def run_153():
        ssh = RC.get_ssh_client(GCF.get_server_conf(ServerName.GPU_153_WAN))
        ssh.send("cd /remote-home/acmis_fdd/sw_research_code/A01_paper_exp;conda activate pytorch1.11")
        ssh.send("pkill screen")
        # ssh.send(f"export EXP_VERSION={JobConfV1.EXP_VERSION}")
        ssh.send(f"screen python {args.entry}")

    def run_100_9():
        ssh = RC.get_ssh_client(GCF.get_server_conf(ServerName.GPU_100_9_WAN))
        ssh.send("cd /remote-home/sunwu/cs_acmis_sunwu/sw_research_code/A01_paper_exp;conda activate base")
        ssh.send("pkill screen")
        ssh.send(f"export EXP_VERSION={JobConfV1.EXP_VERSION}")
        ssh.send(f"screen python {args.entry}")

    def run_219():
        ssh = RC.get_ssh_client(GCF.get_server_conf(ServerName.GPU_100_9_WAN))
        ssh.send("cd /remote-home/sunwu/cs_acmis_sunwu/sw_research_code/A01_paper_exp;conda activate base")
        ssh.send("pkill screen")
        ssh.send(f"export EXP_VERSION={JobConfV1.EXP_VERSION}")
        ssh.send(f"screen python {args.entry}")

    ut = UtilThreads()
    ut.append_without_args(run_153)
    ut.append_without_args(run_100_9)
    ut.start()


def copy_from_109_to_219(exp_name="",
                         _from=ESC.GPU_100_9_LAN,
                         _to=ESC.CPU_10_219_LAN,
                         _ssh=GCF.get_server_conf(ServerName.GPU_100_9_LAN)):
    print("Copy from 109 to 219")
    source = os.path.join(_from.runtime)
    target = os.path.join(_to.runtime)
    cmd = f'sshpass -p {_to.password} rsync -avi  -e "ssh -p {_to.port} -o PubkeyAuthentication=yes   -o stricthostkeychecking=no" -f"+ {exp_name}" -f"- *" {source} root@{_to.ip}:{target}'
    ssh = RC.get_ssh_client(_ssh)
    ssh.exec(cmd)
    print("Done copy from 109 to 219")


def download_local_from_100_9(exp_name, local=ESC.LOCAL_HOST, remote=ESC.GPU_100_9_WAN):
    remote_dir = os.path.join(remote.runtime, exp_name)
    local_dir = os.path.join(local.runtime, exp_name)
    make_dirs(local_dir)
    cmd = f'sshpass -p {remote.password} rsync -avi  -e "ssh -p {remote.port}" root@{remote.ip}:{remote_dir}/ {local_dir}/'
    BashUtil.run_command_print_progress(cmd)


if __name__ == '__main__':
    # list_usage()
    copy_from_153_to_100_9("V502_WT_ALL_01")
"""
 sshpass -p acmis rsync  -e "ssh -p 12208" -avi --debug=FILTER --dry-run -f"+ V345\_VUS***" -f"- *" root@your_server_ip:/remote-home/acmis_fdd/sw_research_code/runtime/  /Users/sunwu/Downloads/download_metrics/


"""
