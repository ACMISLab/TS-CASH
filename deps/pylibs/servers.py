from pylibs.config import ServerName, GCF
from pylibs.utils.util_rsync import Rsync


def upload_to_219():
    conf = GCF.get_server_conf(ServerName.GPU_219_WAN)
    Rsync.upload(conf, local=GCF.get_server_conf(ServerName.LOCAL).work_home)


def upload_to_100_9():
    conf = GCF.get_server_conf(ServerName.GPU_100_9_WAN)
    Rsync.upload(conf, local=GCF.get_server_conf(ServerName.LOCAL).work_home)


def upload_to_100_9_lan():
    conf = GCF.get_server_conf(ServerName.GPU_100_9_LAN)
    Rsync.upload(conf, local=GCF.get_server_conf(ServerName.LOCAL).work_home)


def upload_to_153():
    conf = GCF.get_server_conf(ServerName.GPU_153_WAN)
    Rsync.upload(conf, local=GCF.get_server_conf(ServerName.LOCAL).work_home)


if __name__ == '__main__':
    upload_to_219()
