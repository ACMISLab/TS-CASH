#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/2/29 09:00
# @Author  : gsunwu@163.com
# @File    : util_dask.py
# @Description:

from dask.distributed import LocalCluster, Client

from pylibs.utils.util_servers import Servers, Server


class DaskCluster:
    TYPE_215 = "S215"
    TYPE_ALL = "all"
    TYPE_100_9 = "100.9"
    TYPE_220 = "S220"
    TYPE_219 = "S219"
    TYPE_153 = "S153"
    TYPE_LOCAL = 'local'
    TYPE_164 = "164"
    TYPE_None = "None"

    @staticmethod
    def get_cluster_local():
        return get_local_cluster()

    @staticmethod
    def get_cluster_220():
        return get_cluster_a100_220()

    @staticmethod
    def get_cluster_219():
        return get_cluster_a100_219()

    @staticmethod
    def get_cluster_100_9():
        return get_cluster_100_9()

    @classmethod
    def get_cluster(cls, type_: str):
        match type_:
            case cls.TYPE_220:
                return cls.get_cluster_220()
            case cls.TYPE_219:
                return cls.get_cluster_219()
            case cls.TYPE_LOCAL:
                return cls.get_cluster_219()
            case cls.TYPE_100_9:
                return cls.get_cluster_100_9()
            case _:
                raise ValueError("Unsupported cluster type")

    @classmethod
    def check_gpu_avaliable(cls, client: Client):
        def gpu():
            import tensorflow as tf
            return tf.test.gpu_device_name()

        flag = False
        results = client.run(gpu)
        for res in results.values():
            if str(res).find("GPU") > -1:
                print(f"✅GPU is avaliable: {res}")
                flag = True
        if flag:
            return True
        else:
            raise ValueError("❌ Not found available GPU device;")

    @classmethod
    def get_server(cls, client_type) -> Server:
        match client_type:
            case cls.TYPE_220:
                return Servers.S220
            case cls.TYPE_219:
                return Servers.S219
            case cls.TYPE_100_9:
                return Servers.S100_9
            case cls.TYPE_LOCAL:
                return Servers.LOCAL
            case cls.TYPE_164:
                return Servers.S164
            case cls.TYPE_ALL:
                return Servers.DASK_ALL
            case cls.TYPE_215:
                return Servers.S215
            case _:
                raise ValueError("Unsupported cluster type")


def get_local_cluster_client(host="localhost", port=6006):
    return get_local_cluster()


def get_local_cluster():
    cluster = LocalCluster(n_workers=1,
                           threads_per_worker=1,
                           host="localhost",
                           name="SWLocalClient")
    client = cluster.get_client()
    return client


def get_cluster(host="your_server_ip", port=20219):
    # def get_cluster_220(host="your_server_ip",port=12002):
    """
    220 服务器
    your_server_ip
    12002
    """
    try:
        client = Client(address=f"tcp://{host}:{port}")
        return client
    except:
        raise RuntimeError(f"Could not connect to cluster {host}!")


def get_cluster_a100_220():
    """
     服务 服务端口	校园网访问	互联网访问
     ssh服务 	22	your_server_ip:20246	your_server_ip:24843
     vnc服务 	5901	your_server_ip:20247	your_server_ip:24848
     tensorboard 	6006	your_server_ip:20248	your_server_ip:24849
     Jupyter Notebook 	8888	your_server_ip:20249	your_server_ip:24850
    """
    return get_cluster("your_server_ip", 20248)


def get_cluster_a100_219():
    """
     ssh服务 	22	your_server_ip:20153	your_server_ip:24839
     vnc服务 	5901	your_server_ip:20154	your_server_ip:24840
     tensorboard 	6006	your_server_ip:20155	your_server_ip:24841
     Jupyter Notebook 	8888	your_server_ip:20156	your_server_ip:24842
    """
    return get_cluster("your_server_ip", 20155)


def get_cluster_100_9():
    return get_cluster(host="your_server_ip", port=6006)


def get_cluster_gyxy_cpu():
    """
    贵阳学院 CPU 集群
    Returns
    -------

    """
    return get_cluster(host="your_server_ip", port=6006)


if __name__ == '__main__':
    # print(get_local_cluster_client())
    # print(is_port_listing(9090))
    # print(get_local_cluster_client())
    # time.sleep(30)
    # get_cluster_220()
    pass
