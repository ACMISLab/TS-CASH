import sys

from pylibs.config import GCFV3, ServerNameV2, ExpServerConf
from pylibs.utils.util_system import UtilSys
from pylibs.utils.util_log import get_logger

log = get_logger()


class RedisUtil:
    _POOL = None
    """
    install in ubuntu:
    
    apt-get update
    apt-get install redis-server
    ps -ef | grep redis
    
    
    # 配置远程访问
    cat >/etc/myredis.conf <<EOF
    port 5901
    protected-mode no
    requirepass your_password.
    EOF
    pkill redis-server
    screen redis-server /etc/myredis.conf --loglevel verbose
    
    
    # 测试
    redis-cli -p 5901 
    """

    def __init__(self, conf: ExpServerConf = None):
        import redis
        if conf is None:
            if UtilSys.is_macos():
                conf = GCFV3.get_server_conf(ServerNameV2.REDIS_LOCAL, net_type="wan")
            else:
                conf = GCFV3.get_server_conf(ServerNameV2.REDIS_219, net_type="wan")

        if RedisUtil._POOL is None:
            print(f"Connect to {conf.ip}:{conf.port}")
            RedisUtil._POOL = redis.ConnectionPool(host=conf.ip, port=conf.port, decode_responses=True,
                                                   password=conf.password)

        self.rd = redis.Redis(connection_pool=RedisUtil._POOL)

    def set(self, key, value):
        UtilSys.is_debug_mode() and log.info(f"Set or update radis at: [{key}] ")
        return self.rd.set(key, value)  # 设置 name 对应的值

    def get(self, key):
        """
        If not key, return None
        Parameters
        ----------
        key :

        Returns
        -------

        """
        UtilSys.is_debug_mode() and log.info(f"Get value of key: [{key}]")
        return self.rd.get(key)

    def exist(self, key):
        """
        记录存在返回True,
        不存在返回False

        Parameters
        ----------
        key :

        Returns
        -------

        """
        return self.rd.exists(key) > 0

    def keys(self, keys):
        """
        模糊匹配所有键位 keys 的结果.

        例如:
        keys: v01*
        结果: 返回 v01 开头的所有结果: ['v0120_1', 'v0120_2']

        Parameters
        ----------
        keys :
            模糊匹配的表达式,如 v01*

        Returns
        -------

        """
        return self.rd.mget(self.rd.keys(keys))

    def __del__(self):
        self.rd.close()


if __name__ == '__main__':
    rd = RedisUtil()
    print(rd.exist("0095454308847588d1fe2e5a4518f24b0fdec000"))
    print(rd.exist("1"))
    rd.set("aa", "jsdklfjsldjfsl")
    print(rd.get("aa"))

    print(rd.get("a5f482c21d9e6247693ccf2da2fbf05e9668cd8c"))
    print(rd.get("b"))
    assert rd.get("sdfxsdf2") is None

    rd = RedisUtil()
    rd.get("b")

    print(rd.exist('aa'))
    print(rd.exist('b'))
