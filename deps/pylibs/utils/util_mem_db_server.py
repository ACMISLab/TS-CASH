# pip install --upgrade pip
# pip install cyberdb
import time
import cyberdb
from pylibs.config import GCF, ServerName
from pylibs.utils.util_system import UtilSys


def start_cyberdb_server():
    db = cyberdb.Server()
    # 数据持久化，备份文件为 data.cdb，备份周期 900 秒一次。
    db.set_backup('data.cdb', cycle=900)
    conf = GCF.get_server_conf(ServerName.CYBERDB_LAN)
    # 设置 TCP 地址、端口号、密码，生产环境中密码建议使用大小写字母和数字的组合。
    # start_or_restart 方法不会阻塞运行，若希望该操作阻塞，请使用 run 方法代替 start_or_restart，参数不变。
    db.run(host=conf.ip, port=conf.port, password=conf.password, print_log=True)


if __name__ == '__main__':
    start_cyberdb_server()
