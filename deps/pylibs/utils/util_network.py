import socket
import traceback

from pylibs.utils.util_bash import exec_cmd, exec_cmd_and_return_str

from pylibs.utils.util_log import get_logger, loginfo
from pylibs.utils.util_message import log_warn_msg
from pylibs.utils.util_system import UtilSys
import subprocess

log = get_logger()


def get_host_ip(dns_server='your_server_ip', dns_port=80):
    """
    查询本机ip地址
    Returns
    str
        ip address
    -------

    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((dns_server, dns_port))
        UtilSys.is_debug_mode() and loginfo(s.getsockname())
        ip = s.getsockname()[0]
    except Exception as e:
        log_warn_msg(f"Unable to get IP address, cause: {e}")
        ip = get_host_ip_v2()
    finally:
        s.close()
    return ip


def get_random_idle_port():
    """
    获取一个空闲随机端口

    Returns
    -------

    """
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    _, port = tcp.getsockname()
    tcp.close()
    return port


def is_port_listing(port, host='your_server_ip'):
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect((host, int(port)))
        return True
    except ConnectionRefusedError as e:
        return False
    except Exception as e:
        log.error(f"Error when checkin port is available, cause: {e}")
        return True
    finally:
        if s:
            s.close()


def get_host_ip_v2():
    # https://docs.python.org/3/library/socket.html#socket.getaddrinfo
    try:
        addrs = socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET, proto=socket.IPPROTO_TCP)
        for addr in addrs:
            ip, _ = addr[-1]
            if str(ip).startswith("100"):
                return ip
            elif str(ip).startswith("192"):
                return ip
            else:
                continue
        return default_ip()
    except Exception as e:
        log.error(traceback.format_exc())
        return default_ip()


def default_ip():
    default = "your_server_ip"
    log_warn_msg(f"Getting ip address failed, set as {default}")
    return default


if __name__ == '__main__':
    # print(get_host_ip())
    # assert is_port_listing(1) == False
    # print(f"Host ip: {get_host_ip_v2()}")
    # print(f"Host ip: {get_host_ip()}")
    pass
    # kill_process_on_port(8080)
