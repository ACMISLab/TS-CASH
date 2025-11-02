
from pylibs.utils.util_servers import Server


def set_timezone(server:Server,default_timezone="Asia/Shanghai"):
  """将时区设置为伤害时区

  Args:
      server (Server): _description_
      default_timezone (str, optional): _description_. Defaults to "Asia/Shanghai".
  """
  ssh=server.get_ssh()
  ssh.exec("timedatectl set-timezone {}".format(default_timezone))
  res=ssh.exec("timedatectl")
  assert res.find("Time zone: {} (CST, +0800)".format(default_timezone)) > -1 or f"timezone for {server.ip} is not correct"