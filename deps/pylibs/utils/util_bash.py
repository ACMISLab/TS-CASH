import locale
import os
import pprint
import re
import subprocess
import sys
import time
import traceback

from rich import print

from pylibs.utils.util_log import get_logger
from pylibs.utils.util_system import UtilSys

log = get_logger()


class CMD:
    @staticmethod
    def exe_cmd(command: str, home=None, check=False, timeout=None):
        """
        Timeout in seconds.

        Parameters
        ----------
        command :
        home :
        check :
        timeout : int
            Timeout in seconds.

        Returns
        -------

        """

        if command.find("nohup") > -1:
            if command.find("2>&1") == -1:
                raise ValueError("When nohup is existed, 2>&1 must exist. For example >/dev/null 2>&1")

        if home is not None:
            command = f"cd {home};{command}"
        log.debug(f"Exec command: \n{command} ")

        var = subprocess.run([f"{command}"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             timeout=timeout, check=check)
        if var.returncode != 0:
            return var.stderr.strip().decode("utf-8")
        else:
            res = var.stdout.strip().decode("utf-8")
            return res

    @staticmethod
    def run(command):
        return CMD.run_command_print_progress(command)

    @staticmethod
    def exec(command):
        return CMD.run_command_print_progress(command)

    @staticmethod
    def run_command_print_progress(command):
        """
        实时输出执行结果
        Parameters
        ----------
        cmd :

        Returns
        -------

        """

        if command.find("nohup") > -1:
            if command.find("2>&1") == -1:
                raise ValueError("When nohup is existed, 2>&1 must exist. For example >/dev/null 2>&1")
        log.debug(f"CMD: {command}\n")
        os.system(command)
        # process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # encoding = locale.getpreferredencoding()
        # outputs = []
        # while True:
        #     try:
        #         output = process.stdout.readline()
        #         if output == b'' and process.poll() is not None:
        #             break
        #         if output:
        #             msg = output.strip()
        #             outputs.append(msg.decode(encoding))
        #             print(outputs[-1])
        #     except:
        #         pass
        # rc = process.poll()
        # if rc != 0:
        #     print("---- Exec command error: ", process.stderr.readlines(), file=sys.stderr)
        #     return None
        # process.stdout.close()
        # process.stderr.close()
        # return "\n".join(outputs)


class BashUtil(CMD):
    pass


class UtilBash(CMD):
    pass


class UtilCMD(CMD):
    pass


def get_bash_result(command, check=True):
    var = subprocess.run([f"{command}"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
                         timeout=10, check=check)

    ret = re.sub('\\x1b', '', var.stdout)
    return ret.split("\n")


def exec_cmd(command, check=False, print_result=False, split_to_array=True):
    """
    Exec command.

    ret stdout,stderr
    Parameters
    ----------
    split_to_array :bool
    print_result : bool
    show_command : bool
        Decided whether to show the exec command.
    command : str
        The command to execute
    check : bool
        The check option in subprocess.run

    Returns
    -------
    list
        stdout
    list
        stderr
    """
    UtilSys.is_debug_mode() and log.info(f"Exec command: \n{pprint.pformat(command)}")
    var = subprocess.run([f"{command}"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         encoding="utf-8", timeout=60, check=check)
    stdout = precess_std_results(var.stdout)
    stderr = precess_std_results(var.stderr)
    if var.returncode != 0:
        log.error(f"Exec command err:\n{stderr}")
    else:
        UtilSys.is_debug_mode() and log.info(f"Exec command success:\n{stderr}")

    if split_to_array:
        return stdout.split("\n"), stderr.split("\n")
    else:
        return stdout, stderr


def precess_std_results(res):
    return re.sub('\\x1b', '', res).replace("[0m", "").replace("[31m", "").replace("[39m", "")


def exec_cmd_return_stdout_and_stderr(command, retry=10, retry_interval=3, check=True, timeout=60):
    """
    Exec command.

    ret stdout,stderr
    Parameters
    ----------
    retry : int
        How many times to retry when command is failed
    command : str
        The command to execute
    retry_interval: int
        The interval between each retry

    Returns
    -------
    str
        stdout + stderr
    """
    for i in range(retry):
        try:
            UtilSys.is_debug_mode() and log.info(f"Exec command: \n{pprint.pformat(command)}")
            var = subprocess.run(
                [f"{command}"],
                shell=True,
                encoding="utf-8",
                timeout=timeout,
                check=check,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE
            )
            stdout = re.sub('\\x1b', '', var.stdout).replace("[0m", "")
            stderr = re.sub('\\x1b', '', var.stderr).replace("[0m", "")
            return stdout, stderr
        except Exception as e:
            log.warning(f"Execute command [{command}] failed, retry {i}/{retry} after {retry_interval} seconds."
                        f"\n Caused by {e}")
            traceback.print_exc()
            time.sleep(retry_interval)
    errmsg = f"Execute [{command}] error after retry {retry} times"
    log.error(errmsg)
    raise RuntimeError(errmsg)


def exec_cmd_and_return_str(command, retry=10, retry_interval=3, check=True, timeout=60):
    stdout, stderr = exec_cmd_return_stdout_and_stderr(command=command, retry=retry, retry_interval=retry_interval,
                                                       check=check, timeout=timeout)
    return stdout + stderr


def clear_cmd_output(output):
    return re.sub('\\x1b', '', output).replace("[0m", "").replace("\n", "")


class Bash(CMD):
    pass


if __name__ == '__main__':
    # CMD.run_command_print_progress("xping www.baidu.com")
    # CMD.run_command_print_progress("ping www.baidu.com")
    # Bash.run("ping www.baidu.com")
    Bash.run_command_print_progress("ls -l")
