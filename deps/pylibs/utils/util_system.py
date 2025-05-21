import multiprocessing
import os
import sys


def _is_macos():
    return UtilSys.is_macos()


class UtilSys:
    DEBUG_MODE_FLAG = None
    IS_MACOS = None

    @staticmethod
    def is_macos():
        if UtilSys.IS_MACOS is None:
            if sys.platform == 'darwin':
                UtilSys.IS_MACOS = True
            else:
                UtilSys.IS_MACOS = False
        return UtilSys.IS_MACOS

    @staticmethod
    def get_cpu_num(ratio=1):
        return int(multiprocessing.cpu_count() * ratio)

    @classmethod
    def get_entry_file_name_without_ext(cls, prefix="", data_sample_method=""):
        f_name = str(os.path.basename(sys.argv[0]))
        return str(prefix) + "_" + f_name[:f_name.rfind(".")] + data_sample_method

    @staticmethod
    def get_environ_by_key(key, default=None, raise_error=False):
        """
        返回 键为key 的环境变量.

        如果不存在,返回None
        Parameters
        ----------
        key :

        Returns
        -------

        """
        val = os.environ.get(key)
        if val is None and raise_error:
            raise ValueError(f"Value cant be None for Environ [{key}]")

        return val if val is not None else default

    @staticmethod
    def set_environ(key, value):
        """
        返回 键为key 的环境变量.

        如果不存在,返回None
        Parameters
        ----------
        key :

        Returns
        -------

        """
        os.environ[key] = value
        return os.environ[key]

    @staticmethod
    def is_debug():
        return UtilSys.is_debug_mode()
    @staticmethod
    def is_debug_mode():
        """
        全局控制是否是Debug mode

        使用方式:
        UtilSys.is_debug_mode() and print("DEBUG 信息")

        Parameters
        ----------

        Returns
        -------

        """
        if type(UtilSys.DEBUG_MODE_FLAG) is bool:
            return UtilSys.DEBUG_MODE_FLAG

        if UtilSys.DEBUG_MODE_FLAG is None:
            UtilSys.DEBUG_MODE_FLAG = os.environ.get("PY_DEBUG")

        if UtilSys.DEBUG_MODE_FLAG is None:
            return False
        return UtilSys.DEBUG_MODE_FLAG.strip() == '1'

    @classmethod
    def generate_envs_from_arr(cls, envs: list):
        """
        generate_envs_from_arr
        input:
         envs = [(HPOKeys.DATASET_NAME, dataset_name),
                (HPOKeys.DATA_ID, data_id),
                (HPOKeys.DATA_SAMPLE_RATE, data_id)
                ]
        output:
        export dataset_name=IOPS
        export data_id=KPI-301c70d8-1630-35ac-8f96-bc1b6f4359ea.train.out
        export data_sample_rate=KPI-301c70d8-1630-35ac-8f96-bc1b6f4359ea.train.out



        Parameters
        ----------
        envs :

        Returns
        -------

        """
        str_envs = ""
        for _k, _v in envs:
            str_envs = str_envs + f"export {_k}={_v}\n"
        return str_envs

class SysUtil(UtilSys):
    pass
class US(UtilSys):
    def __init__(self):
        pass


if __name__ == '__main__':
    print(UtilSys.get_environ_by_key("aaa"))
    print(UtilSys.get_environ_by_key("HOME"))
    assert UtilSys.is_debug_mode() is False
    os.environ["PY_DEBUG"] = "1 "
    UtilSys.DEBUG_MODE_FLAG = os.environ["PY_DEBUG"]
    assert UtilSys.is_debug_mode() is True
    os.environ["PY_DEBUG"] = "0"
    UtilSys.DEBUG_MODE_FLAG = os.environ["PY_DEBUG"]
    assert UtilSys.is_debug_mode() is False

    os.environ["PY_DEBUG"] = "1 "
    UtilSys.DEBUG_MODE_FLAG = os.environ["PY_DEBUG"]
    UtilSys.is_debug_mode() and print("DEBUG 信息")
    os.environ["PY_DEBUG"] = "0"
    UtilSys.DEBUG_MODE_FLAG = os.environ["PY_DEBUG"]
    UtilSys.is_debug_mode() and print("不应该执行")
