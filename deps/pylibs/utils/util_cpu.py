import multiprocessing

from pylibs.utils.util_log import get_logger

log = get_logger()


class UtilCPU:
    @staticmethod
    def get_n_cpu():
        """
        获取cpu核心数

        Returns
        -------

        """
        core_count = multiprocessing.cpu_count()
        return core_count


if __name__ == '__main__':
    print(UtilCPU.get_n_cpu())
