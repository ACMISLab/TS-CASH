import subprocess
import traceback
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_system import UtilSys

log = get_logger()


def automatic_memory_decition_by_gpu_total_memory(total_memory):
    # 2080TI: 11019
    # A6000: 49140
    # 11019/3500=3.14  -> 24
    # 49140/3500=14.04  -> 14*8=112
    if total_memory >= 40000:
        # A100,A6000
        return 1024 * 6
    else:
        # 2080TI
        return 1024 * 3


class UGPU:
    @staticmethod
    def get_available_memory(gpu_index):
        """
        Returns the available memory for the given GPU exp_index.
        The units is MB.

        Parameters
        ----------
        gpu_index :

        Returns
        -------

        """
        if UtilSys.is_macos():
            UtilSys.is_debug_mode() and log.info("Skip set GPU memory since in MacOS!")
            return -1
        try:
            command = f"nvidia-smi --exec-gpu=memory.free --format=csv,noheader,nounits --id={gpu_index}"
            UtilSys.is_debug_mode() and log.info("Get GPU available memory command: [{}]".format(command))
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                log.error("Get GPU available memory failed since"
                          f"[{stderr.decode().strip()}]")
                return -1
            else:
                return float(stdout.decode().strip())
        except Exception as e:

            log.error(traceback.format_exc())
            return -1


if __name__ == '__main__':
    UGPU.get_available_memory(0)
