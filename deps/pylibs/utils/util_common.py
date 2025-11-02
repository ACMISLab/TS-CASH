import datetime
import os.path
import sys
import tempfile
import traceback

from pylibs.utils.util_datatime import get_datatime
from pylibs.utils.util_directory import make_dirs


class UtilComm:
    is_init = False

    @staticmethod
    def get_system_runtime():
        runtime_dir = os.path.join(os.path.expanduser("~"), "runtime")
        make_dirs(runtime_dir)
        if UtilComm.is_init is False:
            UtilComm.is_init = True
        return runtime_dir

    @staticmethod
    def get_joblib_cache_dir():
        home = os.path.join(tempfile.gettempdir(), "joblib_cache")
        make_dirs(home)
        return home

    @staticmethod
    def get_backup_dir():
        home = os.path.join(tempfile.gettempdir(), "backup")
        make_dirs(home)
        return home

    @staticmethod
    def get_runtime_directory():
        home = os.path.join(tempfile.gettempdir(), "runtime")
        make_dirs(home)
        return home

    @staticmethod
    def get_entrance_directory():
        home = os.path.abspath(os.path.dirname(sys.argv[0]))
        make_dirs(home)
        return home

    @staticmethod
    def get_file_name(filename, home=os.path.abspath(os.path.dirname(sys.argv[0]))):
        home = UtilComm.get_runtime_directory(home)
        make_dirs(home)
        return os.path.join(home, filename)

    @staticmethod
    def get_workdir(workdir, home=os.path.abspath(os.path.dirname(sys.argv[0]))):
        _dir = os.path.join(UtilComm.get_runtime_directory(home), workdir)
        make_dirs(_dir)
        return _dir

    @staticmethod
    def get_filename_entry():
        # return os.path.basename(sys.argv[0])[:-3]
        return UtilComm.get_runtime_directory()

    @classmethod
    def mark_as_finished(cls):
        """
        Mention that the process is finished.
        Returns
        -------

        """
        print()
        print("+" * 80)

        print("{:*^80}".format(f"Script [{os.path.basename(sys.argv[0])}] is mark_as_finished at {get_datatime()}"))
        print("+" * 80)
        print()

    @staticmethod
    def print(msg):
        try:
            from dask.distributed import print
            print(msg)
        except:
            print(msg)
            traceback.print_exc()


class UC(UtilComm):
    pass


class U(UtilComm):
    pass


if __name__ == '__main__':
    print(UtilComm.get_filename_entry())
    UC.done()
