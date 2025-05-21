import os
import re

from pylibs.utils.util_log import get_logger

log = get_logger()


class NNIDispatcherLog:

    def __init__(self, experiment_working_directory, id, file_base_name='dispatcher.log'):
        """

        Parameters
        ----------
        file_base_name : str
            The file name of dispatcher.log, default dispatcher.log. E.g.,
            /Users/sunwu/nni-experiments/test_exp/qc85ane1/log/dispatcher.log
        experiment_working_directory : str
            The location for nni-experiments, default is  `~/nni-experiments`.
            An example can be py-search-lib/pylibs/test/data/nnilog/err_connect_to_nni_server
        id : str
            the experiment id
        """
        self.experiment_working_directory = experiment_working_directory
        self.experiment_id = id
        self.file_base_name = file_base_name

        # Represents the file /Users/sunwu/nni-experiments/test_exp/qc85ane1/log/dispatcher.log
        self.file_nni_manager_log = os.path.join(self.experiment_working_directory, self.experiment_id, "log",
                                                 self.file_base_name)
        UtilSys.is_debug_mode() and log.info(f"dispatcher file path: {self.file_nni_manager_log}")

        # Represents the contents of file /Users/sunwu/nni-experiments/test_exp/qc85ane1/log/dispatcher.log
        self.file_content = None
        with open(self.file_nni_manager_log, "r") as f:
            self.file_content = f.read()

    def has_error(self) -> bool:
        """
        Get the status from /Users/sunwu/nni-experiments/test_exp/qc85ane1/log/dispatcher.log


        在等待实验运行的时候, 判断 dispatcher.log 中是否存在 Report error to NNI manager / ConnectionRefusedError: [Errno 111] Connect call failed , 如果存在, 则重启实验

        添加判断: 如果实验结束了, 那么停止实验

        Returns
        -------
        bool
            True if experiment is mark_as_finished, False otherwise.
        """

        # Where the dispatcher.log:
        # "/Users/sunwu/nni-experiments/debug_1675855568130661000/qc85ane1/log"

        if re.search("Report error to NNI manager", self.file_content, re.S):
            UtilSys.is_debug_mode() and log.info(f"The experiment {self.experiment_id} is mark_as_finished")
            return True
        else:
            UtilSys.is_debug_mode() and log.info(f"The experiment {self.experiment_id} is not finished")
            return False
