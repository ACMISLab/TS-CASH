import os
import re
from typing import Union

from pylibs.utils.util_log import get_logger
from pylibs.utils.util_nni_mamager_status import NNIManagerStatus

log = get_logger()


class NNIManagerLog:

    def __init__(self, experiment_working_directory, id, file_base_name='nnimanager.log'):
        """

        Parameters
        ----------
        file_base_name : str
            The file name of nnimanager.log, default nnimanager.log
        experiment_working_directory : str
            The location for nni-experiments, default is  `~/nni-experiments`
        id : str
            the experiment id
        """
        self.experiment_working_directory = experiment_working_directory
        self.experiment_id = id
        self.file_base_name = file_base_name

        # Represents the file /Users/sunwu/nni-experiments/test_exp/qc85ane1/log/nnimanager.log
        self.file_nni_manager_log = os.path.join(self.experiment_working_directory, self.experiment_id, "log",
                                                 self.file_base_name)
        UtilSys.is_debug_mode() and log.info(f"nnimanager file path: {self.file_nni_manager_log}")

        # Represents the contents of file /Users/sunwu/nni-experiments/test_exp/qc85ane1/log/nnimanager.log
        self.file_content = None
        with open(self.file_nni_manager_log, "r") as f:
            self.file_content = f.read()

    def is_experiment_done(self) -> bool:
        """
        Get the status from /Users/sunwu/nni-experiments/test_exp/qc85ane1/log/nnimanager.log

        Returns
        -------
        bool
            True if experiment is mark_as_finished, False otherwise.
        """

        # Where the nnimanager.log:
        # "/Users/sunwu/nni-experiments/debug_1675855568130661000/qc85ane1/log"

        if re.search(NNIManagerStatus.NNI_MANAGER_LOG_STATUS_DONE, self.file_content, re.S):
            UtilSys.is_debug_mode() and log.info(f"The experiment {self.experiment_id} is mark_as_finished")
            return True
        else:
            UtilSys.is_debug_mode() and log.info(f"The experiment {self.experiment_id} is not finished")
            return False

    def is_experiment_stopped(self):
        #     Experiment stopped
        if re.search(NNIManagerStatus.NNI_MANAGER_LOG_STATUS_STOP, self.file_content, re.S):
            UtilSys.is_debug_mode() and log.info(f"The experiment {self.experiment_id} is stopped")
            return True
        else:
            UtilSys.is_debug_mode() and log.info(f"The experiment {self.experiment_id} is not stopped")
            return False

    def is_experiment_error(self):
        """
        Get the status from /Users/sunwu/nni-experiments/test_exp/qc85ane1/log/nnimanager.log


        Returns
        -------
        bool
            True if experiment is error, False otherwise.
        """

        # Where the nnimanager.log:
        # "/Users/sunwu/nni-experiments/debug_1675855568130661000/qc85ane1/log"

        if re.search(NNIManagerStatus.NNI_MANAGER_LOG_STATUS_ERROR, self.file_content, re.S):
            log.error(f"The experiment {self.experiment_id} is error. \n {self.file_content}")
            return True
        else:
            UtilSys.is_debug_mode() and log.info(f"The experiment {self.experiment_id} not contained the error. ")
            return False

    def get_nni_namager_file_name(self):
        return self.file_nni_manager_log

    def get_nni_namager_file_content(self):
        return self.file_content

    def has_error(self):
        """
        Error  example:

        [2023-02-13 17:55:36] DEBUG (tuner_command_channel.WebSocketChannel) Silent error: Error: tuner_command_channel: Tuner closed connection
        at WebSocket.handleWsClose (/Users/sunwu/SW-Research/nni-2.10-mac-m1/ts/nni_manager/dist/core/tuner_command_channel/websocket_channel.js:83:26)
        at WebSocket.emit (node:events:538:35)
        at WebSocket.emitClose (/Users/sunwu/SW-Research/nni-2.10-mac-m1/ts/nni_manager/node_modules/express-ws/node_modules/ws/lib/websocket.js:246:10)
        at Socket.socketOnClose (/Users/sunwu/SW-Research/nni-2.10-mac-m1/ts/nni_manager/node_modules/express-ws/node_modules/ws/lib/websocket.js:1127:15)
        at Socket.emit (node:events:526:28)
        at TCP.<anonymous> (node:net:687:12)
        Returns
        -------

        """
        if re.search("error", self.file_content, re.S):
            log.error(f"The experiment {self.experiment_id} is error. \n {self.file_content}")
            return True
        else:
            UtilSys.is_debug_mode() and log.info(f"The experiment {self.experiment_id} not contained the error. ")
            return False

    def get_fail_trials(self) -> Union[None, list]:
        """
        Check whether all the trails is successful.


        list if there are failure trails.
        None if all trails are successful.

        Returns
        -------


        """

        #  trialJob status update: TZKX8, FAILED
        trial_reg = re.findall("trialJob\s*status\s*update:\s*(\w+),\s*FAILED",
                               self.file_content, re.S)
        # trial_reg=['XhEIh', 'TZKX8', 'DSLwy']
        if trial_reg:
            log.error(f"The experiment {self.experiment_id} is error. \n {self.file_content}")
            return trial_reg
        else:
            UtilSys.is_debug_mode() and log.info(f"The experiment {self.experiment_id} not contained the error. ")
            return None
