class NNIManagerStatus:
    """
    This class represents the `status string`  of nni experiment.

    E.g., The experiment with id=qc85ane1 is mark_as_finished if qc85ane1/log/nnimanager.log contains
        NNI_MANAGER_LOG_STATUS_DONE = "INFO \(NNIManager\) Experiment mark_as_finished", where NNI_MANAGER_LOG_STATUS_DONE is a regrex for
        searching the string which indicates that the experiment was mark_as_finished.
    """
    NNI_MANAGER_LOG_STATUS_STOP = "Experiment stopped"
    NNI_MANAGER_LOG_STATUS_DONE = "INFO \(NNIManager\) Experiment mark_as_finished"
    NNI_MANAGER_LOG_PORT_REGEX = "http://your_server_ip:(\d+)"
    NNI_MANAGER_LOG_STATUS_ERROR = "INFO \(NNIManager\) Change NNIManager status from: RUNNING to: ERROR"

    NNICTL_CREATE_SUCCESS = "Setting up..."

    # error when nnictl create xxx
    NNICTL_CREATE_ERROR = "raise ValueError"

    # To stop experiment run "nnictl stop dou24mj6" or "nnictl stop --all"
    NNICTL_CREATE_SUCCESS_EXP_ID_REGEX = "To stop experiment run \"nnictl stop (.*)\" or \"nnictl stop --all\""
    NNICTL_CREATE_SUCCESS_EXP_PORT_REGEX = "Web portal URLs: http://your_server_ip:(\d+)"
