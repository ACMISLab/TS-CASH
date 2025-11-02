import joblib  # 新版本 Scikit-learn
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_system import UtilSys

log = get_logger()


def save_sk_model(model, path="model.m"):
    """
    Save sklearn model to path.
    
    Parameters
    ----------
    model : 
    path : str
        a file with extension .m, e.g., if.m,  which represents a sklearn model.

    Returns
    -------

    """
    UtilSys.is_debug_mode() and log.info(f"save model to {path}")
    joblib.dump(model, path)


def load_sk_model(path):
    """
    Load sklearn model with extension .m

    Parameters
    ----------
    path : str
        a file with extension .m, e.g., if.m, which represents a sklearn model.

    Returns
    -------

    """
    return joblib.load(path)
