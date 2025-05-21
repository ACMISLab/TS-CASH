#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/8/1 14:01
# @Author  : gsunwu@163.com
# @File    : libs.py
# @Description:
import dataclasses
import multiprocessing
import resource
import signal
import tempfile
import warnings
from enum import Enum

import psutil
import sklearn
from numpy import load

from datasets.openml.clean_data import preproc_data
from pylibs.utils.util_class import ClassUtil
from pylibs.utils.util_datatime import get_str_datetime
from pylibs.utils.util_numpy import enable_numpy_reproduce
from pylibs.kvdb_mysql import KVDBMySQL
from smac import Scenario
from tshpo.lib_func import *
from tshpo.tshpo_common import TSHPOCommon

assert sklearn.__version__ == "1.4.2", "sklearn.__version__ must be 1.4.2"
warnings.filterwarnings("ignore")
import asyncio
import json
import os.path
import time
import traceback
import warnings
import ConfigSpace as CS
import autosklearn
import autosklearn.classification
import distributed
import numpy as np  # linear algebra
import pathlib
from scipy.stats import norm
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, Constant
from sklearn.utils.multiclass import type_of_target
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from autosklearn.constants import MULTILABEL_CLASSIFICATION, MULTICLASS_CLASSIFICATION, BINARY_CLASSIFICATION
from autosklearn.data.validation import InputValidator
from autosklearn.pipeline.components.classification import ClassifierChoice
from autosklearn.pipeline.components.data_preprocessing.rescaling.standardize import StandardScalerComponent
from autosklearn.util.stopwatch import StopWatch
from pylibs.utils.util_openai import UtilOpenai
from dataclasses import fields
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from autosklearn.metrics import roc_auc
from smac.initial_design.random_design import RandomInitialDesign
from pylibs.utils.util_rsync import Rsync
from smac import HyperbandFacade, RandomFacade, HyperparameterOptimizationFacade
from smac.runhistory import TrialValue, TrialInfo
from tshpo.lib_class import *

PROJECT_HOME = os.environ["TSHPO_HOME"]
log = get_log()

DATASET_HOME = os.path.join(PROJECT_HOME, "deps/datasets/openml")


class OptMethod(Enum):
    SMAC = "smac"
    SOAR = 'soar'
    Hyperband = "hyperband"
    BayesianOptimization = "bo"
    Random = "rs"


# 无效的数据

@dataclass
class DelRunJobDel:
    """
    定义一个待运行的任务
    """
    X_train: np.ndarray
    y_train: np.ndarray

    X_test: np.ndarray
    y_test: np.ndarray
    # 精度指标，如f1，acc，recall
    metric: str

    # 模型的超参数配置
    config: Configuration

    # 超参数搜索空间
    cs: ConfigurationSpace

    debug: bool

    seed: int

    job_index: int = None

    exp_conf: ExpConf = None

    def __post_init__(self):
        assert self.X_train.shape[0] == self.y_train.shape[0]
        assert self.X_test.shape[0] == self.y_test.shape[0]

    def get_dict(self):
        return {
            "job_config": dict(self.config),
            "job_seed": self.seed,
            "job_opt_metric": self.metric,
            "job_cs_size": len(self.cs),
            "job_x_train.shape": self.X_train.shape,
            "job_x_test.shape": self.X_test.shape,
            "job_index": self.job_index,
            "econf": self.exp_conf.__dict__
        }


def is_macos():
    """
    是否是 macos 系统
    Returns
    -------

    """
    return sys.platform == 'darwin'


class GlobalConfig:
    OUTPUT_HOME = pathlib.Path("./outputs")


# class Analysis:
#     @staticmethod
#     def calc_baseline(file_name):
#         df=pd.read_csv(file_name)
#         df.

class UtilityFunction():
    """An object to compute the acquisition functions.

    Copy from bayesian-optimization 1.5.1

    Parameters
    ----------
    kind: {'ucb', 'ei', 'poi'}
        * 'ucb' stands for the Upper Confidence Bounds method
        * 'ei' is the Expected Improvement method
        * 'poi' is the Probability Of Improvement criterion.

    kappa: float, optional(default=2.576)
            Parameter to indicate how closed are the next parameters sampled.
            Higher value = favors spaces that are least explored.
            Lower value = favors spaces where the regression function is
            the highest.

    kappa_decay: float, optional(default=1)
        `kappa` is multiplied by this factor every iteration.

    kappa_decay_delay: int, optional(default=0)
        Number of iterations that must have passed before applying the
        decay to `kappa`.

    xi: float, optional(default=0.0)
    """

    def __init__(self, kind='ucb', kappa=2.576, xi=0, kappa_decay=1, kappa_decay_delay=0):
        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay

        self.xi = xi

        self._iters_counter = 0

        if kind not in ['ucb', 'ei', 'poi']:
            err = "The utility function " \
                  f"{kind} has not been implemented, " \
                  "please choose one of ucb, ei, or poi."
            raise NotImplementedError(err)
        self.kind = kind

    def update_params(self):
        """Update internal parameters."""
        self._iters_counter += 1

        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay

    def utility(self, x, gp, y_max):
        """Calculate acquisition function.

        Parameters
        ----------
        x : np.ndarray
            Parameters to evaluate the function at.

        gp : sklearn.gaussian_process.GaussianProcessRegressor
            A gaussian process regressor modelling the target function based on
            previous observations.

        y_max : number
            Highest found value of the target function.


        Returns
        -------
        Values of the acquisition function
        """
        if self.kind == 'ucb':
            return self.ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self.ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self.poi(x, gp, y_max, self.xi)
        raise ValueError(f"{self.kind} is not a valid acquisition function.")

    @staticmethod
    def ucb(x, gp, kappa):
        r"""Calculate Upper Confidence Bound acquisition function.

        Similar to Probability of Improvement (`UtilityFunction.poi`), but also considers the
        magnitude of improvement.
        Calculated as

        .. math::
            \text{UCB}(x) = \mu(x) + \kappa \sigma(x)

        where :math:`\Phi` is the CDF and :math:`\phi` the PDF of the normal
        distribution.

        Parameters
        ----------
        x : np.ndarray
            Parameters to evaluate the function at.

        gp : sklearn.gaussian_process.GaussianProcessRegressor
            A gaussian process regressor modelling the target function based on
            previous observations.

        y_max : number
            Highest found value of the target function.

        kappa : float, positive
            Governs the exploration/exploitation tradeoff. Lower prefers
            exploitation, higher prefers exploration.


        Returns
        -------
        Values of the acquisition function
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std

    @staticmethod
    def ei(x, gp, y_max, xi):
        r"""Calculate Expected Improvement acqusition function.

        Similar to Probability of Improvement (`UtilityFunction.poi`), but also considers the
        magnitude of improvement.
        Calculated as

        .. math::
            \text{EI}(x) = (\mu(x)-y_{\text{max}} - \xi) \Phi\left(
                \frac{\mu(x)-y_{\text{max}} -  \xi }{\sigma(x)} \right)
                  + \sigma(x) \phi\left(
                    \frac{\mu(x)-y_{\text{max}} -  \xi }{\sigma(x)} \right)

        where :math:`\Phi` is the CDF and :math:`\phi` the PDF of the normal
        distribution.

        Parameters
        ----------
        x : np.ndarray
            Parameters to evaluate the function at.

        gp : sklearn.gaussian_process.GaussianProcessRegressor
            A gaussian process regressor modelling the target function based on
            previous observations.

        y_max : number
            Highest found value of the target function.

        xi : float, positive
            Governs the exploration/exploitation tradeoff. Lower prefers
            exploitation, higher prefers exploration.


        Returns
        -------
        Values of the acquisition function
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        a = mean - y_max - xi
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def poi(x, gp, y_max, xi):
        r"""Calculate Probability of Improvement acqusition function.

        Calculated as

        .. math:: \text{POI}(x) = \Phi\left( \frac{\mu(x)-y_{\text{max}} -  \xi }{\sigma(x)} \right)

        where :math:`\Phi` is the CDF of the normal distribution.

        Parameters
        ----------
        x : np.ndarray
            Parameters to evaluate the function at.
        gp : sklearn.gaussian_process.GaussianProcessRegressor
            A gaussian process regressor modelling the target function based on
            previous observations.

        y_max : number
            Highest found value of the target function.

        xi : float, positive
            Governs the exploration/exploitation tradeoff. Lower prefers
            exploitation, higher prefers exploration.


        Returns
        -------
        Values of the acquisition function

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi) / std
        return norm.cdf(z)


__DASK_CLIENT__ = None


def get_dask_local_client():
    global __DASK_CLIENT__
    if __DASK_CLIENT__ is None:
        __DASK_CLIENT__ = Client(
            address=f"your_server_ip:6006",
            direct_to_workers=True,
            timeout=30)
        __DASK_CLIENT__.upload_file("./automl_libs.py")
    return __DASK_CLIENT__


def get_dask_local_del():
    # 设置worker的数量，为cpu的1/3
    n_workers = int(os.cpu_count() / 3)
    client = LocalCluster(n_workers=n_workers,
                          threads_per_worker=1,
                          memory_limit="6GiB",
                          processes=True,
                          dashboard_address=":8888",
                          host="your_server_ip").get_client()
    log.debug(f"Dask client: {client}")

    return client


class Steps:
    # 开始贝叶斯优化
    DEFAULT = "_stopwatch_"
    ALL = "_stopwatch_"
    BO_OPTIMIZATION = "bo_optimization"
    # 数据处理
    DATA_PROCESSING = "data_processing"
    # 模型训练
    MODEL_TRAINING = "model_training"

    # 模型predict
    MODEL_PREDICTOIN = "model_predictoin"

    # 模型预选择
    WARM_UP_SELECTION = "warm_up_selection"

    # warm up
    BO_WARM_UP = "bo_warm_up"

    # 数据抽样
    DATA_SAMPLING = "loadamd_splinta_saampling"

    # pca 主成分分析的时间
    PCA = "pca"

    # 数据降维所用的实际爱你 dimensionality reduction algorithm
    DIM_REDU = "dim_reduction"

    # 什么算法开始优化时都可以用这个，例如bo，hyperband
    OPTIMIZATION = "optimization"

    @staticmethod
    def keys():
        #
        return ClassUtil.get_class_attribute_names(Steps)
        # all_attributes = dir(Steps)
        # return [attr for attr in all_attributes if not callable(getattr(Steps, attr)) and not attr.startswith("__")]
        # return [Steps.DEFAULT, Steps.BO_OPTIMIZATION, Steps.DATA_PROCESSING, Steps.MODEL_TRAINING,
        #         Steps.MODEL_PREDICTOIN, Steps.WARM_UP_SELECTION, Steps.BO_WARM_UP, Steps.DATA_SAMPLING,Steps.PCA]


class CpuTimeoutException(Exception):
    """Pynisher exception object returned on a CPU time limit."""
    pass


class TimeoutException(Exception):
    """Pynisher exception object returned when hitting the time limit."""
    pass


class MemorylimitException(Exception):
    """Pynisher exception object returned when hitting the memory limit."""
    pass


class SubprocessException(Exception):
    """Pynisher exception object returned when receiving an OSError while
    executing the subprocess."""
    pass


class PynisherError(Exception):
    """Pynisher exception object returned in case of an internal error.

    This should not happen, please open an issue at github.com/automl/pynisher
    if you run into this."""
    pass


class SignalException(Exception):
    """Pynisher exception object returned in case of a signal being handled by
    the pynisher"""
    pass


class AnythingException(Exception):
    """Pynisher exception object returned if the function call closed
    prematurely and no cause can be determined.

    In this case, the stdout and stderr can contain helpful debug information.
    """
    pass


# create the function the subprocess can execute
def subprocess_func(func, pipe, logger, mem_in_mb, cpu_time_limit_in_s, wall_time_limit_in_s, num_procs,
                    grace_period_in_s, tmp_dir, *args, **kwargs):
    # simple signal handler to catch the signals for time limits
    def handler(signum, frame):
        # logs message with level debug on this logger
        logger.debug("signal handler: %i" % signum)
        if (signum == signal.SIGXCPU):
            # when process reaches soft limit --> a SIGXCPU signal is sent (it normally terminats the process)
            raise (CpuTimeoutException)
        elif (signum == signal.SIGALRM):
            # SIGALRM is sent to process when the specified time limit to an alarm function elapses (when real or clock time elapses)
            logger.debug("timeout")
            raise (TimeoutException)
        else:
            logger.debug("other: %d", signum)
            raise SignalException

    # temporary directory to store stdout and stderr
    if tmp_dir is not None:
        logger.debug(
            'Redirecting output of the function to files. Access them via the stdout and stderr attributes of the wrapped function.')

        stdout = open(os.path.join(tmp_dir, 'std.out'), 'a', buffering=1)
        sys.stdout = stdout

        stderr = open(os.path.join(tmp_dir, 'std.err'), 'a', buffering=1)
        sys.stderr = stderr

    # catching all signals at this point turned out to interfer with the subprocess (e.g. using ROS)
    signal.signal(signal.SIGALRM, handler)
    signal.signal(signal.SIGXCPU, handler)
    signal.signal(signal.SIGQUIT, handler)

    # code to catch EVERY catchable signal (even X11 related ones ... )
    # only use for debugging/testing as this seems to be too intrusive.
    """
    for i in [x for x in dir(signal) if x.startswith("SIG")]:
        try:
            signum = getattr(signal,i)
            print("register {}, {}".format(signum, i))
            signal.signal(signum, handler)
        except:
            print("Skipping %s"%i)
    """

    # set the memory limit
    if mem_in_mb is not None:
        # byte --> megabyte
        mem_in_b = mem_in_mb * 1024 * 1024
        # the maximum area (in bytes) of address space which may be taken by the process.

        # fix current limit exceeds maximum limit
        if sys.platform != "darwin":
            resource.setrlimit(resource.RLIMIT_AS, (mem_in_b, mem_in_b))

    # for now: don't allow the function to spawn subprocesses itself.
    # resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    # Turns out, this is quite restrictive, so we don't use this option by default
    if num_procs is not None:
        resource.setrlimit(resource.RLIMIT_NPROC, (num_procs, num_procs))

    # schedule an alarm in specified number of seconds
    if wall_time_limit_in_s is not None:
        signal.alarm(wall_time_limit_in_s)

    if cpu_time_limit_in_s is not None:
        # From the Linux man page:
        # When the process reaches the soft limit, it is sent a SIGXCPU signal.
        # The default action for this signal is to terminate the process.
        # However, the signal can be caught, and the handler can return control
        # to the main program. If the process continues to consume CPU time,
        # it will be sent SIGXCPU once per second until the hard limit is reached,
        # at which time it is sent SIGKILL.
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_limit_in_s, cpu_time_limit_in_s + grace_period_in_s))

    # the actual function call
    try:
        logger.debug("call function")
        return_value = ((func(*args, **kwargs), 0))
        logger.debug("function returned properly: {}".format(return_value))
    except MemoryError:
        return_value = (None, MemorylimitException)

    except OSError as e:
        return_value = (None, SubprocessException, e.errno)

    except CpuTimeoutException:
        return_value = (None, CpuTimeoutException)

    except TimeoutException:
        return_value = (None, TimeoutException)

    except SignalException:
        return_value = (None, SignalException)

    finally:
        try:
            logger.debug("return value: {}".format(return_value))

            pipe.send(return_value)
            pipe.close()

        except:  # noqa
            # this part should only fail if the parent process is alread dead, so there is not much to do anymore :)
            pass
        finally:
            # recursively kill all children
            p = psutil.Process()
            for child in p.children(recursive=True):
                child.kill()


class enforce_limits(object):
    """
    modified from pynisher
    """

    def __init__(self, mem_in_mb=None, cpu_time_in_s=None, wall_time_in_s=None, num_processes=None,
                 grace_period_in_s=None, logger=None, capture_output=False, context=None):

        if context is None:
            self.context = multiprocessing.get_context()
        else:
            self.context = context

        self.mem_in_mb = mem_in_mb
        self.cpu_time_in_s = cpu_time_in_s
        self.num_processes = num_processes
        self.wall_time_in_s = wall_time_in_s
        self.grace_period_in_s = 0 if grace_period_in_s is None else grace_period_in_s
        self.logger = logger if logger is not None else self.context.get_logger()
        self.capture_output = capture_output

        if self.mem_in_mb is not None:
            self.logger.debug("Restricting your function to {} mb memory.".format(self.mem_in_mb))
        if self.cpu_time_in_s is not None:
            self.logger.debug("Restricting your function to {} seconds cpu time.".format(self.cpu_time_in_s))
        if self.wall_time_in_s is not None:
            self.logger.debug("Restricting your function to {} seconds wall time.".format(self.wall_time_in_s))
        if self.num_processes is not None:
            self.logger.debug("Restricting your function to {} threads/processes.".format(self.num_processes))
        if self.grace_period_in_s is not None:
            self.logger.debug("Allowing a grace period of {} seconds.".format(self.grace_period_in_s))

    def __call__(self, func):

        class function_wrapper(object):
            def __init__(self2, func):
                self2.func = func
                self2._reset_attributes()

            def _reset_attributes(self2):
                self2.result = None
                self2.exit_status = None
                self2.resources_function = None
                self2.resources_pynisher = None
                self2.wall_clock_time = None
                self2.stdout = None
                self2.stderr = None

            def __call__(self2, *args, **kwargs):

                self2._reset_attributes()

                # create a pipe to retrieve the return value
                parent_conn, child_conn = self.context.Pipe(False)
                # import pdb; pdb.set_trace()

                if self.capture_output:
                    tmp_dir = tempfile.TemporaryDirectory()
                    tmp_dir_name = tmp_dir.name

                else:
                    tmp_dir_name = None

                # create and start the process
                subproc = self.context.Process(
                    target=subprocess_func,
                    name="pynisher function call",
                    args=(
                             self2.func,
                             child_conn,
                             self.logger,
                             self.mem_in_mb,
                             self.cpu_time_in_s,
                             self.wall_time_in_s,
                             self.num_processes,
                             self.grace_period_in_s,
                             tmp_dir_name
                         ) + args,
                    kwargs=kwargs,
                )
                self.logger.debug("Function called with argument: {}, {}".format(args, kwargs))

                # start the process

                start = time.time()
                subproc.start()
                child_conn.close()

                try:
                    def read_connection():
                        connection_output = parent_conn.recv()
                        if len(connection_output) == 2:
                            self2.result, self2.exit_status = connection_output
                        elif len(connection_output) == 3:
                            self2.result, self2.exit_status, self2.os_errno = connection_output
                        else:
                            self2.result, self2.exit_status = (None, PynisherError)

                    # read the return value
                    if (self.wall_time_in_s is not None):
                        if parent_conn.poll(self.wall_time_in_s + self.grace_period_in_s):
                            read_connection()
                        else:
                            subproc.terminate()
                            self2.exit_status = TimeoutException
                            # 时间超过了，直接报异常
                            raise asyncio.exceptions.TimeoutError()

                    else:
                        read_connection()

                except EOFError:  # Don't see that in the unit tests :(
                    self.logger.debug(
                        "Your function call closed the pipe prematurely -> Subprocess probably got an uncatchable signal.")
                    self2.exit_status = AnythingException

                finally:
                    self2.resources_function = resource.getrusage(resource.RUSAGE_CHILDREN)
                    self2.resources_pynisher = resource.getrusage(resource.RUSAGE_SELF)
                    self2.wall_clock_time = time.time() - start
                    self2.exit_status = 5 if self2.exit_status is None else self2.exit_status

                    # recover stdout and stderr if requested
                    if self.capture_output:
                        out_file = os.path.join(tmp_dir.name, 'std.out')
                        try:
                            with open(out_file, 'r') as fh:
                                self2.stdout = fh.read()
                        except Exception as e:
                            self.logger.error(
                                f"Cannot recover the output from {out_file} due to {e}")
                        err_file = os.path.join(tmp_dir.name, 'std.err')
                        try:
                            with open(os.path.join(tmp_dir.name, 'std.err'), 'r') as fh:
                                self2.stderr = fh.read()
                        except Exception as e:
                            self.logger.error(
                                f"Cannot recover the output from {err_file} due to {e}")

                        tmp_dir.cleanup()

                    # don't leave zombies behind
                    subproc.join()
                    # exitcode is only available after join
                    self2.exitcode = subproc.exitcode
                return (self2.result)

        return (function_wrapper(func))


def sample_training_data(X_train: np.ndarray, y_train: np.ndarray, num: int, random_state=42):
    """
    返回抽样后的训练数据大小
    Parameters
    ----------
    X_train :
    y_train :

    Returns
    -------

    """
    np.random.seed(random_state)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    n_total = X_train.shape[0]

    if num >= n_total:
        return X_train, y_train
    else:
        train_sample_index = np.random.choice(n_total, num)
        return X_train[train_sample_index], y_train[train_sample_index]


def get_forbidden_models_clause_bak(important_hpys, cs):
    """
    禁止important_hpys.models 中没有选用的那些模型参与后续的超参数优化，目的是减少超参数空间
    Parameters
    ----------
    important_hpys :
    cs :

    Returns
    -------

    """

    forbidden_models = []

    # 获取所有的模型，如果模型没有被正确选择，就加入forbidden models中
    # cs.get("__choice__").choices 共有16个值
    for _model in cs.get("__choice__").choices:
        if _model not in important_hpys.models:
            forbidden_models.append(_model)

    model_forbidden_clause = CS.ForbiddenInClause(cs.get("__choice__"),
                                                  forbidden_models
                                                  )

    #     forbidden_model_clause=ConfigSpace.ForbiddenInClause(important_cs.get("__choice__"),
    #                                   ['adaboost', 'bernoulli_nb', 'decision_tree', 'extra_trees',
    #                                    'gaussian_nb', 'gradient_boosting', 'k_nearest_neighbors',
    #                                    'lda', 'liblinear_svc', 'libsvm_svc', 'mlp',
    #                                    'multinomial_nb', 'passive_aggressive', 'qda', 'sgd']
    #                                   )
    return model_forbidden_clause


def narrow_search_space(important_hpys, cs: ConfigurationSpace):
    """
    减少超参数空间。禁止important_hpys.models 中没有选用的那些模型参与后续的超参数优化，目的是减少超参数空间
    Parameters
    ----------
    important_hpys :
    cs :

    Returns
    -------

    """
    cs = copy.deepcopy(cs)
    # 更新模型的默认值为当前超参数的值，修复下面这个问题：
    # ConfigSpace.exceptions.ForbiddenValueError: Given vector violates forbidden clause Forbidden: __choice__ == 'random_forest'
    cs.get("__choice__").default_value = important_hpys.models[0]
    # 获取所有的模型，如果模型没有被正确选择，就加入forbidden models中
    # cs.get("__choice__").choices 共有16个值

    forbidden_models = []

    # 获取所有的模型，如果模型没有被正确选择，就加入forbidden models中
    # cs.get("__choice__").choices 共有16个值
    for _model in cs.get("__choice__").choices:
        if _model not in important_hpys.models:
            forbidden_models.append(_model)
    try:
        model_forbidden_clause = CS.ForbiddenInClause(cs.get("__choice__"),
                                                      forbidden_models)
        cs.add_forbidden_clause(model_forbidden_clause)
    except:
        traceback.print_exc()
        raise RuntimeError("Forbidden clause error, maybe you forbidden the default value. For example,"
                           "the default value of __choice__ is random_forest, but you forbidden the random_forest value,"
                           "so it raises a value. You can chang the default of __choice__ by cs.get('__choice__').default_value = important_hpys.models[0].")
    # code for debugging:
    # for _model in cs.get("__choice__").choices:
    #     if _model not in important_hpys.models:
    #         # forbidden_models.append(_model)
    #         try:
    #             _fb=CS.ForbiddenEqualsClause(cs.get("__choice__"),_model)
    #             cs.add_forbidden_clause(_fb)
    #         except:
    #             traceback.print_exc()
    #             raise RuntimeError("Forbidden clause error, maybe you forbidden the default value. For example,"
    #                                "the default value of __choice__ is random_forest, but you forbidden the random_forest value,"
    #                                "so it raises a value")
    #     else:
    #         # 参数不需要禁用，什么也不错
    #         pass
    return cs


def get_max_slop_change(history: np.ndarray, his_count=3) -> typing.Union[None, float]:
    """
    计算精度的最大变化量。计算方法：$max(\{loss_i-loss_{i-1}\}_{1}^{HIS})$，HIS 为考虑的历史训练数据数量

    思路：如果最近的3次的模型精度的slop (斜率）变化小于 0.001, 那么就认为找到了合适的数据量

    Parameters
    ----------
    history : 训练历史
    his_count : int
        考虑的历史数据的梳理

    Returns
    -------
    float or np.inf

    """
    #
    # assert len(history) >= his_count, "History must be greater than his_count."
    if len(history) < his_count:
        return None

    con_history = history[-his_count:]
    diff = con_history[1:] - con_history[:-1]
    value = round(max(np.abs(diff)), 6)
    return value


def find_optimized_sample_size_over_dataset(X_train: pd.DataFrame,
                                            y_train: pd.DataFrame,
                                            X_test: pd.DataFrame,
                                            y_test: pd.DataFrame,
                                            sample_step: int,
                                            seed: int,
                                            threshold: float = 0.0005):
    """

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.DataFrame
    X_test : pd.DataFrame
    y_test : pd.DataFrame
    sample_step : int
        抽样的步骤，start=sample_step
    seed : int
        随机种子
    threshold : float
        变化率 slop 停止的阈值
    Returns
    -------

    """
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    losses = []
    for _samp_num in np.arange(start=sample_step, stop=X_train.shape[0] + sample_step + 1, step=sample_step):
        _samp_num = np.min([_samp_num, X_train.shape[0]])
        if _samp_num == X_train.shape[0]:
            # 已经到了全量数据量，直接返回
            return _samp_num
        else:
            train_sample_index = X_train.sample(n=_samp_num, random_state=seed).index
        clf = clf.fit(X_train.loc[train_sample_index], y_train.loc[train_sample_index])
        predict = clf.predict(X_test)
        roc_auc = roc_auc_score(y_test, predict)
        losses.append(roc_auc)

        slop = get_max_slop_change(np.asarray(losses))
        if slop is not None and slop < threshold:
            return _samp_num
        else:
            continue

    return X_train.shape[0]


def find_optimized_data_over_dataset_with_valid_data(X_train_ori: pd.DataFrame,
                                                     y_train_ori: pd.DataFrame,
                                                     sample_step: int,
                                                     seed: int,
                                                     threshold: float = 0.0005):
    """
    返回最佳数据量
   X_train_sampled, y_train_sampled, X_valid, y_valid = find_optimized_data_over_dataset(pd.DataFrame(X_train), pd.DataFrame(y_train), sample_step=512, seed=econf.random_state, threshold=0.0005)

    Parameters
    ----------
    X_train_ori : pd.DataFrame
    y_train_ori : pd.DataFrame
    sample_step : int
        抽样的步骤，start=sample_step
    seed : int
        随机种子
    threshold : float
        变化率 slop 停止的阈值
    Returns
    -------

    """
    from sklearn import tree
    X_train, X_valid, y_train, y_valid = \
        train_test_split(X_train_ori, y_train_ori, random_state=seed, test_size=0.2)

    clf = tree.DecisionTreeClassifier()
    losses = []
    for _samp_num in np.arange(start=sample_step, stop=X_train.shape[0] + sample_step + 1, step=sample_step):
        _samp_num = np.min([_samp_num, X_train.shape[0]])
        if _samp_num == X_train.shape[0]:
            # 已经到了全量数据量，直接返回
            return X_train, y_train, X_valid, y_valid
        else:
            train_sample_index = X_train.sample(n=_samp_num, random_state=seed).index

        # 抽样
        X_train_sampled, y_train_sampled = X_train.loc[train_sample_index], y_train.loc[train_sample_index]
        clf = clf.fit(X_train_sampled, y_train_sampled)
        predict = clf.predict(X_valid)
        roc_auc = roc_auc_score(y_valid, predict)
        losses.append(roc_auc)

        slop = get_max_slop_change(np.asarray(losses))
        if slop is not None and slop < threshold:
            return X_train_sampled, y_train_sampled, X_valid, y_valid
        else:
            continue

    return X_train, y_train, X_valid, y_valid


def find_optimized_data_over_dataset(X_train: pd.DataFrame,
                                     y_train: pd.DataFrame,
                                     X_test: pd.DataFrame,
                                     y_test: pd.DataFrame,
                                     sample_step: int,
                                     seed: int,
                                     threshold: float = 0.0005):
    """
    返回最佳数据量
   X_train_sampled, y_train_sampled, X_valid, y_valid = find_optimized_data_over_dataset(pd.DataFrame(X_train), pd.DataFrame(y_train), sample_step=512, seed=econf.random_state, threshold=0.0005)

    Parameters
    ----------
    X_train_ori : pd.DataFrame
    y_train_ori : pd.DataFrame
    sample_step : int
        抽样的步骤，start=sample_step
    seed : int
        随机种子
    threshold : float
        变化率 slop 停止的阈值
    Returns
    -------

    """
    from sklearn import tree

    clf = tree.DecisionTreeClassifier()
    losses = []
    for _samp_num in np.arange(start=sample_step, stop=X_train.shape[0] + sample_step + 1, step=sample_step):
        _samp_num = np.min([_samp_num, X_train.shape[0]])
        if _samp_num == X_train.shape[0]:
            # 已经到了全量数据量，直接返回
            return X_train, y_train, X_test, y_test
        else:
            train_sample_index = X_train.sample(n=_samp_num, random_state=seed).index

        # 抽样
        X_train_sampled, y_train_sampled = X_train.loc[train_sample_index], y_train.loc[train_sample_index]
        clf = clf.fit(X_train_sampled, y_train_sampled)
        predict = clf.predict(X_test)
        roc_auc = roc_auc_score(y_test, predict)
        losses.append(roc_auc)

        slop = get_max_slop_change(np.asarray(losses))
        if slop is not None and slop < threshold:
            return X_train, y_train, X_test, y_test
        else:
            continue

    return X_train, y_train, X_test, y_test


def loss_over_sample_name(dataset_name, sample_num, fold_index, n_fold=5, seed=42):
    """
    在不同大小的训练数据集上的模型损失，目前只支持auc_roc
    """
    distributed.print(f"start job: {locals()}")
    from sklearn import tree
    losses = []
    for X_train, y_train, X_test, y_test, _fold_index in load_dataset_at_fold(dataset_name=dataset_name,
                                                                              n_fold=n_fold,
                                                                              seed=seed):

        # X_train: pd.DataFrame
        if fold_index == _fold_index:
            # 在训练数据中抽样
            if sample_num >= X_train.shape[0]:
                train_sample_index = X_train.index
            else:
                train_sample_index = X_train.sample(n=sample_num, random_state=seed).index

            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(X_train.loc[train_sample_index], y_train.loc[train_sample_index])
            predict = clf.predict(X_test)
            roc_auc = roc_auc_score(y_test, predict)
            losses.append({
                'dataset': dataset_name,
                'samples': sample_num,
                'roc_auc': roc_auc,
                'fold_index': fold_index
            })
        else:
            continue
    if len(losses) > 0:
        return pd.DataFrame(losses)
    return None


def loss_over_sample_rate(dataset_name, sample_rate, fold_index, n_fold=5, seed=42):
    """
    在不同大小的训练数据集上的模型损失，目前只支持auc_roc
    """
    if sample_rate == 0 or sample_rate > 1:
        return None
    try:
        distributed.print(f"start job: {locals()}")
        from sklearn import tree
        losses = []
        for X_train, y_train, X_test, y_test, _fold_index \
                in load_dataset_at_fold(dataset_name=dataset_name, n_fold=n_fold, seed=seed):

            # X_train: pd.DataFrame
            if fold_index == _fold_index:
                # 在训练数据中抽样
                train_sample_index = X_train.sample(frac=sample_rate, random_state=seed).index

                clf = tree.DecisionTreeClassifier()
                clf = clf.fit(X_train.loc[train_sample_index], y_train.loc[train_sample_index])
                predict = clf.predict(X_test)
                roc_auc = roc_auc_score(y_test, predict)
                losses.append({
                    'dataset': dataset_name,
                    'sample_rate': sample_rate,
                    'roc_auc': roc_auc,
                    'fold_index': fold_index
                })
            else:
                continue
        if len(losses) > 0:
            return pd.DataFrame(losses)
    except:

        return None


def get_auto_sklearn_classification_search_space(y_train, random_state=42,
                                                 exclude=None, include=None) -> ConfigurationSpace:
    """
    获取AutoSKlearn的超参数搜索空间

    Parameters
    ----------
    y_train :

    Returns
    -------

    """
    if exclude is None:
        exclude = ['gradient_boosting', 'passive_aggressive']
    dataset_properties, feat_type = get_classification_dataset_properties(y_train=y_train)
    cfc = ClassifierChoice(dataset_properties=dataset_properties)
    # gradient_boosting 在linux 上会卡住，所以禁用
    if include:
        config = cfc.get_hyperparameter_search_space(feat_type=feat_type, include=include)
    else:
        config = cfc.get_hyperparameter_search_space(feat_type=feat_type, exclude=exclude)

    config.seed(random_state)
    return config


def get_classification_dataset_properties(y_train):
    """
    获取 AutoSKlearn 的超参数搜索空间和配置
    dataset_properties, feat_type = get_classification_dataset_properties(y_train=y_train)
    cfc = ClassifierChoice(dataset_properties=dataset_properties)
    config = cfc.get_hyperparameter_search_space(feat_type=feat_type).get_default_configuration()


    Parameters
    ----------
    y_train :

    Returns
    -------

    """
    input_validator = InputValidator(
        is_classification=True,
        feat_type=None,
        allow_string_features=False,
    )
    y_task = type_of_target(y_train)
    _feat_type = input_validator.feature_validator.feat_type
    task_type = y_task

    multilabel = False
    multiclass = False
    sparse = False

    if task_type == MULTILABEL_CLASSIFICATION:
        multilabel = True
    if task_type == MULTICLASS_CLASSIFICATION:
        multiclass = True
    if task_type == BINARY_CLASSIFICATION:
        pass

    dataset_properties = {
        "multilabel": multilabel,
        "multiclass": multiclass,
        "sparse": sparse,
        "target_type": "classification"
    }
    return dataset_properties, _feat_type


def normalize_data(X_train, y_train):
    """
    标准化数据

    Parameters
    ----------
    X_train :
    y_train :

    Returns
    -------

    """
    scaler = StandardScalerComponent(random_state=True)
    scaler.fit(X_train.values, y_train.values)
    X = scaler.transform(X_train)
    return X, y_train


@DeprecationWarning
def load_dataset_at_fold_back(dataset_name='credit-g', n_fold=5, fold_index=0, seed=42):
    """
    X_train, y_train, X_test, y_test = load_dataset_at_fold(
                    dataset_name=econf.dataset,     n_fold=econf.folds,fold_index=econf.fold_index, seed=econf.random_state)

    Parameters
    ----------
    dataset_name :
    n_fold :
    seed :
    fold_index: int
        If fold_index is None, yeild all fold data, else return specific fold data

    Returns
    -------
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    """
    np.random.seed(seed=seed)
    assert fold_index < n_fold, "fold_index must be less than n_fold"
    if fold_index is None:
        return_fold_index = [i for i in range(n_fold)]
    else:
        return_fold_index = [fold_index]

    npz_file_name = f'{DATASET_HOME}/{dataset_name}/{dataset_name}.npz'

    if not os.path.exists(npz_file_name):
        raise RuntimeError(
            f"{npz_file_name} is not exists, please prerun download_and_process_data.py in dataset directory")

    with np.load(npz_file_name) as data:
        X = data['X']
        y = data['y']
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

    for _fold_index, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        if _fold_index in return_fold_index:
            np.random.seed(seed)
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            return X_train, y_train, X_test, y_test
        else:
            continue


def load_dataset_at_fold(dataset_name='credit-g', fold_index=0, n_fold=10, seed=42):
    """
    econf=ExpConf()
    X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name=econf.dataset, n_fold=10,
                                                            fold_index=0, seed=42)

    Parameters
    ----------
    dataset_name :
    n_fold :
    seed :
    fold_index: int
        If fold_index is None, yeild all fold data, else return specific fold data

    Returns
    -------
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    """
    npz_file_name = f'{DATASET_HOME}/{dataset_name}/{dataset_name}_fold_{fold_index}.npz'
    with load(npz_file_name) as data:
        X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
    return X_train, y_train, X_test, y_test


def get_dataset_task_id_by_name(name):
    maps = {
        "3": "kr-vs-kp",
        "15": "breast-w",
        "29": "credit-approval",
        "31": "credit-g",
        "37": "diabetes",
        "3021": "sick",
        "43": "spambase",
        "49": "tic-tac-toe",
        "219": "electricity",
        "3902": "pc4",
        "3903": "pc3",
        "3904": "jm1",
        "3913": "kc2",
        "3917": "kc1",
        "3918": "pc1",
        "14965": "bank-marketing",
        "10093": "banknote-authentication",
        "10101": "blood-transfusion-service-center",
        "9971": "ilpd",
        "9976": "madelon",
        "9977": "nomao",
        "9978": "ozone-level-8hr",
        "9952": "phoneme",
        "9957": "qsar-biodeg",
        "9946": "wdbc",
        "9910": "Bioresponse",
        "14952": "PhishingWebsites",
        "14954": "cylinder-bands",
        "125920": "dresses-sales",
        "167120": "numerai28.6",
        "167141": "churn",
        "167125": "Internet-Advertisements",
        "146820": "wilt",
        "146819": "climate-model-simulation-crashes"
    }
    for k in maps:
        v = maps[k]
        if v == str(name).strip():
            return k
    raise ValueError(f"Unknown dataset name: {name}")


def load_dataset_at_fold_encode(dataset_name='credit-g', fold_index=0, n_fold=10, seed=42):
    """
    X_train, y_train, X_test, y_test = load_dataset_at_fold(
                    dataset_name=econf.dataset,     n_fold=econf.folds,fold_index=econf.fold_index, seed=econf.random_state)

    Parameters
    ----------
    dataset_name :
    n_fold :
    seed :
    fold_index: int
        If fold_index is None, yeild all fold data, else return specific fold data

    Returns
    -------
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    """
    npz_file_name = f'{DATASET_HOME}/{dataset_name}/{dataset_name}_encode_fold_{fold_index}.npz'
    with load(npz_file_name) as data:
        X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
    return X_train, y_train, X_test, y_test


@dataclass
class BenchmarkItem:
    """
    Usage:
    benchmark_configs = config_load(os.path.join("./", "resources", 'benchmarks', "demo.yaml"))
    for conf in benchmark_configs:
        bch=BenchmarkItem(**conf)
        print(conf)
    """
    name: str = None
    folds: int = None
    max_runtime_seconds: int = None
    description: str = None
    output_dir: str = None
    model_name: str = None
    metric: str = None
    seed: int = None
    dataset_home: str = None

    # 自动觉得训练数据大小时的停止阈值
    # slop=$max(\{loss_i-loss_{i-1}\}_{1}^{HIS})$，HIS 为考虑的历史训练数据数量
    # if slop < threshold 就停止
    threshold: float = None
    # 自动抽样时的step大小
    # np.arange(stop=sample_step, stop=num of dataset, step=sample_step)
    sample_step: int = None

    def __post_init__(self):
        assert self.folds >= 2, f"folds must be >=2, received {self.folds}"

    @property
    def dataset_name(self):
        return self.name

    def get_config_name(self):
        # 组合所有键和值形成文件名， 只保留字母数字下划线
        chars_maps = {
            ".": "_",
            "/": "_",
            "\\": "_",
            " ": "_",
            "-": "_",
        }
        skip_keys = ["dataset_home", "description"]
        items = []
        for field in fields(self):  # 使用 fields() 获取字段
            value = getattr(self, field.name)
            if value in skip_keys:
                continue

            if value is not None:  # 只包括非 None 的值
                # {field.name}
                items.append(f"{value}")
        ret_date = "_".join(items)
        for k in chars_maps.keys():
            ret_date = ret_date.replace(k, chars_maps.get(k))

        return re.sub(r'[^\w]', '', ret_date.lower())

    def get_metric_file_name(self, fold_index):
        """
        返回指标文件的保存路径
        """
        f_name = pathlib.Path(f'{self.output_dir}/{self.get_config_name()}_{fold_index}.csv')
        f_dir = f_name.parent
        if not f_dir.exists():
            f_dir.mkdir(parents=True)
        return f_name.absolute().as_posix()

    def is_metric_file_exists(self, fold_index):
        return os.path.exists(self.get_metric_file_name(fold_index))


def run_benchmark(conf: BenchmarkItem):
    try:
        distributed.print(f'Start job {conf}')

        DATASET_NAME = conf.dataset_name
        metrics = []
        with open(f'{conf.dataset_home}/{DATASET_NAME}/features.json') as f:
            features = json.load(f)
        print('=' * 16, 'LOAD DATASET', '=' * 16)
        data = pd.read_csv(f'{conf.dataset_home}/{DATASET_NAME}/{DATASET_NAME}.csv')
        print('Dataset: ', DATASET_NAME, data.shape, )
        print('PREPROC DATASET')
        X, y = preproc_data(data, features)
        skf = StratifiedKFold(n_splits=conf.folds, shuffle=True, random_state=42)

        for count, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            # Check file is exists to decide whether to skip
            metric_file_name = conf.get_metric_file_name(count)
            if conf.is_metric_file_exists(fold_index=count):
                distributed.print(f"Metric file is exists at {metric_file_name}")
                continue

            RANDOM_SEED = count
            np.random.seed(RANDOM_SEED)
            # shuffle columns for more randomization experiment
            columns_tmp = list(X.columns.values)
            np.random.shuffle(columns_tmp)
            X = X[columns_tmp]

            # Split
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            # print(DATASET_NAME, X_train.shape, X_test.shape)

            # Auto_ml
            START_EXPERIMENT = time.time()
            if conf.metric == "auc":
                metric = roc_auc
            else:
                raise RuntimeError(f"Unsupported metric {conf.metric}")

            automl = autosklearn.classification.AutoSklearnClassifier(
                include={'feature_preprocessor': ["no_preprocessing"]},
                ensemble_size=1,
                initial_configurations_via_metalearning=0,
                allow_string_features=False,
                time_left_for_this_task=conf.max_runtime_seconds,
                metric=metric,
                seed=conf.seed,
                memory_limit=None
            )
            automl.fit(X_train, y_train)

            try:
                predictions = automl.predict_proba(X_test)
            except RuntimeError:
                predictions = automl.predict(X_test)
            y_test_predict_proba = predictions[:, 1]
            # y_test_predict = automl.predict(X_test)

            print('AUC: ', roc_auc_score(y_test, y_test_predict_proba))

            END_EXPERIMENT = time.time()
            _t = {
                'AUC': round(roc_auc_score(y_test, y_test_predict_proba), 4),
                'log_loss': round(log_loss(y_test, y_test_predict_proba), 4),
                'Accuracy': round(accuracy_score(y_test, (y_test_predict_proba > 0.5)), 4),
                'Time_min': (END_EXPERIMENT - START_EXPERIMENT) // 60,
                'start_seconds': START_EXPERIMENT,
                'end_seconds': END_EXPERIMENT,
                "fold_index": count,
                "file": metric_file_name
            }
            _t.update(conf.__dict__)
            metrics.append(_t)
            pd.DataFrame(metrics).to_csv(metric_file_name, index=False)

    except Exception as e:
        distributed.print(traceback.format_exc())


def run_benchmark_one_fold(conf: BenchmarkItem, fold_index: int):
    """
    只运行指定的Fold
    """
    try:
        DATASET_NAME = conf.dataset_name
        with open(f'{conf.dataset_home}/{DATASET_NAME}/features.json') as f:
            features = json.load(f)
        data = pd.read_csv(f'{conf.dataset_home}/{DATASET_NAME}/{DATASET_NAME}.csv')
        X, y = preproc_data(data, features)
        skf = StratifiedKFold(n_splits=conf.folds, shuffle=True, random_state=conf.seed)

        for count, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            # 如果不是指定的fold，跳过
            if fold_index == count:
                distributed.print(f'Start job at fold {fold_index} with id ={conf.get_metric_file_name(fold_index)}')
                # Check file is exists to decide whether to skip
                metric_file_name = conf.get_metric_file_name(count)
                if conf.is_metric_file_exists(fold_index=count):
                    distributed.print(f"Metric file is exists at {metric_file_name}")
                    continue

                np.random.seed(conf.seed)
                # shuffle columns for more randomization experiment
                columns_tmp = list(X.columns.values)
                np.random.shuffle(columns_tmp)
                X = X[columns_tmp]

                # Split
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
                # print(DATASET_NAME, X_train.shape, X_test.shape)

                # Auto_ml
                START_EXPERIMENT = time.time()
                if conf.metric == "auc":
                    metric = roc_auc
                else:
                    raise RuntimeError(f"Unsupported metric {conf.metric}")

                automl = autosklearn.classification.AutoSklearnClassifier(
                    include={'feature_preprocessor': ["no_preprocessing"]},
                    ensemble_size=1,
                    initial_configurations_via_metalearning=0,
                    allow_string_features=False,
                    time_left_for_this_task=conf.max_runtime_seconds,
                    metric=metric,
                    seed=conf.seed,
                    memory_limit=None,
                    n_jobs=1
                )
                automl.fit(X_train, y_train)

                try:
                    predictions = automl.predict_proba(X_test)
                except RuntimeError:
                    predictions = automl.predict(X_test)
                y_test_predict_proba = predictions[:, 1]
                # y_test_predict = automl.predict(X_test)

                print('AUC: ', roc_auc_score(y_test, y_test_predict_proba))

                END_EXPERIMENT = time.time()
                _t = {
                    'AUC': round(roc_auc_score(y_test, y_test_predict_proba), 4),
                    'log_loss': round(log_loss(y_test, y_test_predict_proba), 4),
                    'Accuracy': round(accuracy_score(y_test, (y_test_predict_proba > 0.5)), 4),
                    'Time_min': (END_EXPERIMENT - START_EXPERIMENT) // 60,
                    'start_seconds': START_EXPERIMENT,
                    'end_seconds': END_EXPERIMENT,
                    "fold_index": count,
                    "file": metric_file_name
                }
                _t.update(conf.__dict__)
                pd.DataFrame([_t]).to_csv(metric_file_name, index=False)
            else:
                # skip to run this fold
                pass

    except Exception as e:
        distributed.print(traceback.format_exc())


def benchmark_over_auto_sample_num(conf: BenchmarkItem, fold_index: int):
    """
    自动确认训练数据大小时，在automl上的模型的精度，目前只支持auc_roc

    # dataset_name, fold_index, n_fold=5, seed=42, sample_step=1024,
    #                               threshold=0.005
    """
    try:
        distributed.print(f"start job: {locals()}")
        from sklearn import tree
        losses = []
        for X_train, y_train, X_test, y_test, _fold_index in load_dataset_at_fold(dataset_name=conf.dataset_name,
                                                                                  n_fold=conf.folds,
                                                                                  seed=conf.seed):

            # X_train: pd.DataFrame
            metric_file_name = conf.get_metric_file_name(_fold_index)
            if conf.is_metric_file_exists(fold_index=_fold_index):
                distributed.print(f"Metric file is exists at {metric_file_name}")
                continue
            if fold_index == _fold_index:
                # 在训练数据中抽样
                sample_num = find_optimized_sample_size_over_dataset(X_train, y_train, X_test, y_test,
                                                                     sample_step=conf.sample_step,
                                                                     seed=conf.seed, threshold=conf.threshold)
                if sample_num == X_train.shape[0]:
                    train_sample_index = X_train.index
                else:
                    train_sample_index = X_train.sample(n=sample_num, random_state=conf.seed).index

                # Auto_ml
                START_EXPERIMENT = time.time()
                if conf.metric == "auc":
                    metric = roc_auc
                else:
                    raise RuntimeError(f"Unsupported metric {conf.metric}")

                automl = autosklearn.classification.AutoSklearnClassifier(
                    include={'feature_preprocessor': ["no_preprocessing"]},
                    ensemble_size=1,
                    initial_configurations_via_metalearning=0,
                    allow_string_features=False,
                    time_left_for_this_task=conf.max_runtime_seconds,
                    metric=metric,
                    seed=conf.seed,
                    memory_limit=None,
                    n_jobs=1
                )
                automl.fit(X_train.loc[train_sample_index], y_train.loc[train_sample_index])

                try:
                    predictions = automl.predict_proba(X_test)
                except RuntimeError:
                    predictions = automl.predict(X_test)
                y_test_predict_proba = predictions[:, 1]
                # y_test_predict = automl.predict(X_test)

                distributed.print('AUC: ', roc_auc_score(y_test, y_test_predict_proba))

                END_EXPERIMENT = time.time()
                _t = {
                    'AUC': round(roc_auc_score(y_test, y_test_predict_proba), 4),
                    'log_loss': round(log_loss(y_test, y_test_predict_proba), 4),
                    'Accuracy': round(accuracy_score(y_test, (y_test_predict_proba > 0.5)), 4),
                    'Time_min': (END_EXPERIMENT - START_EXPERIMENT) // 60,
                    'start_seconds': START_EXPERIMENT,
                    "sample_num": sample_num,
                    'end_seconds': END_EXPERIMENT,
                    "fold_index": fold_index,
                    "file": metric_file_name
                }
                _t.update(conf.__dict__)
                pd.DataFrame([_t]).to_csv(metric_file_name, index=False)
            else:
                continue
        if len(losses) > 0:
            return pd.DataFrame(losses)
        return None
    except:
        distributed.print(traceback.format_exc())
        return None


@dataclasses.dataclass
class TrainingHistory:
    """
    th = TrainingHistory(cs=cs)
    ...
    th.add_history(_config, f1)

    """
    cs: ConfigurationSpace
    econf: ExpConf
    configs: list[Configuration] = None
    targets: list[RunValue] = None
    seed: int = None
    debug: bool = True

    # todo： 转为最小化问题，越小越好
    loss_if_none: float = 9999
    stop_watch: StopWatch = None

    def get_training_time(self):
        """
        获取模型的训练时间
        Returns
        -------

        """
        time = 0
        for v in self.targets:
            time = time + v.elapsed_seconds
        return time

    def get_data_processing_time(self):
        """
        获取模型的训练时间
        Returns
        -------

        """
        return round(self.stop_watch.wall_elapsed(Steps.DATA_PROCESSING), 4)

    def __post_init__(self):
        self.configs = []
        self.targets = []
        if self.stop_watch is None:
            self.stop_watch = StopWatch()

    def __hash__(self):
        # 计算哈希值，使用不可变的属性
        return hash((self.seed, tuple(self.configs), tuple(self.targets), self.debug))

    def __eq__(self, other):
        if not isinstance(other, TrainingHistory):
            return NotImplemented
        return (self.seed, tuple(self.configs), tuple(self.targets), self.debug) == \
            (other.seed, tuple(other.configs), tuple(other.targets), other.debug)

    @property
    def count(self):
        return len(self.configs)

    def is_empyt(self):
        return len(self.configs) == 0

    def get_important_index_by_important_hpys(self, important_hpys):
        if important_hpys is None:
            return None
        else:
            out = []
            for _col_index, (_hp_key, _hp_val) in enumerate(dict(self.cs).items()):
                if _hp_key in important_hpys.hyperparameters:
                    out.append(_col_index)
            return out

    def get_array_history(self):
        """
        将 ConfigurationSpace 的实例转为一个二维数组，以便surrogate model 训练。

        最后一列是 loss，如 f1 score

        Parameters
        ----------
        cs :

        Returns
        -------

        """

        data = self.get_array_of_configurations(self.configs)

        # todo: 转诶最小化问题
        _target = self.get_target_array()['default'].values

        _target_2d = _target.reshape(_target.shape[0], 1)
        _target_2d = _target_2d
        final_data = np.concatenate([data, _target_2d], axis=1)

        # 归一化每一列
        return final_data

    def get_max_target_config(self) -> Configuration:
        """
        获取最大目标值(loss, f1) 对应的配置
        Returns
        -------

        """
        target = np.asarray(self.targets)
        return self.configs[target.argmax()]

    def get_max_target_config_by_model_name(self, model_name: str) -> typing.Union[Configuration, None]:
        """
        获取最大目标值(loss, f1) 对应的模型配置
        Returns
        -------

        """
        # 将None 变为0，以便于更好处理
        _target = [0 if v is None else v for v in self.targets]
        target = np.asarray(self.targets)
        sorted_losses = np.argsort(target)

        for candidate_i in sorted_losses:
            if self.configs[candidate_i]['__choice__'] == model_name:
                return self.configs[candidate_i]

        # Cant find model name candidate
        return None

    def get_max_target_config_vector(self) -> np.ndarray:
        """
        获取最大目标值(loss, f1) 对应的配置转成的向量
        Returns
        -------

        """
        data = self.get_array_of_configurations(self.configs)
        target = np.asarray(self.targets)
        return data[target.argmax()]

    @property
    def max(self):
        """
        获取default中最大的指标
        Returns
        -------

        """
        return self.get_target_array()['default'].max()

    @property
    def min(self):
        """
        获取default中最小的指标
        Returns
        -------

        """
        return self.get_target_array()['default'].min()

    def add_history(self, config: Configuration, run_value: RunValue):
        if run_value.default is None or run_value.default is np.nan:
            target = self.loss_if_none
        self.configs.append(config)
        self.targets.append(run_value)

        log.debug(
            f"Add training history, count: {self.count}, elapsed: {self.wall_elapsed}s, target:{run_value}, config: {config}\n")

    def get_gpt_history(self, decimals=4):
        output_str = []
        output_str.append("hyperparameter configurations; target performance (larger is better)")
        for i in range(len(self.configs)):

            # 配置如果是小数，保存 decimals 位小数
            _t = {}

            _config = self.configs[i]
            for _k in _config.keys():
                _v = _config.get(_k)
                if isinstance(_v, float) or isinstance(_v, np.float64):
                    _v = round(_v, decimals)
                _t[_k] = _v

            # target(loss) 如果是小数，保存 decimals 位小数
            _value = self.targets[i]
            if isinstance(_value, np.float64):
                _value = round(_value, decimals)
            else:
                pass

            output_str.append(str(_t) + ";" + str(_value))
        return "\n".join(output_str)

    def get_optimized_models(self) -> tuple:
        return self.cs.get("__choice__").choices

    def get_configs_and_targets(self):
        return zip(self.configs, self.targets)

    def get_config_names(self):
        return list(dict(self.cs).keys())

    def get_array_of_configurations(self, candidate_samples, normalize=False):
        """
        空值用0替代

        Parameters
        ----------
        candidate_samples :

        Returns
        -------

        """
        # 行数就是当前历史配置的数量
        n_row = len(candidate_samples)

        # 列数就是超参数配置的数量
        n_col = len(self.cs)
        data = np.zeros(shape=(n_row, n_col))

        # 从超参数空间生成向量：1）变量已经保存的配置，2）变量超参数空间，如果有值，就给定值，否者值就是0
        # 遍历每个已经保存的数据
        for _row_index, _c in enumerate(candidate_samples):
            # 遍历超参数空间，以生成矩阵形式的超参数
            for _col_index, (_hp_key, _hp_val) in enumerate(dict(self.cs).items()):
                if _c.get(_hp_key) is None:
                    continue

                # 如果是分类变量，那么值取值在数组中的索引
                if isinstance(_hp_val, CategoricalHyperparameter):
                    _options = list(_hp_val.choices)
                    data[_row_index, _col_index] = _options.index(_c[_hp_key])

                elif isinstance(_hp_val, UniformFloatHyperparameter):
                    if _hp_key in list(_c.keys()):
                        data[_row_index, _col_index] = round(_c.get(_hp_key), 6)
                elif isinstance(_hp_val, UniformIntegerHyperparameter):
                    if _hp_key in list(_c.keys()):
                        data[_row_index, _col_index] = _c.get(_hp_key)

                elif isinstance(_hp_val, Constant):

                    # todo: is this right?
                    data[_row_index, _col_index] = 1

                else:
                    raise RuntimeError(f"Unsupported hyperparameter type {type(_hp_val)}")

        # 标准化每一列
        if normalize:
            col_mean = np.mean(data, axis=0)
            col_std = np.std(data, axis=0)
            return data - col_mean / col_std
        else:
            return data

    def get_target_array(self):
        values = []
        for _v in self.targets:
            values.append(_v.get_dict())
        return pd.DataFrame(values)

    def get_max_target(self):
        ta = self.get_target_array()
        return ta['default'].max()

    def get_important_hpys_col_indexes(self, select_hpys=None) -> list[int] or None:
        if select_hpys is None:
            return None
        else:
            out = []
            for _col_index, (_hp_key, _hp_val) in enumerate(self.cs.get_hyperparameter_names()):
                if _hp_key in select_hpys:
                    out.append(_col_index)
            return out

    @staticmethod
    def parse_autosklearn_conf(config: Configuration):
        """
        解析AutoSKlearn生成的模型

        # config 是从ConfigSpace中的一个配置
        choice, model_configs = TrainingHistory.parse_autosklearn_conf(config)
        # 实例化模型并训练
        clf = ClassifierChoice.get_components()[choice](**model_configs)
        clf.iterative_fit(X_train, y_train)

        Parameters
        ----------
        config :

        Returns
        -------

        """
        model_configs = {}
        choice = config.get("__choice__")
        for _k in config:
            if _k != "__choice__":
                model_configs[_k.replace(f"{choice}:", "")] = config[_k]
        return choice, model_configs

    @staticmethod
    def parse_autosklearn_conf_with_default_filled(run_job: RunJob):
        """
        解析AutoSKlearn生成的模型

        # config 是从ConfigSpace中的一个配置
        choice, model_configs = TrainingHistory.parse_autosklearn_conf(config)
        # 实例化模型并训练
        clf = ClassifierChoice.get_components()[choice](**model_configs)
        clf.iterative_fit(X_train, y_train)

        Parameters
        ----------
        config :

        Returns
        -------

        """
        default_model_conf = TrainingHistory.get_model_default_hpys(run_job)
        model_configs = {}
        choice = run_job.config.get("__choice__")
        for _k in run_job.config:
            if _k != "__choice__":
                model_configs[_k] = run_job.config[_k]

        # 更新超参数配置
        default_model_conf.update(model_configs)

        # 去除前缀
        return_config = {}
        for _k, _v in default_model_conf.items():
            return_config[str(_k).split(":")[-1]] = _v

        return choice, return_config

    def get_id(self):
        his = self.get_array_history()
        mean = his.mean()
        max = his.max()
        std = his.std()
        cumsum = his.cumsum()
        return get_str_md5(str([mean, max, std, cumsum]))

    def sort_configurations(self, top_n=10):
        """
        将超参数配置按照性能降序排序。第一个配置是最好的，最后一个配置是最坏的. 为了性能，我只返回前 top_n 个.

        排序时按照 RunValue 中的default value 得值来排序的
        Parameters
        ----------
        ascending :

        Returns
        -------

        """
        loss = self.get_target_array()['default']
        sort_index = np.argsort(loss)[::-1]

        outputs = []
        for i in sort_index:
            outputs.append((self.configs[i], self.targets[i]))
            if len(outputs) >= top_n:
                break
        return outputs

    def get_top_n_best_configs_and_acc(self, top_n=10):
        """
        获取当前发现的最好的n个配置（top_n）
        Parameters
        ----------
        top_n :

        Returns
        -------

        """
        confs = self.sort_configurations(top_n=top_n)
        return confs

    def get_top3_model_name_and_metric(self, top_n=3):
        """
        获取当前发现的最好的n个配置.

        返回样例
        {'top0_measure_default': 0.8525, 'top0_model_name': 'decision_tree', 'top1_measure_default': 0.8, 'top1_model_name': 'multinomial_nb', 'top2_measure_default': 0.795, 'top2_model_name': 'random_forest'}
        Parameters
        ----------
        top_n :

        Returns
        -------

        """
        assert len(self.targets) >= top_n, \
            f"len(self.target) must larger than {top_n}. got top_n={top_n}, len(self.targets)={len(self.targets)}"
        ret = {}
        confs = self.sort_configurations(top_n=top_n)
        for i in range(top_n):
            ret[f'top{i}_model_name'] = confs[i][0]["__choice__"]
            ret[f'top{i}_measure_default'] = confs[i][1].default
        return ret

    def get_top_n_best_model_names(self, top_n=10):
        """
        获取当前发现的最好的n个模型（top_n），用于减少模型的搜索空间

        Parameters
        ----------
        top_n :

        Returns
        -------

        """
        top_configs = self.get_top_n_best_configs_and_acc(top_n=30)

        models = []
        for _conf, _loss in top_configs:
            models.append(_conf["__choice__"])
            if len(models) >= top_n:
                break
        return list(set(models))

    @property
    def metrics_file_name(self):
        return self.get_metrics_file_name()

    def get_metrics_file_name(self):
        """
        自动决定当前配置所对应的文件目录

        Parameters
        ----------
        econ :

        Returns
        -------

        """
        assert self.econf is not None, "self.econf can't be None"
        # file_base = pathlib.Path(self.econf.entry_file_name).stem
        file_base = pathlib.Path(self.econf.config_file_name).stem
        # df.to_csv('data.csv.gz', compression='gzip')
        file_name = pathlib.Path(self.get_metrics_home(), file_base, str(self.econf.wall_time_limit_in_s),
                                 f"{self.econf.get_id()}.csv")
        home = file_name.parent

        if not home.exists():
            try:
                home.mkdir(parents=True)
            except FileExistsError:
                # 目录已经存在，证明也是成功的
                pass
        return file_name.absolute().as_posix()

    def save(self, econf: ExpConf, status="success"):
        """
        保存结果到文件
        Parameters
        ----------
        econf :

        Returns
        -------

        """
        # 处理最大最小的acc
        # maximize_acc = dict(self.get_target_array().round(4).max())
        # minimize_acc = dict(self.get_target_array().round(4).min())
        # maximize_acc = {k + "_max": v for k, v in maximize_acc.items()}
        # minimize_acc = {k + "_min": v for k, v in minimize_acc.items()}

        metrics = {
            "id": self.econf.get_id(),
            "default_max": self.max,
            "default_min": self.min,
            "status": status,
            "#instances": self.count,
            "model_training_time": self.get_training_time(),  # 模型训练时间
            "data_processing_time": self.get_data_processing_time(),  # 数据预处理时间
            "walk_time": self.wall_elapsed,  # 总时间
            "configs_and_metrics": str(list(zip(self.configs, self.targets)))
            # "configs": self.configs,
            # "targets": self.targets
        }
        # metrics.update(maximize_acc)
        # metrics.update(minimize_acc)

        # topn的数据
        # top_best_metrics = self.get_top3_model_name_and_metric()
        # metrics.update(top_best_metrics)

        # 训练时间相关
        wall_times = {}
        for _k in Steps.keys():
            wall_times["t_" + _k] = self.stop_watch.wall_elapsed(_k)

        metrics.update(wall_times)
        metrics.update(econf.__dict__)

        file_name = self.get_metrics_file_name()
        log.info(f"Metrics is save to {file_name}")
        pd.DataFrame([metrics]).to_csv(file_name, index=False)
        # pd.DataFrame([metrics]).to_pickle(file_name + ".pkl")

    def save_empty(self, econf: ExpConf):
        """
        保存结果到文件
        Parameters
        ----------
        econf :

        Returns
        -------

        """
        self.save(econf, status="Error, saved for placeholder.")

    def is_metric_cache_available(self):
        file_name = self.get_metrics_file_name()
        return os.path.exists(file_name)

    def load_metrics_file_csv(self):
        return pd.read_csv(self.get_metrics_file_name())

    def get_metrics_home(self):
        if os.environ.get('METRICS_HOME') is not None:
            return os.environ.get('METRICS_HOME')
        else:
            return "./outputs/"

    def is_timedout(self, econf: ExpConf):
        assert self.stop_watch is not None, "selp.stop_watch cannot be None"
        total_wall_time = self.wall_elapsed
        is_reached_wall_time_limit = total_wall_time >= econf.wall_time_limit_in_s
        log.debug(
            f"Is time out: {is_reached_wall_time_limit}, total wall time: {total_wall_time}, time budget: {econf.wall_time_limit_in_s}, n iterations of warm up selection: {econf.warm_start_selection_iteration}")

        if is_reached_wall_time_limit:
            log.debug("===============Time has run out=================\n\n\n")
        return is_reached_wall_time_limit

    def get_remain_time_sceconds(self):
        remain_seconds = self.econf.wall_time_limit_in_s - self.stop_watch.wall_elapsed(Steps.ALL)
        return float(remain_seconds)

    @property
    def runtime_left(self):
        """
        实验剩余的训练时间
        Returns
        -------

        """
        return self.get_remain_time_sceconds()

    @property
    def wall_elapsed(self):
        """
        返回实验开始（实例化StopWatch) 到现在的时间，单位s
        Returns
        -------

        """
        return round(self.stop_watch.wall_elapsed(Steps.DEFAULT), 2)

    def get_feature_names_by_index(self, inportant_index, percent=0.1):
        keys = list(dict(self.cs).keys())

        count = int(len(keys) * percent)
        return list(np.asarray(keys)[inportant_index[:count]])

    @classmethod
    def get_model_default_hpys(cls, run_job):
        try:
            model_name = run_job.config["__choice__"]
            configs = {}
            for _i in run_job.cs.get_hyperparameters():
                if str(_i.name).startswith(model_name):
                    configs[_i.name] = _i.default_value
            return configs
        except Exception as e:
            raise e


def suggest_v1(th: TrainingHistory, cs: ConfigurationSpace, surrogate_model=None, random_state=42, select_hpys=None):
    """
{index}/{econf.iteration}
    Parameters
    ----------
    th :
    cs :
    surrogate_model :
    random_state :
    select_hpys : list
        指定要优化的超参数，None 表示指定全部

    Returns
    -------

    """
    # 初始化默认 surrogate model
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    if surrogate_model is None:
        surrogate_model = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_state,
        )
    # 如果训练历史为空，就随机返回一个配置
    if th.is_empyt():
        return cs.sample_configuration(size=1)

    train_history_array = th.get_array_history()

    important_indexes = th.get_important_hpys_col_indexes(select_hpys)
    candidate_samples = cs.sample_configuration(size=10000)
    train_y = train_history_array[:, -1]
    if important_indexes is not None:
        # 选择了重要的超参数
        train_x = train_history_array[:, important_indexes]
        candidate_x = th.get_array_of_configurations(candidate_samples)[:, important_indexes]
    else:
        # 使用所有超参数
        train_x = train_history_array[:, 0:-1]
        candidate_x = th.get_array_of_configurations(candidate_samples)

    surrogate_model.fit(train_x, train_y)

    kappa = 2.576

    acq = UtilityFunction.ucb(
        x=candidate_x,
        kappa=kappa,
        gp=surrogate_model
    )

    # candidata_predict_score = surrogate_model.predict(th.get_array_of_configurations(candidate_samples))
    best_candidata_index = acq.argmax()
    next_config = candidate_samples[best_candidata_index]

    if select_hpys is None:
        return next_config
    else:
        cur_max_config = th.get_max_target_config()
        for key in select_hpys:
            cur_max_config[key] = next_config[key]
        return cur_max_config


@dataclass
class ImportHyperparameters:
    models: list[str] = None
    hyperparameters: list[str] = None

    def __post_init__(self):
        if self.models is None:
            self.models = []
        if self.hyperparameters is None:
            self.hyperparameters = []

    def add_model(self, _model: str):
        self.models.append(_model)

    def add_hpy(self, _v: str):
        self.hyperparameters.append(_v)


@dataclass
class LLMResponseDecoder:
    """
    解析LLM 返回的重要超参数
    """
    prompts: str = None
    response: str = None

    def decode_import_hpys(self, response_llm: str, cs: ConfigurationSpace) -> ImportHyperparameters:
        # 使用正则表达式去除控制字符
        cleaned_text = re.sub(r'[\r\n]+', '', response_llm)

        # 提前正则表达式
        json_matches = re.findall('```json(\[.*?\])```', cleaned_text, re.DOTALL)

        ih = ImportHyperparameters()
        for _json_str in json_matches:
            _json_arr = json.loads(_json_str)
            for _json in _json_arr:
                for _k, _v in _json.items():

                    # 添加模型信息
                    _model = _json[_k]
                    if _model in cs.get("__choice__").choices:
                        ih.add_model(_model)

                    # 添加超参数信息
                    if str(_k).startswith("key_hp"):
                        ih.add_hpy(_v)

        return ih


class LLMDecisionMaking:
    @staticmethod
    def get_importance_config_by_self_consistency(model_name: str, th: TrainingHistory, ratio: float = 0.3,
                                                  consistency_time=1, gpt_type="gpt3.5") -> list:
        """
        Get the training data without labels

        SELF-CONSISTENCY method

        Returns
        -------

        """
        messages = [
            {"role": "system", "content": f"""
As an assistant, your task is to optimize the hyperparameters of the {model_name} model according to the  training history, you must give the results think the following steps step by step.
Step 1, rank the importance of each hyperparameter based on historical training data by thinking step by step.
Step 2, assign a importance score (from 0 to 1) to each hyperparameter, where a higher score indicates greater importance.
Step 3: output the hyperparameters and corresponding score in a json format, where each key is wrapped by ".

Here are the hyperparameter configurations and corresponding performance we've tried:
                        """},
            {"role": "user", "content": th.get_gpt_history()}
        ]

        # 多次出现，那么就认为更有可能是真的
        # SELF-CONSISTENCY method
        outputs = []
        for i in range(consistency_time):
            if gpt_type == "gpt3.5":
                res = UtilOpenai.parse_message_without_stream(UtilOpenai().chat_gpt35(messages))
            elif gpt_type == "gpt4_turbo":
                res = UtilOpenai.parse_message_without_stream(UtilOpenai().chat_gpt4_turbo(messages))
            else:
                raise RuntimeError(f"Unsupported gpt type: {gpt_type}")

            # 定义正则表达式模式: 匹配输出的key和权重
            pattern = r'"(\w+)":\s*([\d.]+)'

            # 使用re.findall()函数匹配所有键值对
            matches = re.findall(pattern, res)

            # 将匹配结果转换为字典
            result_dict = {key: float(value) for key, value in matches}
            outputs.append(result_dict)

        # 只优化前20%的超参数
        _df = pd.DataFrame(outputs)
        _important = _df.mean(axis=0)
        _important_sort = _important.sort_values(ascending=False)

        # 选择最重要的的那几个参数，然后调优
        _n_keys = int(np.ceil(len(th.cs) * ratio))
        return list(_important_sort[:_n_keys].keys())

    @staticmethod
    def get_top_importance_config_by_self_consistency(model_name: str, th: TrainingHistory, top_n: float = 0.3,
                                                      consistency_time=3, gpt_type="gpt3.5") -> list:
        """
        Get the training data without labels

        SELF-CONSISTENCY method

        Returns
        -------

        """
        messages = [
            {"role": "system", "content": f"""
As an assistant, your task is to optimize the hyperparameters of the {model_name} model according to the  training history, you must give the results think the following steps step by step.
Step 1, Tank the importance of each hyperparameter based on historical training data by analysing the relationship between each hyperparameter configuration and the target performance. You must thinking step by step.
Step 2, Assign a importance score (from 0 to 1) to each hyperparameter, where a higher score indicates greater importance.
Step 3: Output the hyperparameters and corresponding score in a json format, where each key is wrapped by ".

Here are the hyperparameter configurations and corresponding performance we've tried:
                        """},
            {"role": "user", "content": th.get_gpt_history()}
        ]

        # 多次出现，那么就认为更有可能是真的
        # SELF-CONSISTENCY method
        outputs = []
        for i in range(consistency_time):
            if gpt_type == "gpt3.5":
                res = UtilOpenai.parse_message_without_stream(UtilOpenai().chat_gpt35(messages))
            elif gpt_type == "gpt4_turbo":
                res = UtilOpenai.parse_message_without_stream(UtilOpenai().chat_gpt4_turbo(messages))
            else:
                raise RuntimeError(f"Unsupported gpt type: {gpt_type}")

            print(res)

            # 定义正则表达式模式: 匹配输出的key和权重
            pattern = r'"(\w+)":\s*([\d.]+)'

            # 使用re.findall()函数匹配所有键值对
            matches = re.findall(pattern, res)

            # 将匹配结果转换为字典
            result_dict = {key: float(value) for key, value in matches}
            outputs.append(result_dict)

        # 只优化前20%的超参数
        _df = pd.DataFrame(outputs)
        _important = _df.mean(axis=0)
        _important_sort = _important.sort_values(ascending=False)

        # 选择最重要的的那几个参数，然后调优
        return list(_important_sort[:top_n].keys())

    @staticmethod
    def get_top_importance_config_by_self_consistency_multi_model(
            config_space: ConfigurationSpace,
            th: TrainingHistory,
            top_n: float = 3,
            consistency_time=3, gpt_type="gpt3.5") -> typing.Tuple[ImportHyperparameters, LLMResponseDecoder]:
        """
        Get the training data without labels

        SELF-CONSISTENCY method

        Returns
        -------

        """
        search_space = config_space.get_hyperparameters()
        history = th.get_gpt_history()
        prompts = f"""Your task is to optimize hyperparameters based on the provided training history and search space. The search space is enclosed by =====, while the training history is marked by -----.

You must give the results think the following steps step by step.

1)Rank the importance of each hyperparameter by analyzing the relationship between hyperparameter configurations and target performance using historical training data, by think step b
2)Assign an importance score (from 0 to 1) to each hyperparameter, where a higher score indicates greater importance. Consider the association between the model and hyperparameters during this assessment.
3)Predict {top_n} important models along with {top_n} associated hyperparameters based on the analysis. The JSON output should start with '```json` and end with '```'.

Here are the search space:
=====
{search_space}
=====

Here are the hyperparameter configurations and corresponding performances we've tried:
-----
{history}
-----

One output  example is as follows:
[
{{
    "__choice__": "lda",
    "key_hp1": "lda:shrinkage",
    "key_hp2": "lda:tol",
    "key_hp3": "None"
}},
{{
    "__choice__": "bernoulli_nb",
    "key_hp1": "bernoulli_nb:alpha",
    "key_hp2": "bernoulli_nb:fit_prior",
    "key_hp3": "None"
}},
{{
    "__choice__": "adaboost",
    "key_hp1": "adaboost:algorithm",
    "key_hp2": "adaboost:learning_rate",
    "key_hp3": "adaboost:n_estimators"
}}
]
"""

        messages = [
            {"role": "system", "content": prompts},
            # {"role": "user", "content": "-----" + th.get_gpt_history() + "\n-----"}
        ]

        # 多次出现，那么就认为更有可能是真的
        # SELF-CONSISTENCY method
        outputs = []
        for i in range(consistency_time):
            if gpt_type == "gpt3.5":
                res = UtilOpenai.parse_message_without_stream(UtilOpenai().chat_gpt35(messages))
            elif gpt_type == "gpt4o-mini":
                res = UtilOpenai.parse_message_without_stream(UtilOpenai().chat_gpt4o_mini_2024_07_18(messages))
            else:
                raise RuntimeError(f"Unsupported gpt type: {gpt_type}")

            print(res)
            # 保存日志到内存供查看
            pd.Series({
                "prompt": str(messages),
                "response": res
            }).to_csv("gpt_history.csv", mode='a')

            decoder = LLMResponseDecoder(prompts=str(messages), response=res)

            # todo: implement SELF-CONSISTENCY method
            important_hpys = decoder.decode_import_hpys(res, config_space)
            return important_hpys, decoder


def suggest(th: TrainingHistory,
            cs: ConfigurationSpace,
            important_hpys: ImportHyperparameters = None,
            surrogate_model=None,
            random_state=42,
            is_select_model_hps: bool = False
            ):
    """

    Parameters
    ----------
    important_hpys : ImportHyperparameters
        指定要优化的超参数，None 表示指定全部
    cs : ConfigurationSpace
        超参数空间，可以增加 ForbiddenClause 以减少超参数空间的配置
    th :
    surrogate_model :
    random_state :

    Returns
    -------

    """
    # 初始化默认 surrogate model
    log.debug("Start suggestion by BO ...")
    if surrogate_model is None:
        surrogate_model = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_state,
        )
    # 如果训练历史为空，就随机返回一个配置
    if th.is_empyt():
        return cs.sample_configuration(size=1)

    train_history_array = th.get_array_history()
    important_indexes = th.get_important_index_by_important_hpys(important_hpys)
    candidate_samples = cs.sample_configuration(size=10000)

    train_y = train_history_array[:, -1]
    if important_indexes is not None:
        # 选择了重要的超参数
        train_x = train_history_array[:, important_indexes]
        candidate_x = th.get_array_of_configurations(candidate_samples)[:, important_indexes]
    else:
        # 使用所有超参数
        train_x = train_history_array[:, 0:-1]
        candidate_x = th.get_array_of_configurations(candidate_samples)

    surrogate_model.fit(train_x, train_y)

    kappa = 2.576

    acq = UtilityFunction.ucb(
        x=candidate_x,
        kappa=kappa,
        gp=surrogate_model
    )

    # candidata_predict_score = surrogate_model.predict(th.get_array_of_configurations(candidate_samples))
    best_candidata_index = acq.argmax()
    next_config = candidate_samples[best_candidata_index]
    log.debug("Suggest of PO is done")
    # 是否选择模型的超参数
    if is_select_model_hps:
        if important_hpys is None:
            return next_config
        else:
            # 将没有考虑的那些模型超参数设置为当前发现的最优值
            next_select_model_name = next_config['__choice__']
            cur_max_config = th.get_max_target_config_by_model_name(next_select_model_name)
            if cur_max_config is None:
                return next_config
            else:
                for key in important_hpys.hyperparameters:
                    if key.startswith(next_select_model_name):
                        cur_max_config[key] = next_config[key]
                return cur_max_config

    else:
        return next_config


def bo_ask_minimize(
        th: TrainingHistory,
        cs: ConfigurationSpace,
        surrogate_model=None,
        random_state=42,
):
    """
    贝叶斯优化，转为最小化问题了.
    历史数据存在th中了，相当于tell在history中，这里只负责ask

    Parameters
    ----------
    th :
    cs :
    surrogate_model :
    random_state :

    Returns
    -------

    """
    # 初始化默认 surrogate model
    log.debug("Start suggestion by BO ...")
    if surrogate_model is None:
        surrogate_model = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_state,
        )
    # 如果训练历史为空，就随机返回一个配置
    if th.is_empyt():
        return cs.sample_configuration(size=1)

    train_history_array = th.get_array_history()
    candidate_samples = cs.sample_configuration(size=10000)

    train_x = train_history_array[:, 0:-1]
    train_y = train_history_array[:, -1]

    candidate_x = th.get_array_of_configurations(candidate_samples)

    # 加个负号，转为最小值问题
    surrogate_model.fit(train_x, -train_y)

    kappa = 2.576

    acq = UtilityFunction.ucb(
        x=candidate_x,
        kappa=kappa,
        gp=surrogate_model
    )
    best_candidata_index = acq.argmax()
    next_config = candidate_samples[best_candidata_index]
    log.debug("Suggest of PO is done")
    # 是否选择模型的超参数
    return next_config


def get_cs_id(cs: ConfigurationSpace) -> str:
    """
    获取ConfigurationSpace的id，一个id表示个ConfigurationSpace
    Parameters
    ----------
    cs :

    Returns
    -------

    """
    return get_str_md5(str(dict(cs)))


def train_model_on_budget(running_time_left, run_job: RunJob):
    """
    使用dask 执行训练
    Parameters
    ----------
    running_time_left :
    config :
    X_train :
    y_train :
    X_test :
    y_test :

    Returns
    -------

    """
    if run_job.config.keys():
        pass
    running_time_left = int(running_time_left)
    log.info(f"Budget time left {running_time_left} s")
    if running_time_left <= 0:
        raise FunctionTimedOut()
    if run_job.debug:
        loss = train_model(run_job)
    else:
        loss = func_timeout(running_time_left, train_model,
                            kwargs={"run_job": run_job})
    return loss


def train_model_on_budget_enforce_limit(running_time_left, config, X_train, y_train, X_test, y_test):
    """
    使用dask 执行训练
    Parameters
    ----------
    running_time_left :
    config :
    X_train :
    y_train :
    X_test :
    y_test :

    Returns
    -------

    """

    running_time_left = int(running_time_left)
    log.info(f"Budget time left {running_time_left} s")
    if running_time_left <= 0:
        raise asyncio.exceptions.TimeoutError()

    safe_function = enforce_limits(wall_time_in_s=running_time_left, logger=log)(train_model)
    vus_roc = safe_function(config=config, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    return vus_roc


def train_model_on_budget_dask(running_time_left, config, X_train, y_train, X_test, y_test):
    """
    使用dask 执行训练
    Parameters
    ----------
    running_time_left :
    config :
    X_train :
    y_train :
    X_test :
    y_test :

    Returns
    -------

    """
    client = get_dask_local_client()
    feature = client.submit(train_model, config=config, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    if running_time_left < 0:
        raise asyncio.exceptions.TimeoutError()
    if running_time_left < 1:
        running_time_left = 1
    assert running_time_left >= 1, "running_time_left should be greater than  or euqal to 1"
    # fix: running_time_left must be in seconds
    vus_roc = feature.result(timeout=int(running_time_left))
    return vus_roc


def train_model(run_job: RunJob) -> RunValue:
    """
    返回loss，如f1，acc等，越大越好。越小越好的要加个负号，转为最大值问题
    Parameters
    ----------
    run_job :

    Returns
    -------

    """

    _start_time = time.time()
    enable_numpy_reproduce(run_job.seed)
    # 使用新抽取的超参数覆盖默认超参数，实现更新的目的
    choice, model_configs = TrainingHistory.parse_autosklearn_conf_with_default_filled(run_job)

    log.debug(f"Training model {choice}, "
              f"X_train.shape:{run_job.X_train.shape}, X_test.shape:{run_job.X_test.shape}")

    # 实例化模型并训练
    model_configs['random_state'] = run_job.seed
    clf = ClassifierChoice.get_components()[choice](**model_configs)
    clf.fit(run_job.X_train, run_job.y_train)

    y_predict = clf.predict(run_job.X_test)
    y_predict_proba = clf.predict_proba(run_job.X_test)[:, 1]

    _f1 = f1_score(run_job.y_test, y_predict)
    _precision = precision_score(run_job.y_test, y_predict)
    _recall = recall_score(run_job.y_test, y_predict)

    _roc_auc = roc_auc_score(run_job.y_test, y_predict_proba)

    _log_loss = log_loss(run_job.y_test, y_predict_proba)
    _accuracy = accuracy_score(run_job.y_test, y_predict)

    # 决定默认优化的是什么参数
    _default = None
    if run_job.metric == AnaHelper.METRIC_ROC_AUC:
        _default = _roc_auc
    elif run_job.metric == AnaHelper.METRIC_F1:
        _default = _f1
    elif run_job.metric == AnaHelper.METRIC_ACCURACY:
        _default = _accuracy
    elif run_job.metric == AnaHelper.METRIC_LOG_LOSS:
        _default = _log_loss
    else:
        raise RuntimeError(f"Unknown specific metric {run_job.metric}")
    _end_time = time.time()

    return RunValue(
        default=round(_default, 4),
        f1=round(_f1, 4),
        precision=round(_precision, 4),
        recall=round(_recall, 4),
        roc_auc=round(_roc_auc, 4),
        log_loss=round(_log_loss, 4),
        accuracy=round(_accuracy, 4),
        elapsed_seconds=round(_end_time - _start_time, 4)
    )


def train_model_smac(run_job: RunJob) -> RunValue:
    """
    返回loss，如f1，acc等，越小越好，为了兼容smac。
    SMAC always minimizes (the smaller the better)
    ref: https://automl.github.io/SMAC3/main/3_getting_started.html#target-function
    Parameters
    ----------
    run_job :

    Returns
    -------

    """
    try:

        enable_numpy_reproduce(run_job.seed)
        # 使用新抽取的超参数覆盖默认超参数，实现更新的目的
        choice, model_configs = TrainingHistory.parse_autosklearn_conf_with_default_filled(run_job)

        log.info(f"Training model {choice}, "
                 f"X_train.shape:{run_job.X_train.shape}, X_test.shape:{run_job.X_test.shape}")

        # 实例化模型并训练
        model_configs['random_state'] = run_job.seed

        # fix: ValueError: Expected n_neighbors <= n_samples_fit, but n_neighbors = 86, n_samples_fit = 84, n_samples = 54
        if choice == "k_nearest_neighbors":
            if model_configs['n_neighbors'] >= model_configs['n_neighbors']:
                model_configs['n_neighbors'] = run_job.X_train.shape[0] - 1
        if run_job.is_cache_avaiable():
            query_object = copy.deepcopy(model_configs)
            query_object.update({
                "__choice__": run_job.config["__choice__"],
                "model": run_job.config["__choice__"],
                "dataset": run_job.exp_conf.dataset,
                "fold_index": run_job.exp_conf.fold_index}
            )

            db = KVDBMySQL()
            cache_res_dict = db.query(query_object)
        else:
            cache_res_dict = None
        if cache_res_dict is not None:
            # cache_res = res.json()
            # cache_res_dict = ast.literal_eval(cache_res)
            # cache_res_dict={'accuracy': 0.64, 'elapsed_seconds': 1.0288, 'error_msg': '', 'f1': 0.5714, 'log_loss': -1.0, 'precision': 0.5714, 'recall': 0.5714, 'roc_auc': 0.6913}
            print("🚀load from cache...")
            _f1 = cache_res_dict["f1"]
            _precision = cache_res_dict["precision"]
            _recall = cache_res_dict["recall"]
            _roc_auc = cache_res_dict["roc_auc"]
            _log_loss = -1
            _accuracy = cache_res_dict["accuracy"]
            _elapsed_seconds = cache_res_dict["elapsed_seconds"]
        else:
            print("🚗train new model...")
            _start_time = time.time()
            clf = ClassifierChoice.get_components()[choice](**model_configs)

            clf.fit(run_job.X_train, run_job.y_train)

            y_predict = clf.predict(run_job.X_test)
            y_predict_proba = clf.predict_proba(run_job.X_test)[:, 1]

            _f1 = f1_score(run_job.y_test, y_predict)
            _precision = precision_score(run_job.y_test, y_predict)
            _recall = recall_score(run_job.y_test, y_predict)

            _roc_auc = roc_auc_score(run_job.y_test, y_predict_proba)

            _log_loss = log_loss(run_job.y_test, y_predict)

            _accuracy = accuracy_score(run_job.y_test, y_predict)
            _end_time = time.time()
            _elapsed_seconds = round(_end_time - _start_time, 4)

        # 转为最小值问题
        if run_job.metric == AnaHelper.METRIC_ROC_AUC:
            _default = 1 - _roc_auc
        elif run_job.metric == AnaHelper.METRIC_F1:
            _default = 1 - _f1
        elif run_job.metric == AnaHelper.METRIC_ACCURACY:
            _default = 1 - _accuracy
        elif run_job.metric == AnaHelper.METRIC_PRECISION:
            _default = 1 - _precision
        elif run_job.metric == AnaHelper.METRIC_LOG_LOSS:
            _default = _log_loss
        else:
            raise RuntimeError(f"Unknown specific metric {run_job.metric}")

        run_val = RunValue(
            default=round(_default, 4),
            f1=round(_f1, 4),
            precision=round(_precision, 4),
            recall=round(_recall, 4),
            roc_auc=round(_roc_auc, 4),
            log_loss=round(_log_loss, 4),
            accuracy=round(_accuracy, 4),
            elapsed_seconds=_elapsed_seconds,
            error_msg=""
        )
        # 只缓存全量数据训练的算法
        if run_job.exp_conf.data_sample_rate == 1 and run_job.exp_conf.feature_selec_rate == 1:
            try:
                db.add(query_object, run_val.__dict__)
            except Exception as e:
                log.error(f"❌ Failed to cache model {choice}")
                log.error(e)
        return run_val

    except Exception as e:
        traceback.print_exception(e)
        return RunValue(
            default=1,
            f1=0,
            precision=0,
            recall=0,
            roc_auc=0,
            log_loss=0,
            accuracy=0,
            elapsed_seconds=0,
            error_msg=traceback.format_exc()
        )
        # raise e


def train_model_smac_repro(run_job: RunJob) -> RunValue:
    """
    返回loss，如f1，acc等，越小越好，为了兼容smac。
    TODO：待修复代码，为了复现实验
    SMAC always minimizes (the smaller the better)
    ref: https://automl.github.io/SMAC3/main/3_getting_started.html#target-function
    Parameters
    ----------
    run_job :

    Returns
    -------

    """

    _start_time = time.time()
    enable_numpy_reproduce(run_job.seed)
    # 使用新抽取的超参数覆盖默认超参数，实现更新的目的
    choice, model_configs = TrainingHistory.parse_autosklearn_conf_with_default_filled(run_job)

    log.info(f"Training model {choice}, "
             f"X_train.shape:{run_job.X_train.shape}, X_test.shape:{run_job.X_test.shape}")

    # 实例化模型并训练
    model_configs['random_state'] = run_job.seed
    clf = ClassifierChoice.get_components()[choice](**model_configs)
    clf.fit(run_job.X_train, run_job.y_train)

    y_predict = clf.predict(run_job.X_test)
    y_predict_proba = clf.predict_proba(run_job.X_test)[:, 1]

    _f1 = f1_score(run_job.y_test, y_predict)
    _precision = precision_score(run_job.y_test, y_predict)
    _recall = recall_score(run_job.y_test, y_predict)

    _roc_auc = roc_auc_score(run_job.y_test, y_predict_proba)

    _log_loss = log_loss(run_job.y_test, y_predict)

    _accuracy = accuracy_score(run_job.y_test, y_predict)

    # 决定默认优化的是什么参数
    _default = None
    if run_job.metric == AnaHelper.METRIC_ROC_AUC:
        _default = 1 - _roc_auc
    elif run_job.metric == AnaHelper.METRIC_F1:
        _default = 1 - _f1
    elif run_job.metric == AnaHelper.METRIC_ACCURACY:
        _default = 1 - _accuracy
    elif run_job.metric == AnaHelper.METRIC_PRECISION:
        _default = 1 - _precision
    elif run_job.metric == AnaHelper.METRIC_LOG_LOSS:
        _default = _log_loss

    else:
        raise RuntimeError(f"Unknown specific metric {run_job.metric}")
    _end_time = time.time()

    return RunValue(
        default=round(_default, 4),
        f1=round(_f1, 4),
        precision=round(_precision, 4),
        recall=round(_recall, 4),
        roc_auc=round(_roc_auc, 4),
        log_loss=round(_log_loss, 4),
        accuracy=round(_accuracy, 4),
        elapsed_seconds=round(_end_time - _start_time, 4)
    )


def gather_all_csv_files(files, duplicates=False):
    """
    读取所有给定的文件，并返回一个pd

    demo:
    from pylibs.utils.util_analysis import gather_all_files_csv
    gather_all_files_csv("./outputs").to_csv("all.csv")

    Parameters
    ----------
    duplicates :

    Returns
    -------

    """
    data = []
    for f in files:
        if f is None:
            continue
        try:
            data.append(pd.read_csv(f))
        except:
            traceback.print_exc()

    if duplicates:
        return pd.concat(data, axis=0)
    else:
        return pd.concat(data, axis=0).drop_duplicates()


@DeprecationWarning
def run_configs(fn, econfigs: list[ExpConf], debug):
    """
    运行所有的配置。
    Parameters
    ----------
    fn :
    econfigs :
    debug :

    Returns
    -------

    """
    pass


def benchmark_small_test_pivot_table(df, filename):
    res = pd.pivot_table(df, index=['dataset', 'wall_time_limit_in_s'], columns=['warm_start_selection_iteration'],
                         values=['min_loss'],
                         aggfunc=[np.mean, np.std, np.count_nonzero])
    res = res.round(4)
    res.to_excel(filename)


def benchmark_baseline_pivot_table(df, filename):
    res = pd.pivot_table(df, index=['dataset', 'init_data_size', 'warm_start_selection_iteration'],
                         columns=['wall_time_limit_in_s', 'metric'],
                         values=['min_loss'],
                         aggfunc=[np.mean, np.std, np.count_nonzero])
    res = res.round(4)
    res.to_excel(filename)


def get_all_datasets():
    """
    返回数据集的名称，按数据的大小排序，小的在前

    Returns
    -------

    """
    # 数据来源 openml/dataset_statics.py
    all_datasets = {
        'task_id': {28: '125920', 12: '3913', 33: '146819', 27: '14954', 24: '9946', 18: '9971', 2: '29', 1: '15',
                    17: '10101', 4: '37', 7: '49', 3: '31', 23: '9957', 14: '3918', 16: '10093', 9: '3902', 10: '3903',
                    13: '3917', 21: '9978', 19: '9976', 0: '3', 31: '167125', 25: '9910', 5: '3021', 6: '43',
                    32: '146820', 30: '167141', 22: '9952', 11: '3904', 26: '14952', 20: '9977', 15: '14965', 8: '219',
                    29: '167120'},
        'dataset_name': {28: 'dresses-sales', 12: 'kc2', 33: 'climate-model-simulation-crashes', 27: 'cylinder-bands',
                         24: 'wdbc', 18: 'ilpd', 2: 'credit-approval', 1: 'breast-w',
                         17: 'blood-transfusion-service-center', 4: 'diabetes', 7: 'tic-tac-toe', 3: 'credit-g',
                         23: 'qsar-biodeg', 14: 'pc1', 16: 'banknote-authentication', 9: 'pc4', 10: 'pc3', 13: 'kc1',
                         21: 'ozone-level-8hr', 19: 'madelon', 0: 'kr-vs-kp', 31: 'Internet-Advertisements',
                         25: 'Bioresponse', 5: 'sick', 6: 'spambase', 32: 'wilt', 30: 'churn', 22: 'phoneme', 11: 'jm1',
                         26: 'PhishingWebsites', 20: 'nomao', 15: 'bank-marketing', 8: 'electricity',
                         29: 'numerai28.6'},
        '#instances': {28: 500, 12: 522, 33: 540, 27: 540, 24: 569, 18: 583, 2: 690, 1: 699, 17: 748, 4: 768, 7: 958,
                       3: 1000, 23: 1055, 14: 1109, 16: 1372, 9: 1458, 10: 1563, 13: 2109, 21: 2534, 19: 2600, 0: 3196,
                       31: 3279, 25: 3751, 5: 3772, 6: 4601, 32: 4839, 30: 5000, 22: 5404, 11: 10885, 26: 11055,
                       20: 34465, 15: 45211, 8: 45312, 29: 96320}}
    return list(all_datasets['dataset_name'].values())


def get_small_datasets():
    """
    返回行数最小的前10个数据集
    Returns
    -------

    """
    return get_all_datasets()[:10]


def get_fail_dataset_v1():
    return [
        'climate-model-simulation-crashes',
        'credit-g',
        'dresses-sales',
        'ilpd',
        'madelon',
        'pc3',
        'qsar-biodeg',
        # 'numerai28.6', # todo,to large
    ]


################################################################


# 正则表达式模式
pattern = re.compile(
    r"\(Configuration\(values=(.*?precision=.*?\)\))",
    re.DOTALL
)

import pandas as pd
import re

PATTERN_CONFIG_AND_METRICS = re.compile(
    r"\(Configuration\(values=(.*?precision=.*?\)\))",
    re.DOTALL
)


def find_top_n_model(series: pd.Series, metric="auc_roc"):
    """
    找出表现最好的前几个模型，前1，2，3
    Parameters
    ----------
    series :

    Returns
    -------

    """
    datas = re.findall(PATTERN_CONFIG_AND_METRICS, series['configs_and_metrics'])
    output = []
    for d in datas:
        mystr = d.replace("})", "}").replace("))", ")").replace("\n", "").strip()
        model_conf, ret_value = eval(mystr)
        run_value = ret_value  # type:RunValue

        output.append({
            "loss": run_value.roc_auc,
            "model": model_conf["__choice__"]
        })
    model_acc = pd.DataFrame(output)
    model_acc_sort = model_acc.sort_values(by=metric, ascending=False)

    # 移除
    # model_sort_by_name = model_acc_sort.drop_duplicates(subset='model')

    series[f'top_1_model_name_{metric}'] = model_acc_sort.iloc[0]['model']
    series[f'top_1_model_value_{metric}'] = model_acc_sort.iloc[0]['loss']
    series[f'top_2_model_name_{metric}'] = model_acc_sort.iloc[1]['model']
    series[f'top_2_model_value_{metric}'] = model_acc_sort.iloc[1]['loss']

    return series


def find_top_two_models_for_each_dataset(df):
    """
    找出每个数据集上前2的模型名称。


    Parameters
    ----------
    df :

    Returns
    -------

    """
    top_n_models = []
    for (dataset, iteration, important_feature_ratio), v in df.groupby(
            by=['dataset', 'max_iterations', 'important_feature_ratio']):
        models = []
        for _model in v['top_1_model_name'].to_list():
            models.append({
                "model": _model,
                "count": 1
            })
        for _model in v['top_2_model_name'].to_list():
            models.append({
                "model": _model,
                "count": 1
            })
            # top3_model=eval(v['top_3_models'])
            # print(top3_model)
        res = pd.DataFrame(models).groupby("model").count().reset_index()
        final_res = res.sort_values(by=['count'], ascending=False)
        # print(final_res.to_csv(index=False))
        print(final_res.to_csv(index=False, header=False))
        top_n_models.append({
            "important_feature_ratio": important_feature_ratio,
            "iteration": iteration,
            "dataset": dataset,
            "sort_models": final_res.to_csv(index=False, header=False),
            "summary_top3_model_names": res.nlargest(3, "count", keep="all")['model'].to_list(),
            "wall_time_in_seconds_sum": v['wall_time_in_seconds'].sum(),
            "wall_time_in_seconds_mean": v['wall_time_in_seconds'].mean()
        })
    return pd.DataFrame(top_n_models)


@dataclasses.dataclass
class ExpHelper:
    econf: ExpConf

    @staticmethod
    def get_all_algs():
        """
        获取当前算法空间中所有的算法
        Returns
        -------

        """
        cs = get_auto_sklearn_classification_search_space(y_train=[0, 1],
                                                          random_state=12)

        return cs['__choice__'].choices

    def load_data(self, max_samples=None):
        """
        max_samples 决定是否抽样
        max_samples=None： 不抽样，使用全部数据
        max_samples= N ： 随机抽取N条数据
        Parameters
        ----------
        max_samples : int or None

        Returns
        -------

        """
        econf = self.econf
        enable_numpy_reproduce(econf.random_state)
        X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name=econf.dataset, n_fold=econf.folds,
                                                                fold_index=econf.fold_index, seed=econf.random_state)
        if max_samples is not None and X_train.shape[0] > max_samples:
            max_samples = int(max_samples)
            # Sample date
            train_sample_index = np.random.choice(X_train.shape[0], size=max_samples, replace=False)
            # 抽样
            X_train_sampled, y_train_sampled = X_train[train_sample_index], y_train[train_sample_index]
        else:
            X_train_sampled = X_train
            y_train_sampled = y_train

        return X_train_sampled, y_train_sampled, X_test, y_test

    def load_data_with_dim_redu(self, max_samples=None, n_components=2, dim_redu_method="PCA"):
        """
        加载数据的同时降维
        max_samples 决定是否抽样
        max_samples=None： 不抽样，使用全部数据
        max_samples= N ： 随机抽取N条数据
        Parameters
        ----------
        max_samples : int or None

        Returns
        -------

        """
        econf = self.econf
        X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name=econf.dataset, n_fold=econf.folds,
                                                                fold_index=econf.fold_index, seed=econf.random_state)
        if max_samples is not None and X_train.shape[0] > max_samples:
            max_samples = int(max_samples)
            # Sample date
            train_sample_index = np.random.choice(X_train.shape[0], size=max_samples, replace=False)
            # 抽样
            X_train_sampled, y_train_sampled = X_train[train_sample_index], y_train[train_sample_index]
        else:
            X_train_sampled = X_train
            y_train_sampled = y_train

        return DimentionReductionHelper().transform(X_train_sampled, y_train_sampled, X_test, y_test,
                                                    method=dim_redu_method, n_components=n_components)

    @DeprecationWarning
    def load_data_with_dim_redu_and_sample(self):
        """
        加载数据的同时抽样+降维

        有放回抽样，保证正负样本量等于max_samples

        max_samples 决定是否抽样
        max_samples=None： 不抽样，使用全部数据
        max_samples= N ： 随机抽取N条数据
        Parameters
        ----------
        max_samples : int or None

        Returns
        -------

        """
        econf: ExpConf = self.econf
        sample_method = econf.data_sample_method
        dim_redu_method = econf.feature_selec_method

        X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name=econf.dataset, n_fold=econf.folds,
                                                                fold_index=econf.fold_index, seed=econf.random_state)
        n_components = self.check_n_component(X_train, econf)
        n_samples = self.check_n_samples(X_train, econf)

        if sample_method == "stratified":
            _n_ss_sample = int(n_samples / 2)
            X_train_neg_index = np.where(y_train == 0)[0]
            X_train_pos_index = np.where(y_train == 1)[0]

            # _n_real_sample = np.min([len(X_train_neg_index), len(X_train_pos_index), _n_ss_sample])
            _n_neg_sample = np.min([len(X_train_neg_index), _n_ss_sample])
            _n_pos_sample = np.min([len(X_train_pos_index), _n_ss_sample])

            sampled_x_neg = \
                X_train_neg_index[np.random.choice(len(X_train_neg_index), size=_n_neg_sample, replace=False)]

            sampled_x_pos = \
                X_train_pos_index[np.random.choice(len(X_train_pos_index), size=_n_pos_sample, replace=True)]

            # 合并两个向量
            ret_index = np.concatenate((sampled_x_neg, sampled_x_pos))

            # 打乱合并后的向量
            np.random.shuffle(ret_index)

            # 抽样
            X_train_sampled, y_train_sampled = X_train[ret_index], y_train[ret_index]
            assert y_train_sampled.mean() == 0.5, "分层抽样结果不正确"

            # 是否要降维
            if n_components is None:
                return X_train_sampled, y_train_sampled, X_test, y_test
            else:
                return DimentionReductionHelper().transform(X_train_sampled, y_train_sampled, X_test, y_test,
                                                            method=dim_redu_method, n_components=n_components)
        elif sample_method == "random":

            train_sample_index = np.random.choice(X_train.shape[0], size=n_samples, replace=True)
            # 抽样
            X_train_sampled, y_train_sampled = X_train[train_sample_index], y_train[train_sample_index]

            # 是否要降维
            if n_components is None:
                return X_train_sampled, y_train_sampled, X_test, y_test
            else:
                return DimentionReductionHelper().transform(X_train_sampled, y_train_sampled, X_test, y_test,
                                                            method=dim_redu_method, n_components=n_components)

        elif sample_method is None:
            # 不抽样，不降维
            if n_components is None:
                return X_train, y_train, X_test, y_test
            else:
                return DimentionReductionHelper().transform(X_train, y_train, X_test, y_test,
                                                            method=dim_redu_method, n_components=n_components)
        else:
            raise RuntimeError(f"不支持的抽样方法： {sample_method}")

    def load_col_dim_data(self, X_train, y_train, X_test, y_test):
        if self.econf.feature_selec_rate >= 1 or self.econf.feature_selec_rate is None:
            # 不降维，直接返回
            return X_train, y_train, X_test, y_test
        else:
            if self.econf.feature_selec_method in ['PCA']:
                return self.data_pca(X_train, y_train, X_test, y_test)
            elif self.econf.feature_selec_method in ['RF']:
                return self.data_fs(X_train, y_train, X_test, y_test)
            else:
                raise RuntimeError(f"Unsupported col_redu_strategy {econf.col_redu_strategy}")

    def load_col_dim_data_v2(self, X_train, y_train, X_test, y_test):
        if self.econf.feature_selec_rate >= 1 or self.econf.feature_selec_rate is None:
            # 不降维，直接返回
            return X_train, y_train, X_test, y_test
        else:
            from tshpo.feature_selection import get_feature_select_method
            clf = get_feature_select_method(self.econf.feature_selec_method)
            return clf.transform(X_train, y_train, X_test, y_test, self.econf)

    def load_sampling_data(self, X_train, y_train, X_test, y_test):
        """
        加载数据的同时抽样+降维

        有放回抽样，保证正负样本量等于max_samples

        max_samples 决定是否抽样
        max_samples=None： 不抽样，使用全部数据
        max_samples= N ： 随机抽取N条数据
        Parameters
        ----------
        max_samples : int or None

        Returns
        -------

        """

        if self.econf.data_sample_rate >= 1 or self.econf.data_sample_rate is None:
            # 不抽样
            return X_train, y_train, X_test, y_test
        else:
            # 抽样
            # n_components = self.check_n_component(X_train, econf)
            econf: ExpConf = self.econf
            sample_method = econf.data_sample_method
            dim_redu_method = econf.feature_selec_method
            n_samples = self.check_n_samples(X_train, econf)

            if sample_method == "stratified":
                _n_ss_sample = int(n_samples / 2)
                X_train_neg_index = np.where(y_train == 0)[0]
                X_train_pos_index = np.where(y_train == 1)[0]

                _n_neg_sample = np.min([len(X_train_neg_index), _n_ss_sample])
                _n_pos_sample = np.min([len(X_train_pos_index), _n_ss_sample])

                sampled_x_neg = \
                    X_train_neg_index[np.random.choice(len(X_train_neg_index), size=_n_neg_sample, replace=False)]

                sampled_x_pos = \
                    X_train_pos_index[np.random.choice(len(X_train_pos_index), size=_n_pos_sample, replace=True)]

                # 合并两个向量
                ret_index = np.concatenate((sampled_x_neg, sampled_x_pos))

                # 打乱合并后的向量
                np.random.shuffle(ret_index)

                # 抽样
                X_train_sampled, y_train_sampled = X_train[ret_index], y_train[ret_index]
                # assert y_train_sampled.mean() == 0.5, "分层抽样结果不正确"

                # 是否要降维, 最多允许3000个样本
                return X_train_sampled[:TSHPOFramework.max_n_rows], y_train_sampled[
                                                                    :TSHPOFramework.max_n_rows], X_test, y_test
            elif sample_method == "random":

                train_sample_index = np.random.choice(X_train.shape[0], size=n_samples, replace=True)
                # 抽样
                X_train_sampled, y_train_sampled = X_train[train_sample_index], y_train[train_sample_index]
                # 是否要降维
                return X_train_sampled[:TSHPOFramework.max_n_rows], y_train_sampled[
                                                                    :TSHPOFramework.max_n_rows], X_test, y_test
            elif sample_method is None or sample_method == "disable":
                # 不抽样，不降维
                return X_train, y_train, X_test, y_test
            else:
                raise RuntimeError(f"不支持的抽样方法： {sample_method}")

    def load_sampling_data_v2(self, X_train, y_train, X_test, y_test):
        """
        加载数据的同时抽样+降维

        有放回抽样，保证正负样本量等于max_samples

        max_samples 决定是否抽样
        max_samples=None： 不抽样，使用全部数据
        max_samples= N ： 随机抽取N条数据
        Parameters
        ----------
        max_samples : int or None

        Returns
        -------

        """

        if self.econf.data_sample_rate is None or self.econf.data_sample_rate >= 1:
            # 不抽样
            return X_train, y_train, X_test, y_test
        else:
            from tshpo.data_sampling import get_data_sampling_method
            clf = get_data_sampling_method(self.econf.data_sample_method)
            return clf.transform(X_train, y_train, X_test, y_test, self.econf)

    def data_pca(self, X_train, y_train, X_test, y_test):
        """
        加载数据的同时抽样+降维

        有放回抽样，保证正负样本量等于max_samples

        max_samples 决定是否抽样
        max_samples=None： 不抽样，使用全部数据
        max_samples= N ： 随机抽取N条数据
        Parameters
        ----------
        max_samples : int or None

        Returns
        -------

        """
        if self.econf.feature_selec_rate == 1:
            return X_train, y_train, X_test, y_test
        econf: ExpConf = self.econf
        sample_method = econf.data_sample_method
        dim_redu_method = econf.feature_selec_method
        n_components = self.check_n_component(X_train, econf)
        # 最多允许30个特征
        n_components = np.min([n_components, TSHPOFramework.max_n_features])

        return DimentionReductionHelper().transform(X_train, y_train, X_test, y_test,
                                                    method=dim_redu_method,
                                                    n_components=n_components)

    def data_fs(self, X_train, y_train, X_test, y_test):
        """

        max_samples 决定是否抽样
        max_samples=None： 不抽样，使用全部数据
        max_samples= N ： 随机抽取N条数据
        Parameters
        ----------
        max_samples : int or None

        Returns
        -------

        """
        if self.econf.feature_selec_rate == 1:
            return X_train, y_train, X_test, y_test
        econf: ExpConf = self.econf
        sample_method = econf.data_sample_method
        dim_redu_method = econf.feature_selec_method
        return FeatureSealectionHelper().transform(X_train, y_train, X_test, y_test,
                                                   method=econf.feature_selec_method,
                                                   feature_selec_ratio=econf.feature_selec_rate)

    @DeprecationWarning
    def load_data_with_feature_selection_and_sample(self):
        """
        加载数据的同时抽样+降维

        有放回抽样，保证正负样本量等于max_samples

        max_samples 决定是否抽样
        max_samples=None： 不抽样，使用全部数据
        max_samples= N ： 随机抽取N条数据
        Parameters
        ----------
        max_samples : int or None

        Returns
        -------

        """
        econf: ExpConf = self.econf
        sample_method = econf.data_sample_method
        dim_redu_method = econf.feature_selec_method

        X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name=econf.dataset, n_fold=econf.folds,
                                                                fold_index=econf.fold_index, seed=econf.random_state)
        n_samples = self.check_n_samples(X_train, econf)

        if sample_method == "stratified":
            _n_ss_sample = int(n_samples / 2)
            X_train_neg_index = np.where(y_train == 0)[0]
            X_train_pos_index = np.where(y_train == 1)[0]

            # _n_real_sample = np.min([len(X_train_neg_index), len(X_train_pos_index), _n_ss_sample])
            _n_neg_sample = np.min([len(X_train_neg_index), _n_ss_sample])
            _n_pos_sample = np.min([len(X_train_pos_index), _n_ss_sample])

            sampled_x_neg = \
                X_train_neg_index[np.random.choice(len(X_train_neg_index), size=_n_neg_sample, replace=False)]

            sampled_x_pos = \
                X_train_pos_index[np.random.choice(len(X_train_pos_index), size=_n_pos_sample, replace=True)]

            # 合并两个向量
            ret_index = np.concatenate((sampled_x_neg, sampled_x_pos))

            # 打乱合并后的向量
            np.random.shuffle(ret_index)

            # 抽样
            X_train_sampled, y_train_sampled = X_train[ret_index], y_train[ret_index]
            assert y_train_sampled.mean() == 0.5, "分层抽样结果不正确"

            # 是否要降维
            if econf.feature_selec_rate is None:
                return X_train_sampled, y_train_sampled, X_test, y_test
            else:
                return FeatureSealectionHelper().transform(X_train_sampled, y_train_sampled, X_test, y_test,
                                                           method=econf.feature_selec_method,
                                                           feature_selec_ratio=econf.feature_selec_rate)
        elif sample_method == "random":

            train_sample_index = np.random.choice(X_train.shape[0], size=n_samples, replace=True)
            # 抽样
            X_train_sampled, y_train_sampled = X_train[train_sample_index], y_train[train_sample_index]

            # 是否要降维
            if econf.feature_selec_rate is None:
                return X_train_sampled, y_train_sampled, X_test, y_test
            else:
                return FeatureSealectionHelper().transform(X_train_sampled, y_train_sampled, X_test, y_test,
                                                           method=econf.feature_selec_method,
                                                           feature_selec_ratio=econf.feature_selec_rate)

        elif sample_method is None:
            # 不抽样，不降维
            if econf.feature_selec_rate is None:
                return X_train, y_train, X_test, y_test
            else:
                return FeatureSealectionHelper().transform(X_train, y_train, X_test, y_test,
                                                           method=econf.feature_selec_method,
                                                           feature_selec_ratio=econf.feature_selec_rate)
        else:
            raise RuntimeError(f"不支持的抽样方法： {sample_method}")

    def check_n_component(self, X_train, econf):
        if econf.feature_selec_rate is not None:
            # 计算要保留的特征数量
            n_components = int(X_train.shape[1] * econf.feature_selec_rate)
            # 将特征数量限制在2到30之间
            n_components = max(n_components, 2)

            # 保证n_components小于 np.min([X_train.shape[0], X_train.shape[1]])
            _tc = np.min([X_train.shape[0], X_train.shape[1]])
            if n_components >= _tc:
                n_components = _tc
        else:
            n_components = None
        return n_components

    def check_n_samples(self, X_train, econf):
        if econf.data_sample_rate is None:
            return X_train.shape[0]
        else:
            return int(X_train.shape[0] * econf.data_sample_rate)

    def load_search_space(self):
        """
        这里我们只处理二分类任务
        Returns
        -------

        """
        econf = self.econf
        assert isinstance(econf.is_trim_sp, bool), "econf.is_trim_sp must be a boolean value"
        if econf.n_high_performing_model is not None and econf.n_high_performing_model < AnaHelper.N_ALGS:

            enable_models = HPTrimHelper.get_high_performing_models(dataset=econf.dataset,
                                                                    metric=econf.metric,
                                                                    top_n=econf.n_high_performing_model)
            cs: ConfigurationSpace = get_auto_sklearn_classification_search_space(y_train=[0, 1],
                                                                                  random_state=econf.random_state,
                                                                                  include=enable_models)

            log.debug(f"Trim CS to:  {cs}")
        else:
            cs: ConfigurationSpace = get_auto_sklearn_classification_search_space(y_train=[0, 1],
                                                                                  random_state=econf.random_state)
            assert len(cs) == 84
            log.debug(f"Use default CS:  {cs}")

        # 修正算法的数量，将-1视为选择全部的模型
        n_algs = cs['__choice__'].num_choices
        econf.n_high_performing_model = n_algs if econf.n_high_performing_model == -1 else n_algs
        assert n_algs == econf.n_high_performing_model, \
            f"Selected models {cs['__choice__']} is not equal to n_high_performing_model {econf.n_high_performing_model}"

        # 如果模型数量等于全部，那么将is_trim_sp 改为False. 因为选择全部算法就表示没有trim_sp 了
        if n_algs == AnaHelper.N_ALGS:
            econf.is_trim_sp = False
        return cs

    def load_search_space_with_given_models(self, enable_models):
        """
        这里我们只处理二分类任务
        Returns
        -------

        """

        econf = self.econf
        cs: ConfigurationSpace = get_auto_sklearn_classification_search_space(y_train=[0, 1],
                                                                              random_state=econf.random_state,
                                                                              include=enable_models)
        return cs

    @staticmethod
    def load_search_space_of_each_model(n_hpys=5):
        """
        在每个算法模型中随机抽取N个配置，而不是在整个超参数空间中随机抽取，因为会导致某些算法没有被抽到
        Returns
        -------

        """
        assert n_hpys is not None, "n_hpys cant be None"
        algs = ExpHelper.get_all_algs()

        outputs = []
        for _alg in algs:
            cs = get_auto_sklearn_classification_search_space(y_train=[0, 1],
                                                              include=[_alg])
            #  先随机抽之后再取前N个
            samples = cs.sample_configuration(size=1000)
            outputs.extend(samples[:n_hpys])
        return outputs

    def load_watch(self):
        return StopWatch()

    def prepare_smac_output_dir(self):
        output_dir = pathlib.Path("smac3_output_tmp")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        return output_dir.absolute().as_posix()

    def dim_redu(self, X_train, y_train, X_test, y_test, econf: ExpConf, min_dim=2, max_dim=20):
        """
        使用不同的方法降维数据。

        Parameters
        ----------
        X_train :
        y_train :
        X_test :
        y_test :
        econf :
        min_dim :
        max_dim :

        Returns
        -------

        """
        n_select_component = int(X_train.shape[1] * econf.important_feature_ratio)
        if n_select_component <= min_dim:
            n_select_component = min_dim

        if n_select_component >= max_dim:
            n_select_component = max_dim

        X_train_dim_redu, _, X_test__dim_redu, _ = DimentionReductionHelper().transform(
            X_train, y_train, X_test, y_test,
            method=econf.dim_redu_method,
            n_components=n_select_component
        )
        assert X_train_dim_redu.shape[1] >= min_dim
        assert X_train_dim_redu.shape[1] <= max_dim
        return X_train_dim_redu, X_test__dim_redu

    @classmethod
    def check_number_of_models(cls, _t):
        assert _t.shape[0] == AnaHelper.N_ALGS, f"模型数量必须是{AnaHelper.N_ALGS}, 收到 {_t.shape[0]}"


# 占位符，兼容smac3, 无其他作用
class MyScenario(Scenario):
    pass


class EXPResults:
    compared_hpo_methods_and_files = [
        ["ROAR", "./04_compare_roar_v2_original_20240911_0808.csv"],
        ["BO", "./04_compare_bo_v2_original_20240911_2337.csv"],
        ["Hyperband", "./04_compare_hyperband_v3_original_20240911_0634.csv"],
        ["Random", "./04_compare_random_v2_original_20240911_0636.csv"],
        ["SMAC", "./04_compare_smac_v3_original_20240912_0945.csv"],
    ]


class PlotNormal:
    @staticmethod
    def normal_acc(key):
        maps = {
            "precision_max": "Precision",
            "recall_max": "Recall",
            "roc_auc_max": "ROC AUC",
            "f1_max": "F1 Score",
        }
        return maps.get(key)


def append_task_id(df: pd.DataFrame):
    """
    在给定的df中加上task_id 列，task_id 是从名字对应到id的。
    给定的df中必须包括dataset列
    Parameters
    ----------
    df :

    Returns
    -------

    """
    return AnaHelper.append_task_id(df)


class DimentionReductionHelper:
    TSNE = "TSNE"
    PCA = "PCA"
    FastICA = "FastICA"
    GRP = "GRP"
    MDS = "MDS"
    # ALL_METHODS = [TSNE, PCA, FastICA, GRP, MDS]
    ALL_METHODS = [PCA, FastICA, GRP, MDS]

    def transform(self, X_train, y_train, X_test, y_test, n_components=2, method="TSNE"):
        """
        X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name="pc1", n_fold=10,
                                                        fold_index=0, seed=42)

        Parameters
        ----------
        X_train :
        y_train :
        X_test :
        y_test :

        Returns
        -------

        """
        assert n_components >= 2
        if method == self.TSNE:
            from sklearn.manifold import TSNE
            lda = TSNE(n_components=n_components)  # 选择降维到2维
            lda.fit(X_train)
            X_train_trans = lda.fit_transform(X_train)
            X_test_trans = lda.fit_transform(X_test)
            return X_train_trans, y_train, X_test_trans, y_test

        elif method == self.PCA:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)  # 选择前20%的主成分
            X_train_trans = pca.fit_transform(X_train)
            X_test_trans = pca.transform(X_test)
            return X_train_trans, y_train, X_test_trans, y_test

        elif method == self.FastICA:
            from sklearn.decomposition import FastICA
            lda = FastICA(n_components=n_components)  # 选择降维到2维
            # 3. 拟合模型并转换数据
            lda.fit(X_train)
            X_train_trans = lda.fit_transform(X_train)
            X_test_trans = lda.fit_transform(X_test)
            return X_train_trans, y_train, X_test_trans, y_test
        elif method == self.GRP:
            from sklearn.random_projection import SparseRandomProjection as SRP
            lda = SRP(n_components=n_components)  # 选择降维到2维
            # 3. 拟合模型并转换数据
            lda.fit(X_train)
            X_train_trans = lda.fit_transform(X_train)
            X_test_trans = lda.fit_transform(X_test)
            return X_train_trans, y_train, X_test_trans, y_test
        elif method == self.MDS:
            from sklearn.manifold import MDS
            lda = MDS(n_components=n_components)  # 选择降维到2维
            # 3. 拟合模型并转换数据
            lda.fit(X_train)
            X_train_trans = lda.fit_transform(X_train)
            X_test_trans = lda.fit_transform(X_test)
            return X_train_trans, y_train, X_test_trans, y_test
        else:
            raise RuntimeWarning("Method %s not supported" % method)


class FeatureSealectionHelper:
    RF = "RF"

    ALL_METHODS = [RF]

    def transform(self, X_train, y_train, X_test, y_test, method="rs", feature_selec_ratio=1):
        """
        X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name="pc1", n_fold=10,
                                                        fold_index=0, seed=42)

        Parameters
        ----------
        X_train :
        y_train :
        X_test :
        y_test :

        Returns
        -------

        """
        if method == self.RF:

            # from sklearn.manifold import TSNE
            # lda = TSNE(n_components=n_components)  # 选择降维到2维
            # lda.fit(X_train)
            # X_train_trans = lda.fit_transform(X_train)
            # X_test_trans = lda.fit_transform(X_test)
            # return X_train_trans, y_train, X_test_trans, y_test
            #

            # 创建随机森林模型
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # 计算阈值
            feature_importances = model.feature_importances_
            threshold = np.percentile(feature_importances, 100 * (1 - feature_selec_ratio))

            indices = np.where(feature_importances >= threshold)[0]

            # 最多允许30个特征
            indices = indices[: TSHPOFramework.max_n_features]
            return X_train[:, indices], y_train, X_test[:, indices], y_test


        else:
            raise RuntimeWarning("Method %s not supported" % method)


def load_compare_data_with_or_without_trim(method_name,
                                           file_name="./04_compare_roar_v2_original_20240911_0808.csv",
                                           metric=AnaHelper.METRIC_ROC_AUC):
    df = pd.read_csv(file_name)

    outputs = []
    for _dataset_name, v in df.groupby(by='dataset'):
        _baseline = v[(v['is_trim_sp'] == False) & (v['metric'] == metric)]
        assert _baseline.shape[0] <= 10, "参与验证的数量错误，必须是10折交叉验证的"
        _compare = v[(v['is_trim_sp'] == True) & (v['metric'] == metric)]

        for _n_high_performing_model, _compare_item in _compare.groupby(by='n_high_performing_model'):
            assert _compare_item.shape[0] <= 10, "参与验证的数量错误，必须是10折交叉验证的"
            outputs.append({
                "metric": metric,
                "method_name": method_name,
                "dataset": _dataset_name,
                "mean_f": _baseline['default'].mean(),
                "std_f": _baseline['default'].std(),
                "mean_t": _compare_item['default'].mean(),
                "std_t": _compare_item['default'].std(),
                "n_high_performing_model": _n_high_performing_model
            })

        # g_df = df.groupby(by=['dataset', 'is_trim_sp']).mean()
        # g_df.reset_index(inplace=True)
        # plot_data = pd.melt(g_df, id_vars=['dataset', 'is_trim_sp'], value_vars=['default'])
        # plot_data = append_task_id(plot_data)
        # plot_data['method_name'] = method_name
        # plot_data['n_high_performing_model'] = n_high_performing_model
        # return plot_data
    return pd.DataFrame(outputs)


class TSHPOFramework:
    """
    实验定义
    """
    max_n_features = 30
    max_n_rows = 3000

    @staticmethod
    def get_overall_results(debug=is_macos(), max_iterations=100):
        random_state = 42
        n_folds = 10
        configs = []
        datasets = get_small_datasets()[:5]
        metrics = AnaHelper.get_all_metrics()
        if debug:
            datasets = datasets[:1]
        for _dataset in datasets:
            for _metric in metrics:
                for fold_index in range(10):
                    for _max_iterations in [100]:
                        # 选择不同的TOP模型时精度, -1 表示不选模型，是基准
                        for _n_high_performing_model in [-1, 2, 4, 6, 8, 10, 12, 14]:
                            configs.append(ExpConf(dataset=_dataset,
                                                   folds=n_folds,
                                                   fold_index=fold_index,
                                                   random_state=random_state,
                                                   bo_n_warm_start=int(max_iterations * 0.1),
                                                   wall_time_limit_in_s=-1,
                                                   debug=debug,
                                                   metric=_metric,
                                                   max_iterations=_max_iterations,
                                                   is_trim_sp=True if _n_high_performing_model != -1 else False,
                                                   n_high_performing_model=_n_high_performing_model
                                                   ))

        return configs

    @staticmethod
    def gather_files(files, config_name):
        log.info("Exps is finished")

        # gather_file_home = pathlib.Path('outputs', pathlib.Path(econfigs[0].entry_file_name).stem)
        gather_file_home = pathlib.Path('outputs', pathlib.Path(config_name).stem)
        gather_file_name = pathlib.Path(gather_file_home,
                                        gather_file_home.stem + f"_original_{get_str_datetime()}.csv.gz").absolute().as_posix()
        log.error(f"Gathering files is saved to  {gather_file_name}")
        all_metrics = gather_all_csv_files(files)
        all_metrics.to_csv(gather_file_name)
        if not is_macos():
            # MyQiniu().upload_file(gather_file_name)
            # 上传到本地服务器，而不是七牛云
            Rsync.upload_file(Servers.S100_9_HOST, local_file=gather_file_name)

        return all_metrics, gather_file_home

    @staticmethod
    def start(_run, configs, debug):
        print(f"Total tasks: {len(configs)}")
        debug = debug if not is_macos() else True
        print(configs[0])
        flag = input("Check the task info, continue run(y|n)?")
        if flag.lower() != "y":
            return None, None
        print("Exps starting ...")
        if sys.platform == "darwin" and GlobalConfig.OUTPUT_HOME.exists():
            shutil.rmtree(GlobalConfig.OUTPUT_HOME.as_posix())

        if debug:
            out_files = []
            for _c in configs:
                metric_file = _run(_c)
                out_files.append(metric_file)

                print(1)
            return None, None
        else:
            parallel_jobs = int(os.cpu_count() * 0.7)
            out_files = Parallel(n_jobs=parallel_jobs)(
                delayed(_run)(econf) for econf in
                tqdm(configs, total=len(configs), leave=True, desc="Tasks", ncols=88, position=0))
            log.info("Exps is finished")
        gather_file_home = pathlib.Path('outputs', pathlib.Path(configs[0].config_file_name).stem)
        gather_file_name = pathlib.Path(gather_file_home,
                                        gather_file_home.stem + f"_original_{get_str_datetime()}.csv.gz").absolute().as_posix()
        print(f"Gathering files is saved to  {gather_file_name}")
        all_metrics = gather_all_csv_files(out_files)
        all_metrics.to_csv(gather_file_name)

        if not is_macos():
            # MyQiniu().upload_file(gather_file_name)
            # 上传到本地服务器，而不是七牛云
            # from pyutils.util_servers import Servers
            Rsync.upload_file(Servers.S100_9_HOST, local_file=gather_file_name)

        return all_metrics, gather_file_home

    @staticmethod
    def prepare_resources(econf: ExpConf):

        assert econf.fold_index is not None, "Fold index cannot be None"
        assert econf.max_iteration is None or econf.max_iteration >= 10, "Max iterations must larger than 10"
        # Step0: prepare env
        eh = ExpHelper(econf)
        watch = eh.load_watch()
        cs = eh.load_search_space()
        history = TrainingHistory(cs=cs, seed=econf.random_state, stop_watch=watch, econf=econf)
        # step1: data processing
        watch.start(Steps.DATA_PROCESSING)
        # 加载原始数据
        X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name=econf.dataset, n_fold=econf.folds,
                                                                fold_index=econf.fold_index, seed=econf.random_state)
        # 行抽样
        # X_train, y_train, X_test, y_test = eh.load_sampling_data(X_train, y_train, X_test, y_test)
        X_train, y_train, X_test, y_test = eh.load_sampling_data_v2(X_train, y_train, X_test, y_test)

        # 列降维
        X_train, y_train, X_test, y_test = eh.load_col_dim_data_v2(X_train, y_train, X_test, y_test)

        watch.stop(Steps.DATA_PROCESSING)

        return X_train, y_train, X_test, y_test, cs, history, watch

    @staticmethod
    def prepare_resources_with_pruned_cs(econf: ExpConf, pruned_cs: ConfigurationSpace, is_reduced_data=False):
        """准备环境，cs要手动给定
        is_reduced_data：是否预先处理数据
        """
        assert econf.fold_index is not None, "Fold index cannot be None"
        assert econf.max_iteration is None or econf.max_iteration >= 10, "Max iterations must larger than 10"
        # Step0: prepare env
        eh = ExpHelper(econf)
        watch = eh.load_watch()
        history = TrainingHistory(cs=pruned_cs, seed=econf.random_state, stop_watch=watch, econf=econf)
        # step1: data processing
        watch.start(Steps.DATA_PROCESSING)
        # 加载原始数据
        X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name=econf.dataset, n_fold=econf.folds,
                                                                fold_index=econf.fold_index, seed=econf.random_state)
        if is_reduced_data:
            # 行抽样
            # X_train, y_train, X_test, y_test = eh.load_sampling_data(X_train, y_train, X_test, y_test)
            X_train, y_train, X_test, y_test = eh.load_sampling_data_v2(X_train, y_train, X_test, y_test)

            # 列降维
            X_train, y_train, X_test, y_test = eh.load_col_dim_data_v2(X_train, y_train, X_test, y_test)
        watch.stop(Steps.DATA_PROCESSING)
        return X_train, y_train, X_test, y_test, history, watch

    @staticmethod
    def prepare_resources_with_pruned_cs_without_data_reduced(econf: ExpConf, pruned_cs: ConfigurationSpace):
        """准备环境，cs要手动给定
        is_reduced_data：是否预先处理数据
        """
        assert econf.fold_index is not None, "Fold index cannot be None"
        assert econf.max_iteration is None or econf.max_iteration >= 10, "Max iterations must larger than 10"
        # Step0: prepare env
        eh = ExpHelper(econf)
        watch = eh.load_watch()
        history = TrainingHistory(cs=pruned_cs, seed=econf.random_state, stop_watch=watch, econf=econf)
        # step1: data processing
        watch.start(Steps.DATA_PROCESSING)
        # 加载原始数据
        X_train, y_train, X_test, y_test = load_dataset_at_fold(dataset_name=econf.dataset, n_fold=econf.folds,
                                                                fold_index=econf.fold_index, seed=econf.random_state)
        watch.stop(Steps.DATA_PROCESSING)
        return X_train, y_train, X_test, y_test, history, watch

    @staticmethod
    def update_history(info: TrialInfo, run_value: RunValue, optimizer, econf: ExpConf, history: TrainingHistory):

        # 结构保存为2部分，第一部分是我的数据结构，便于持久化；第二部分是SMAC3的信息，用于调用SMAC3框架

        # 第一部分，我的数据就够，持久化
        history.add_history(info.config, run_value)

        # 第二部分，这里需要处理最大值和最小值问题
        # TrialValue(cost=9.2466, time=0.0447, status=<StatusType.SUCCESS: 1>, starttime=0.0, endtime=0.0, additional_info={})
        trial_value = TrialValue(cost=run_value.default, time=run_value.elapsed_seconds)
        optimizer.tell(info, trial_value)

    @staticmethod
    def get_optimizer(econf: ExpConf, cs, history):

        def _train_placeholder(config=None, seed=None, budget=None):  # 占位符，兼容smac3, 无实际作用
            pass

        scenario = Scenario(cs, deterministic=True, n_trials=1, min_budget=10, max_budget=100,
                            seed=econf.random_state, )

        initial_design = RandomInitialDesign(scenario, n_configs=None, n_configs_per_hyperparameter=10,
                                             max_ratio=0.25, additional_configs=None, seed=econf.random_state)

        if econf.hpo_opt_method == OptMethod.Hyperband.value:
            intensifier = HyperbandFacade.get_intensifier(
                scenario,
            )
            smac = HyperbandFacade(scenario,
                                   _train_placeholder,
                                   intensifier=intensifier,
                                   initial_design=initial_design,
                                   overwrite=True,
                                   logging_level=False)
        elif econf.hpo_opt_method == OptMethod.BayesianOptimization.value:
            smac = MyBayesionOptimization(history, cs)
        elif econf.hpo_opt_method == OptMethod.Random.value:
            smac = MyRandomSearch(history, cs)
        elif econf.hpo_opt_method == OptMethod.SOAR.value:
            smac = RandomFacade(scenario, _train_placeholder, logging_level=False, overwrite=True)
        elif econf.hpo_opt_method == OptMethod.SMAC.value:
            intensifier = HyperparameterOptimizationFacade.get_intensifier(
                scenario,
                max_config_calls=100,  # We basically use one seed per config only
            )
            # Now we use SMAC to find the best hyperparameters
            smac = HyperparameterOptimizationFacade(
                scenario,
                _train_placeholder,
                intensifier=intensifier,
                initial_design=initial_design,
                overwrite=True,
                logging_level=False,
            )
            # smac = HyperparameterOptimizationFacade(scenario, _train_placeholder, logging_level=False, overwrite=True)
        else:
            raise RuntimeError(f"Unsupported optimization method: {econf.hpo_opt_method}")
        return smac


@dataclass
class MyBayesionOptimization:
    """
    最小化任务
    """
    history: TrainingHistory
    cs: ConfigurationSpace

    # the hyperparameters of GP
    surrogate_model: callable = None
    kappa: float = 2.576
    size_of_candidate_samples: int = 1000  # 随机抽样时抽取多少个来评估呢？

    # 用来预热的超参数优化的数量
    n_warmuu_sample: int = 15

    def __post_init__(self):
        if self.surrogate_model is None:
            self.surrogate_model = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=self.history.seed,
            )

        # 初始化15个超参数
        self.init_configs = self.cs.sample_configuration(size=self.n_warmuu_sample)

    def optimize(self):

        """
        贝叶斯优化，转为最小化问题了.
        历史数据存在th中了，相当于tell在history中，这里只负责ask

        Parameters
        ----------
        th :
        cs :
        surrogate_model :
        random_state :

        Returns
        -------

        """
        # 初始化默认 surrogate model
        log.debug("Start suggestion by BO ...")
        # 如果训练历史为空，就随机返回一个配置
        th = self.history
        cs = self.cs
        if th.is_empyt():
            return cs.sample_configuration(size=1)

        train_history_array = th.get_array_history()
        candidate_samples = cs.sample_configuration(size=self.size_of_candidate_samples)

        train_x = train_history_array[:, 0:-1]
        train_y = train_history_array[:, -1]

        candidate_x = th.get_array_of_configurations(candidate_samples)

        # 加个负号，转为最小值问题
        self.surrogate_model.fit(train_x, -train_y)

        acq = UtilityFunction.ucb(
            x=candidate_x,
            kappa=self.kappa,
            gp=self.surrogate_model
        )
        best_candidata_index = acq.argmax()
        next_config = candidate_samples[best_candidata_index]
        log.debug("Suggest of PO is done")
        # 是否选择模型的超参数
        return next_config

    def tell(self, info: TrialInfo, value: TrialValue, save: bool = True) -> None:
        """
        tell 的工作已经迁移到history中，在history 中已经做了tell的工作，这里不需要再重新添加
        Parameters
        ----------
        info :
        value :
        save :

        Returns
        -------

        """
        pass

    def ask(self):
        if self.history.count < self.n_warmuu_sample:
            # warm up
            return TrialInfo(config=self.init_configs[self.history.count])
        else:
            # optimization
            _config = self.optimize()
            return TrialInfo(config=_config)


@dataclass
class MyRandomSearch:
    history: TrainingHistory
    cs: ConfigurationSpace
    counter: int = -1

    def __post_init__(self):
        self.counter = -1
        self.configs = self.cs.sample_configuration(size=10000)

    def tell(self, info: TrialInfo, value: TrialValue, save: bool = True) -> None:
        pass

    def ask(self):
        self.counter += 1
        return TrialInfo(config=self.configs[self.counter])


if __name__ == '__main__':
    search_space = ExpHelper.load_search_space_of_each_model(2)
    HPTrimHelper.generate_outputs(
        TSHPOCommon.get_result_data("c09_select_optimal_alg_v2_original_20241031_1352.csv.gz"))
