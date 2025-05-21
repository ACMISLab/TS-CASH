import os.path
from enum import Enum


# OptMetricsType
class Metric:
    metric_train_loss = "metric_train_loss".upper()
    metric_valid_loss = "metric_valid_loss".upper()
    metric_test_loss = "metric_test_loss".upper()
    metric_valid_affiliation_f1 = "metric_valid_affiliation_f1".upper()
    metric_valid_affiliation_precision = "metric_valid_affiliation_precision".upper()
    metric_valid_affiliation_recall = "metric_valid_affiliation_recall".upper()
    metric_valid_point_wise_precision = "metric_valid_point_wise_precision".upper()
    metric_valid_point_wise_recall = "metric_valid_point_wise_recall".upper()
    metric_valid_point_wise_f1 = "metric_valid_point_wise_f1".upper()
    metric_valid_point_adjusted_precision = "metric_valid_point_adjusted_precision".upper()
    metric_valid_point_adjusted_recall = "metric_valid_point_adjusted_recall".upper()
    metric_valid_point_adjusted_f1 = "metric_valid_point_adjusted_f1".upper()
    metric_valid_revised_point_adjusted_precision = "metric_valid_revised_point_adjusted_precision".upper()
    metric_valid_revised_point_adjusted_recall = "metric_valid_revised_point_adjusted_recall".upper()
    metric_valid_revised_point_adjusted_f1 = "metric_valid_revised_point_adjusted_f1".upper()
    metric_test_affiliation_f1 = "metric_test_affiliation_f1".upper()
    metric_test_affiliation_precision = "metric_test_affiliation_precision".upper()
    metric_test_affiliation_recall = "metric_test_affiliation_recall".upper()
    metric_test_point_wise_precision = "metric_test_point_wise_precision".upper()
    metric_test_point_wise_recall = "metric_test_point_wise_recall".upper()
    metric_test_point_wise_f1 = "metric_test_point_wise_f1".upper()
    metric_test_point_adjusted_precision = "metric_test_point_adjusted_precision".upper()
    metric_test_point_adjusted_recall = "metric_test_point_adjusted_recall".upper()
    metric_test_point_adjusted_f1 = "metric_test_point_adjusted_f1".upper()
    metric_test_revised_point_adjusted_precision = "metric_test_revised_point_adjusted_precision".upper()
    metric_test_revised_point_adjusted_recall = "metric_test_revised_point_adjusted_recall".upper()
    metric_test_revised_point_adjusted_f1 = "metric_test_revised_point_adjusted_f1".upper()
    metric_time_app_start = "metric_time_app_start".upper()
    metric_time_app_end = "metric_time_app_end".upper()
    metric_time_train_start = "metric_time_train_start".upper()
    metric_time_train_end = "metric_time_train_end".upper()
    metric_time_eval_start = "metric_time_eval_start".upper()
    metric_time_eval_end = "metric_time_eval_end".upper()
    metric_default = "default".upper()


# Keys For Metric
class KMC:
    # Training time with seconds
    TRAINING_TIME_SECONDS = "TRAINING_TIME_SECOND"
    TRAINING_TIME_MINUTE = "TRAINING_TIME_MINUTE"
    # Training time with nanoseconds
    TRAINING_TIME = "TRAINING_TIME"

    METRIC_TRAIN_LOSS = "METRIC_TRAIN_LOSS"
    METRIC_VALID_LOSS = "METRIC_VALID_LOSS"
    METRIC_TEST_LOSS = "METRIC_TEST_LOSS"
    METRIC_VALID_AFFILIATION_F1 = "METRIC_VALID_AFFILIATION_F1"
    METRIC_VALID_AFFILIATION_PRECISION = "METRIC_VALID_AFFILIATION_PRECISION"
    METRIC_VALID_AFFILIATION_RECALL = "METRIC_VALID_AFFILIATION_RECALL"
    METRIC_VALID_POINT_WISE_PRECISION = "METRIC_VALID_POINT_WISE_PRECISION"
    METRIC_VALID_POINT_WISE_RECALL = "METRIC_VALID_POINT_WISE_RECALL"
    METRIC_VALID_POINT_WISE_F1 = "METRIC_VALID_POINT_WISE_F1"
    METRIC_VALID_POINT_ADJUSTED_PRECISION = "METRIC_VALID_POINT_ADJUSTED_PRECISION"
    METRIC_VALID_POINT_ADJUSTED_RECALL = "METRIC_VALID_POINT_ADJUSTED_RECALL"
    METRIC_VALID_POINT_ADJUSTED_F1 = "METRIC_VALID_POINT_ADJUSTED_F1"
    METRIC_VALID_REVISED_POINT_ADJUSTED_PRECISION = "METRIC_VALID_REVISED_POINT_ADJUSTED_PRECISION"
    METRIC_VALID_REVISED_POINT_ADJUSTED_RECALL = "METRIC_VALID_REVISED_POINT_ADJUSTED_RECALL"
    METRIC_VALID_REVISED_POINT_ADJUSTED_F1 = "METRIC_VALID_REVISED_POINT_ADJUSTED_F1"
    METRIC_TEST_AFFILIATION_F1 = "METRIC_TEST_AFFILIATION_F1"
    METRIC_TEST_AFFILIATION_PRECISION = "METRIC_TEST_AFFILIATION_PRECISION"
    METRIC_TEST_AFFILIATION_RECALL = "METRIC_TEST_AFFILIATION_RECALL"
    METRIC_TEST_POINT_WISE_PRECISION = "METRIC_TEST_POINT_WISE_PRECISION"
    METRIC_TEST_POINT_WISE_RECALL = "METRIC_TEST_POINT_WISE_RECALL"
    METRIC_TEST_POINT_WISE_F1 = "METRIC_TEST_POINT_WISE_F1"
    METRIC_TEST_POINT_ADJUSTED_PRECISION = "METRIC_TEST_POINT_ADJUSTED_PRECISION"
    METRIC_TEST_POINT_ADJUSTED_RECALL = "METRIC_TEST_POINT_ADJUSTED_RECALL"
    METRIC_TEST_POINT_ADJUSTED_F1 = "METRIC_TEST_POINT_ADJUSTED_F1"
    METRIC_TEST_REVISED_POINT_ADJUSTED_PRECISION = "METRIC_TEST_REVISED_POINT_ADJUSTED_PRECISION"
    METRIC_TEST_REVISED_POINT_ADJUSTED_RECALL = "METRIC_TEST_REVISED_POINT_ADJUSTED_RECALL"
    METRIC_TEST_REVISED_POINT_ADJUSTED_F1 = "METRIC_TEST_REVISED_POINT_ADJUSTED_F1"
    METRIC_TIME_APP_START = "METRIC_TIME_APP_START"
    METRIC_TIME_APP_END = "METRIC_TIME_APP_END"
    METRIC_TIME_TRAIN_START = "METRIC_TIME_TRAIN_START"
    METRIC_TIME_TRAIN_END = "METRIC_TIME_TRAIN_END"
    METRIC_TIME_EVAL_START = "METRIC_TIME_EVAL_START"
    METRIC_TIME_EVAL_END = "METRIC_TIME_EVAL_END"
    METRIC_DEFAULT = "DEFAULT"


# Experiment Information
class EI(Metric):
    # the pure HPs for algorithm, such as:  {'window_size': 12.659870508216075, 'n_estimators': 319.28829052624206}
    DATA_ID_SHORT = "DATA_ID_SHORT"
    MODEL = "TRIALCMD_MODEL"
    HPS = "HPS"

    # The HPs for NNI: {"parameter_id": 7,
    # "parameter_source": "algorithm",
    # "parameters": {"window_size": 25.616942433075963},
    # "parameter_index": 0}
    HYPERPARAMETER = "HYPERPARAMETER"

    DATA_SAMPLE_RATE_STR = "DATA_SAMPLE_RATE_STR"
    DATA_SAMPLE_RATE_MATH_STR = "DATA_SAMPLE_RATE_MATH_STR"
    EXPID = "EXPID"
    EXPINFO_MAXTRIALNUMBER = "EXPINFO_MAXTRIALNUMBER"
    MAXTRIALNUMBER = EXPINFO_MAXTRIALNUMBER
    timestamp = "timestamp"
    trialJobId = "trialJobId"
    parameterId = "parameterId"
    type = "type"
    sequence = "sequence"
    metrics = "metrics"
    hyperparameter = "hyperparameter"
    logpath = "logpath"
    event = "event"
    expinfo = "expinfo"
    expid = "expid"
    short_exp_name = "short_exp_name"
    EXP_NAME = "EXP_NAME"
    DATA_ID = "DATA_ID"
    SEED = "SEED"
    DATA_SAMPLE_RATE = "DATA_SAMPLE_RATE"

    expinfo_experimentName = "expinfo_experimentName"
    expinfo_experimentType = "expinfo_experimentType"
    expinfo_searchSpaceFile = "expinfo_searchSpaceFile"
    expinfo_searchSpace = "expinfo_searchSpace"
    expinfo_trialCommand = "expinfo_trialCommand"
    expinfo_trialCodeDirectory = "expinfo_trialCodeDirectory"
    expinfo_trialConcurrency = "expinfo_trialConcurrency"
    expinfo_maxTrialNumber = "expinfo_maxTrialNumber"
    expinfo_useAnnotation = "expinfo_useAnnotation"
    expinfo_debug = "expinfo_debug"
    expinfo_logLevel = "expinfo_logLevel"
    expinfo_experimentWorkingDirectory = "expinfo_experimentWorkingDirectory"
    expinfo_tuner = "expinfo_tuner"
    expinfo_trainingService = "expinfo_trainingService"
    trialcmd_dataset = "trialcmd_dataset"
    trialcmd_data_id = "trialcmd_data_id"
    trialcmd_seed = "trialcmd_seed"
    trialcmd_data_sample_method = "trialcmd_data_sample_method"
    trialcmd_data_sample_rate = "trialcmd_data_sample_rate"
    trialcmd_data_sample_rate_str = "trialcmd_data_sample_rate_str"
    trialcmd_model = "trialcmd_model"
    trialcmd_gpustrialJobId = "trialcmd_gpustrialJobId"
    trialcmd_gpus = "trialcmd_gpus"


class KEI(EI):
    pass


class SampleType:
    """
    Randomly sampling in whole set.
    """

    RANDOM = "random"

    """
    Sample training set in the normal rolling windows. 
    For the abnormal rolling windows, we do not sample, since the number of it is small. 
    
    """
    NORMAL_RANDOM = "normal_random"

    """
    Equally sampling from the normal rolling windows and abnormal rolling windows
    """
    STRATIFIED = "stratified"

    """
    Latin Hypercube Sampling (‘lhs’)
    """
    LHS = "lhs"

    """
    Distance1, 这个距离考虑两个方法:最值和距离. 
    一方面,考虑每个窗口的95%的最大值和5%的最小值(防止异常值). 
    另一方面,考虑窗口与0的距离.
    
    具体实现:(假如x是一个窗口)
    _max = np.percentile(x1, 95, axis=1)
    _min = np.percentile(x1, 5, axis=1)
    # 考虑不同区间,平方防止正反相减
    _f1 = np.power(_max, 2) + np.power(_min, 2)

    _dist = np.sum(np.power(x1 - 0, 2), axis=1)
    # 考虑窗口与0的距离
    _f2 = _f1 + _dist

    # 归一化
    return (_f2 - _f2.min()) / (_f2.max() - _f2.min())
    """
    DIST1 = "dist1"


class ConstMetric:
    #  timestamp   INTEGER,
    #  trialJobId  TEXT,
    #  parameterId TEXT,
    #  type        TEXT,
    #  sequence    INTEGER,
    KEY_METRIC_AUC = "AUC"
    KEY_METRIC_MSE = "MSE"
    KEY_TIMESTAMP = "timestamp"
    KEY_TRIAL_JOB_ID = "trialJobId"
    KEY_PARAMETER_ID = "parameterId"
    KEY_TRIAL_COMMAND = "trialCommand"
    KEY_TYPE = "type"
    KEY_SEQUENCE = "sequence"
    KEY_TEST_LOSS = "TEST_LOSS"
    KEY_VALID_LOSS = "VALID_LOSS"
    KEY_AUC = "AUC"
    KEY_DEFAULT = "default"
    KEY_MODEL_ID = "model_id"
    KEY_BEST_THRESHOLD = "best_threshold"
    KEY_BEST_RECALL = "best_recall"
    KEY_BEST_PRECISION = "best_prec"
    KEY_BEST_F1 = "best_f1"
    KEY_VALUE_FINAL = 'FINAL'
    KEY_TUNER = "nni_tuner"
    KEY_EXP_NAME = 'exp_name'
    KEY_EXP_ID = 'exp_id'
    KEY_TRIAL_CMD = "trial_cmd"
    KEY_RUNNING_TIME = "running_time"
    KEY_ALERT_DELAY = "alert_delay"
    KEY_DATASET = 'm_dataset'
    KEY_TRAINING_SET_RATE_FLOAT = 'training_set_rate_float'
    KEY_TRAINING_SET_RATE = 'training_set_rate'
    KEY_DATA_SAMPLE_RATE = "data_sample_rate"

    BEST_THRESHOLD = "best_threshold"
    BEST_RECALL = "best_recall"
    BEST_PRECISION = "best_prec"
    BEST_F1 = "best_f1"
    RUNNING_TIME = "running_time"
    TRIAL_CMD = "trial_cmd"
    FINAL_BEST_F1 = "best_f1"
    INTER_ALERT_DELAY_ON_BEST_F_CORE = "alert_delay_on_best_fscore"
    INTER_BEST_THRESHOLD = "inter_best_threshold"
    INTER_BEST_RECALL = "inter_best_recall"
    INTER_BEST_PREC = "inter_best_precision"
    INTER_BEST_FSCORE = "inter_best_fscore"
    K_FOLD_INDEX = "k_fold_index"
    ALERT_DELAY = "alert_delay"
    AUC = "auc"


class ConstKeras:
    KEY_LOSS = 'loss'
    KEY_TRAINING_LOSS = 'loss'
    KEY_VAL_LOSS = "val_loss"


class ConstTuner:
    MLHS_RNS = "MLHSRNS"
    LHS_RNS = "LHSRNS"
    LHS_RRS = "LHSRRS"
    RS_RNS = "RSRNS"
    RS_RRS = "RSRRS"
    RANDOM = "Random"
    TPE = "TPE"
    EVOLUTION = "Evolution"

    # Sample base method
    DDS = "DDS"
    HALTON = "HALTON"
    LHS = "LHS"
    RS = "RS"
    SCBOL = "SCBOL"
    ALL_SAMPLE_TUNER = [DDS, HALTON, LHS, RS, SCBOL]


class ConstDataset(Enum):
    SAMPLING_RANDOM = "RS"


class Config:
    # The exp_index of the GPU to use for training. e.g., str "0,1" means to use the first and second GPU to train the
    # model. Multi-GPUs can be used by comma separated. None means to use all the GPUs.
    ARGS_KEY_DEVICE = "--device"
    ARGS_KEY_VALID_RATE = "--valid_rate"
    ARGS_KEY_CONFIG_SIZE = "--test_size"
    ARGS_KEY_GPUS = "--gpus"

    # Debug mode
    ARGS_KEY_DEBUG = "--debug"
    ARGS_KEY_HPCONFIG = "--hps"
    PAPER_01_DIR = "/Users/sunwu/SyncResearch/01P_HOSearchBased"
    KEY_Y = "y"
    KEY_X = "X"

    ARGS_KEY_EPOCHS = "--epochs"
    ARGS_KEY_EXP_NAME = "--exp_name"
    ARGS_KEY_THRESHOLD = "--threshold"
    ARGS_KEY_SEARCH_SPACE_FILE = "--search_space_file"
    ARGS_KEY_TEST_FILE_OPTION = "--test"
    ARGS_KEY_DATASET = "--dataset"
    ARGS_KEY_DATA_ID = "--data_id"
    ARGS_KEY_MODEL = "--model"
    ARGS_KEY_HOME = "--home"
    ARGS_KEY_CHECKPOINT_INTERVAL = "--check_point_interval"
    ARGS_KEY_SEED = "--seed"
    ARGS_KEY_DATA_SAMPLE_METHOD = "--data_sample_method"
    ARGS_KEY_DATA_SAMPLE_RATE = "--data_sample_rate"
    ARGS_MAX_TRIAL_NUM = "--max_trial_number"
    ARGS_KEY_NNI_TUNER = "--nni_tuner"

    KEY_HYPER_PARAMETERS = "parameters"
    # the search space space dir for nni
    NNI_EXP_SEARCH_SPACE_DIR = "search_spaces"
    NNI_CREATING_EXP_KEY_WORDS = "nnictl stop --all"
    NNI_DB_NAME = "nni.sqlite"
    NNI_STATUS_DONE = "DONE"
    NNI_JOB_ERROR_KEY_WORDS = "is not idle"
    NNI_RUNNING_KEY_WORDS = "RUNNING"
    NNI_RUNNING_SCRIPT_NAME = "running_cmd.sh"
    NNI_EXPERIMENT_SAVE_DIR = "nni-experiments"
    NNI_EXPERIMENTS = "experiments"

    EXP_DIRECTORY_SAVE_HOME = os.path.join("experiments", "configs")

    @staticmethod
    def get_values(args, key):
        """
        通过参数选项(如 -file) 来获取解析到的 args 中配置的值，常见用法：
        AppLunbo.get_values(args,AppLunbo.ARGS_KEY_DATASET)

        Parameters
        ----------
        args : argparse.Namespace
            通过 parser.parse_and_run() 获取到的 Namespase
        key : str
            要获取的值的 key， 如 -file

        Returns
        -------

        """
        return args.__dict__[key[1:]]


class Emjoi:
    FINISHED = "🚩"
    START = "🚀"
    SPARKLES_TRIBLE = "✨✨✨"
    WAITING = "🕗"
    COMMAND = "📄"
    SUCCESS = "✅"
    FAILED = "❌"
    ERROR = "❌"
    SPARKLES = "✨"
    WARNING = "🚨"
    CAR = "🚗"
    SETTING = "🔧"
    SEARCHING = "🔎"
    METRIC = "👁️"
    INFO = "🖥️"
    OK = "✅"


class HyperParameter:
    SLIDING_WINDOW = 'sliding_window'
    INPUT_DIM = None
    OUTPUT_FEATURE = "output_features"
    L2_REGULARIZER = "l2_regularizer"
    NUMBER_OF_FEATURE = "n_features"
    LAMBDA2 = "lambda2"
    LAMBDA1 = 'lambda1'
    DROP_RATE = 'drop_rate'
    ESTIMATION_ACTIVATION = 'est_acti'
    NUMBER_OF_ESTIMATION_NEURONS = 'n_est_neurons'
    N_LAYERS = "n_layers"
    N_NEURONS = "n_neurons"
    N_NEURONS_LAYER1 = "n_neurons_layer1"
    N_NEURONS_LAYER2 = "n_neurons_layer2"
    N_NEURONS_LAYER3 = "n_neurons_layer3"
    NUMBER_OF_ESTIMATION_LAYERS = 'n_est_layers'
    COMPRESSION_ACTIVATION = 'comp_acti'
    NUMBER_OF_COMPRESSION_LATENT_DIM = "n_comp_lateten_z"
    NUMBER_OF_COMPRESSION_NEURALS = 'n_comp_neurons'
    NUMBER_OF_COMPRESSION_HIDDEN_NETWORK_LAYERS = "n_comp_layers"
    BATCH_SIZE: str = "batch_size"
    WINDOW_SIZE = "window_size"
    LATENT_DIM = "latent_dim"
    EPOCHS = "epochs"
    ACTIVATION_FUNCTION = "activation_function"
    HIDDEN_NEURONS = "hidden_neurons"
    L2_FACTOR = 'l2_factor'
    MISSING_DATA_INJECTION_RATE = "missing_data_injection_rate"
    # LEARNING_RETE = "learning_rate"
    LEARNING_RATE = "learning_rate"
    LR_ANNEAL_EPOCHS = "lr_anneal_epochs"
    LR_ANNEAL_FACTOR = "lr_anneal_factor"
    GRAD_CLIP_NORM = "grad_clip_norm"
    NUM_L_SAMPLES = "num_l_samples"
    HIDDEN_ACTIVATION = "hidden_activation"
    ENCODER_NEURONS = "encoder_neurons"
    DECODER_NEURONS = "decoder_neurons"
    OUTPUT_ACTIVATION = "output_activation"
    INPUT_DIM = "input_dim"

    @staticmethod
    def valid_search_space(search_space):
        """
        Check the search space, to:
        1. correct the user,e.g., convert float to int

        Parameters
        ----------
        search_space : dict
            the search space

        Returns
        -------

        """
        int_hyper_parameter = [
            HyperParameter.EPOCHS,
            HyperParameter.BATCH_SIZE,
            HyperParameter.N_NEURONS,
            HyperParameter.N_LAYERS,
            HyperParameter.LATENT_DIM,
            HyperParameter.OUTPUT_FEATURE,
            HyperParameter.NUMBER_OF_COMPRESSION_HIDDEN_NETWORK_LAYERS,
            HyperParameter.NUMBER_OF_COMPRESSION_NEURALS,
            HyperParameter.NUMBER_OF_COMPRESSION_LATENT_DIM,
            HyperParameter.NUMBER_OF_ESTIMATION_LAYERS,
            HyperParameter.NUMBER_OF_ESTIMATION_NEURONS,
            HyperParameter.N_NEURONS_LAYER1,
            HyperParameter.N_NEURONS_LAYER2,
            HyperParameter.N_NEURONS_LAYER3,
            HyperParameter.SLIDING_WINDOW,
        ]

        for key, value in search_space.items():
            if key in int_hyper_parameter:
                if value is not None:
                    search_space[key] = int(value)

        if search_space.get(HyperParameter.N_NEURONS_LAYER3) is not None:
            if search_space.get(HyperParameter.N_NEURONS_LAYER1) is None or search_space.get(
                    HyperParameter.N_NEURONS_LAYER2) is None:
                raise ValueError(f"{HyperParameter.N_NEURONS_LAYER2} and {HyperParameter.N_NEURONS_LAYER1} can't be "
                                 f"None.")
            search_space[HyperParameter.ENCODER_NEURONS] = [
                search_space[HyperParameter.N_NEURONS_LAYER1],
                search_space[HyperParameter.N_NEURONS_LAYER2],
                search_space[HyperParameter.N_NEURONS_LAYER3],
            ]

        elif search_space.get(HyperParameter.N_NEURONS_LAYER2) is not None:
            if search_space.get(HyperParameter.N_NEURONS_LAYER1) is None:
                raise ValueError(f"{HyperParameter.N_NEURONS_LAYER1} can't be "
                                 f"None.")
            search_space[HyperParameter.ENCODER_NEURONS] = [
                search_space[HyperParameter.N_NEURONS_LAYER1],
                search_space[HyperParameter.N_NEURONS_LAYER2]
            ]
        elif search_space.get(HyperParameter.N_NEURONS_LAYER1) is not None:
            search_space[HyperParameter.ENCODER_NEURONS] = [
                search_space[HyperParameter.N_NEURONS_LAYER1],
            ]
            assert search_space.get(HyperParameter.N_NEURONS_LAYER2) is None
            assert search_space.get(HyperParameter.N_NEURONS_LAYER3) is None

        search_space[HyperParameter.DECODER_NEURONS] = search_space[HyperParameter.ENCODER_NEURONS][::-1]
        return search_space


class KpiFiles:
    """
    The file user of train set must be start_or_restart with train_xxx.
    The file user of test set is  replaced the train_  by _test.

    E.g.
    If the user of a train file is directory/train_<yyy>.csv,
    then the user of the test file is directory/test_<yyy>.csv

    """
    KPI_AIOPS_ALL = [
        "datasets/aiops/results/kpi_0efb375b-b902-3661-ab23-9a0bb799f4e3.csv",
        "datasets/aiops/results/kpi_301c70d8-1630-35ac-8f96-bc1b6f4359ea.csv",
        "datasets/aiops/results/kpi_e0747cad-8dc8-38a9-a9ab-855b61f5551d.csv",
        "datasets/aiops/results/kpi_a8c06b47-cc41-3738-9110-12df0ee4c721.csv",
        "datasets/aiops/results/kpi_54350a12-7a9d-3ca8-b81f-f886b9d156fd.csv",
        "datasets/aiops/results/kpi_c02607e8-7399-3dde-9d28-8a8da5e5d251.csv"
    ]

    KPI_AIOPS_01 = "datasets/aiops/results/train_0efb375b-b902-3661-ab23-9a0bb799f4e3.csv"
    KPI_AIOPS_02 = "datasets/aiops/results/train_301c70d8-1630-35ac-8f96-bc1b6f4359ea.csv"
    KPI_AIOPS_03 = "datasets/aiops/results/train_e0747cad-8dc8-38a9-a9ab-855b61f5551d.csv"
    KPI_AIOPS_AB21 = "datasets/aiops/results/train_ab216663-dcc2-3a24-b1ee-2c3e550e06c9.csv"
    # The m_dataset of Donut
    KPI_DONUT_CPU = "datasets/donut/train_cpu4.csv"
    KPI_DONUT_ETH = "datasets/donut/train_server_res_eth1out_curve_6.csv"
    KPI_DONUT_G = "datasets/donut/train_g.csv"
    KPI_ODDS_SATIMAGE = "../datasets/odds/satimage-2.npz"


class ConstActivationFunction:
    RELU = "relu"
    SELU = "selu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    SOFTPLUS = "softplus"


class ConstNNI:
    KEY_TABLE_METRICDATA = "MetricData"
    KEY_TABLE_EXPERIMENTPROFILE = "ExperimentProfile"
    KEY_TABLE_TRIALJOBEVENT = "TrialJobEvent"
    KEY_CMD_ID = "Id"
    KEY_CMD_NAME = "Name"
    KEY_CMD_Status = "Status"
    KEY_CMD_Port = "Port"
    STATUS_FINAL = 'FINAL'
    STATUS_FAILED = "FAILED"
    STATUS_ERROR = "ERROR"
    STATUS_RUNNING = "RUNNING"
    STATUS_NO_MORE_TRIAL = "NO_MORE_TRIAL"
    VALUE_DB_NAME = "nni.sqlite"
    KEY_TRAIL_JOB_ID = "trialJobId"
    KEY_TRIAL_COMMAND = "trial_cmd"
    KEY_PARAMS = "params"
    KEY_EXP_NAME = "exp_name"
    KEY_EVENT = "event"
    KEY_HYPER_PARAMETERS = "hyper_parameters"
    KEY_PARAMETERS = "parameters"
    KEY_DEFAULE = "default"
    KEY_LOG_PATH = "logPath"
    KEY_EXP_ID = "exp_id"
    NNI_DB_NAME = "nni.sqlite"
    NNI_EXPERIMENT_SAVE_DIR = "nni-experiments"


class DataSet(Enum):
    TRAIN_DATA_SET_NAME: "X"
    TRAIN_LABEL = "y"


class ConstModel:
    VAE = "VAE"
    AE = "AE"
    DEEP_SVDD = "DEEP_SVDD"


class CK:
    """
    Common Keys(ck) for dict
    """
    # The name for a machine learning model
    SEED = "seed"
    KPI_ID = "trialcmd_data_id"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    AVG = "avg"
    MODEL_NAME = "model_name"

    # The sampling rate.
    SAMPLE_RATE_STR = "str_sample_rate"
    SAMPLE_RATE = "sample_rate"
    FASTER_THAN = "faster_than"
    ELAPSED_TIME_SECONDS = "elapsed_time_seconds"
    ELAPSED_TIME_NANOSECOND = "elapsed_time_nanosecond"
