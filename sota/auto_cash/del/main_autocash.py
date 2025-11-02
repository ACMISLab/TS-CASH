import time
import time
import traceback

import pygmo as pg
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, CategoricalHyperparameter, Configuration
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, log_loss, accuracy_score

from pylibs.utils.util_numpy import enable_numpy_reproduce
from sota.auto_cash.auto_cash import AutoCASHUtil
from sota.auto_cash.auto_cash_helper import get_model_args_from_dict
from sota.auto_cash.model_trainer import ModelTrainer, ModelTrainConf
from tshpo.automl_libs import get_auto_sklearn_classification_search_space, load_dataset_at_fold
from autosklearn.pipeline.components.classification import ClassifierChoice

SEED = 42
enable_numpy_reproduce(SEED)


def narrow_alg_search_space(alg_important_hpy, cs):
    """减少单个算法的超参数搜索空间, 只保留cs中重要的超参数alg_important_hpy
    Auto-CASH: 3.5.1. Potential priority
    """
    # 创建目标配置空间
    cstmp = ConfigurationSpace()
    # 根据条件筛选超参数（例如：只复制默认值为 0 的超参数）
    for hpy in cs.get_hyperparameters():
        if hpy.name in alg_important_hpy:
            cstmp.add_hyperparameter(hpy)
    return cstmp


def get_model_args_from_cs(conf):
    """将ConfigSpace的配置转为dict, 作为算法的参数

    输入:
    {'mlp:activation': 'relu', 'mlp:alpha': 0.0001, 'mlp:batch_size': 'auto', 'mlp:beta_1': 0.9, 'mlp:beta_2': 0.999, 'mlp:early_stopping': 'train', 'mlp:epsilon': 1e-08, 'mlp:hidden_layer_depth': 1, 'mlp:learning_rate_init': 0.001, 'mlp:n_iter_no_change': 32, 'mlp:num_nodes_per_layer': 32, 'mlp:shuffle': 'True', 'mlp:solver': 'adam', 'mlp:tol': 0.0001, 'random_state': 42}

    输出:
    {'activation': 'relu', 'alpha': 0.0001, 'batch_size': 'auto', 'beta_1': 0.9, 'beta_2': 0.999, 'early_stopping': 'train', 'epsilon': 1e-08, 'hidden_layer_depth': 1, 'learning_rate_init': 0.001, 'n_iter_no_change': 32, 'num_nodes_per_layer': 32, 'random_state': 42, 'shuffle': 'True', 'solver': 'adam', 'tol': 0.0001}
    """
    model_configs = conf.get_dictionary()
    _model_name = model_configs['__choice__']
    del model_configs['__choice__']
    model_configs.update({"random_state": SEED})

    values = {}
    for k, v in model_configs.items():
        values.update({k.replace(f"{_model_name}:", ""): v})
    return values


class AutoCASHProblem:
    def __init__(self, dataset, fold_index, random_state=42):
        self.dataset = dataset
        self.fold_index = fold_index
        self.random_state = random_state
        alg_name, alg_important_hpy, best_hpy_from_history = acu.init_auto_cash(dataset=dataset,
                                                                                fold_index=fold_index)
        self.auto_cash_selected_alg = alg_name
        cs = get_auto_sklearn_classification_search_space(y_train=[0, 1],
                                                          include=[alg_name])
        print(cs)
        #  只保留cs中重要的超参数
        self.cs_narrow = narrow_alg_search_space(alg_important_hpy, cs)

        self.import_hpy = alg_important_hpy

        # 初始哈超参数为最优的超参数
        # AutoCASH: Our observation is that the hyperparameter settings are similar for the same algorithm for different tasks.
        self.best_init_conf = best_hpy_from_history

        # self.cs = cs
        self.dim = len(cs.get_hyperparameters())

        self.X_train, self.y_train, self.X_test, self.y_test = load_dataset_at_fold(dataset_name=dataset, n_fold=5,
                                                                                    fold_index=self.fold_index,
                                                                                    seed=random_state)

    def fitness(self, x):
        values = self.best_init_conf
        values['__choice__'] = self.auto_cash_selected_alg
        for v, param in zip(x, self.cs_narrow):
            # param='mlp:early_stopping'
            # v= 0.2412
            _hpy = self.cs_narrow[param]
            if isinstance(_hpy, CategoricalHyperparameter):
                choice_value = _hpy.choices[int(round(v, 0))]
                values.update({param: choice_value})

        # 评估算法
        _start_time = time.time()

        model_configs = get_model_args_from_dict(values)
        mt = ModelTrainer(ModelTrainConf(
            model=self.auto_cash_selected_alg,
            model_configs=model_configs,
            dataset_name=self.dataset,
            fold_index=self.fold_index,
            seed=self.random_state
        ))
        train_res = mt.train()
        return [train_res.f1]

    def get_bounds(self):
        """这里返回的是3.5.1. Potential priority修剪之后的超参数空间"""
        lower_bounds = []
        upper_bounds = []
        for param in self.cs_narrow.get_hyperparameters():
            if isinstance(param, UniformFloatHyperparameter):
                lower_bounds.append(param.lower)
                upper_bounds.append(param.upper)
            elif isinstance(param, CategoricalHyperparameter):
                lower_bounds.append(0)
                upper_bounds.append(len(param.choices) - 1)
        return lower_bounds, upper_bounds


if __name__ == '__main__':

    SEED = 42
    N_INDIVIDUALS = 10
    DATASET = "bank-marketing"
    FOLD_INDEX = 0
    # 时间约束
    acu = AutoCASHUtil()

    # 定义问题
    problem = pg.problem(AutoCASHProblem(dataset=DATASET, fold_index=FOLD_INDEX))

    # 创建算法实例
    algo = pg.algorithm(pg.sga(gen=1))  # 设置为1代，手动控制循环

    # 创建种群
    population = pg.population(problem, size=N_INDIVIDUALS, seed=SEED)
    population_new = algo.evolve(population)
    print(population, population_new)
