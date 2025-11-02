from dataclasses import dataclass
import pygmo as pg
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, CategoricalHyperparameter

from smac.runhistory import TrialInfo, TrialValue
from sota.auto_cash.auto_cash import AutoCASHUtil
from tshpo.automl_libs import get_auto_sklearn_classification_search_space
from tshpo.hpo.api import HPO


class AutoCASHProblem:
    def __init__(self, cs):
        self.cs = cs
        self.dim = len(cs.get_hyperparameters())

    def fitness(self, x):
        config = self.cs.sample_configuration()
        config_values = config.get_array()

        # 示例目标函数：连续变量的平方和加上分类变量的简单函数
        continuous_sum = sum(c ** 2 for c in config_values[:2])
        categorical_sum = sum(config_values[2:])

        return [continuous_sum + categorical_sum]

    def get_bounds(self):
        lower_bounds = []
        upper_bounds = []
        for param in self.cs.get_hyperparameters():
            if isinstance(param, UniformFloatHyperparameter):
                lower_bounds.append(param.lower)
                upper_bounds.append(param.upper)
            elif isinstance(param, CategoricalHyperparameter):
                lower_bounds.append(0)
                upper_bounds.append(len(param.choices) - 1)
        return lower_bounds, upper_bounds


@dataclass
class AutoCASH(HPO):
    """
    最小化任务
    """
    dataset: str = "bank-marketing"
    fold_index: int = 0
    init_cs: ConfigurationSpace = None
    # 种群（population）中的个体数量
    n_individuals: int = 10

    def __post_init__(self):
        acu = AutoCASHUtil()
        alg_name, alg_important_hpy, best_hpy_from_history = acu.init_auto_cash(dataset=self.dataset,
                                                                                fold_index=self.fold_index)
        cs = get_auto_sklearn_classification_search_space(y_train=[0, 1],
                                                          include=[alg_name])
        # 定义问题
        self.problem = pg.problem(AutoCASHProblem(cs))

        # 创建算法实例
        self.algo = pg.algorithm(pg.sga(gen=1))  # 设置为1代，手动控制循环

        # 创建种群
        self.population = pg.population(self.problem, size=self.n_individuals, seed=self.econf.random_state)

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
        # 演化种群，直到时间耗尽
        population = self.algo.evolve(self.population)
        return population
