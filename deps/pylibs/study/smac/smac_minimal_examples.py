from ConfigSpace import Configuration, ConfigurationSpace
import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
iris = datasets.load_iris()


def train(config: Configuration, seed: int = 0) -> float:
    classifier = SVC(C=config["C"], random_state=seed)
    scores = cross_val_score(classifier, iris.data, iris.target, cv=5)
    return 1 - np.mean(scores)


configspace = ConfigurationSpace({"C": (0.100, 1000.0)})

scenario = Scenario(configspace, deterministic=True, n_trials=200)

smac = HyperparameterOptimizationFacade(scenario, train)
incumbent = smac.optimize()
print(incumbent)