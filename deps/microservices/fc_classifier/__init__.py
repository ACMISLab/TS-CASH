# #!/usr/bin/python3
# # _*_ coding: utf-8 _*_
# # @Time    : 2025-08-30 08:49
# # @Author  : xxx@163.com
# # @File    : __init__.py
# # @Description: FC分类器自动加载模块
#
# from typing import Type
# import os
# from collections import OrderedDict
#
# from ConfigSpace.configuration_space import ConfigurationSpace
# from ConfigSpace.hyperparameters import CategoricalHyperparameter
#
# from autosklearn.askl_typing import FEAT_TYPE_TYPE
# from autosklearn.pipeline.components.base import (
#     AutoSklearnChoice,
#     AutoSklearnClassificationAlgorithm,
#     ThirdPartyComponents,
#     _addons,
#     find_components,
# )
#
# # 获取当前目录
# classifier_directory = os.path.split(__file__)[0]
#
# # 自动发现组件 - 使用当前模块名作为package
# package_name = 'fc_classifier' if __package__ is None else __package__
# try:
#     _classifiers = find_components(
#         package_name, classifier_directory, AutoSklearnClassificationAlgorithm
#     )
# except Exception as e:
#     print(f"Warning: Could not auto-discover components: {e}")
#     _classifiers = OrderedDict()
#
# # 延迟导入分类器以避免循环导入
# def _import_classifiers():
#     """延迟导入所有分类器"""
#     classifiers = {}
#
#     try:
#         from fc_ada import AdaBoostClassifierAutoSklearn
#         classifiers['AdaBoostClassifierAutoSklearn'] = AdaBoostClassifierAutoSklearn
#     except ImportError as e:
#         print(f"Warning: Could not import AdaBoostClassifierAutoSklearn: {e}")
#
#     try:
#         from fc_art import ARTClassifierAutoSklearn
#         classifiers['ARTClassifierAutoSklearn'] = ARTClassifierAutoSklearn
#     except ImportError as e:
#         print(f"Warning: Could not import ARTClassifierAutoSklearn: {e}")
#
#     try:
#         from fc_gat import GATClassifierAutoSklearn
#         classifiers['GATClassifierAutoSklearn'] = GATClassifierAutoSklearn
#     except ImportError as e:
#         print(f"Warning: Could not import GATClassifierAutoSklearn: {e}")
#
#     try:
#         from fc_gcn import GCNClassifierAutoSklearn
#         classifiers['GCNClassifierAutoSklearn'] = GCNClassifierAutoSklearn
#     except ImportError as e:
#         print(f"Warning: Could not import GCNClassifierAutoSklearn: {e}")
#
#     try:
#         from fc_mlp import MLPClassifierAutoSklearn
#         classifiers['MLPClassifierAutoSklearn'] = MLPClassifierAutoSklearn
#     except ImportError as e:
#         print(f"Warning: Could not import MLPClassifierAutoSklearn: {e}")
#
#     try:
#         from fc_rf import RandomForestClassifierAutoSklearn
#         classifiers['RandomForestClassifierAutoSklearn'] = RandomForestClassifierAutoSklearn
#     except ImportError as e:
#         print(f"Warning: Could not import RandomForestClassifierAutoSklearn: {e}")
#
#     try:
#         from fc_sage import GraphSAGEClassifierAutoSklearn
#         classifiers['GraphSAGEClassifierAutoSklearn'] = GraphSAGEClassifierAutoSklearn
#     except ImportError as e:
#         print(f"Warning: Could not import GraphSAGEClassifierAutoSklearn: {e}")
#
#     try:
#         from fc_svm import SVMClassifierAutoSklearn
#         classifiers['SVMClassifierAutoSklearn'] = SVMClassifierAutoSklearn
#     except ImportError as e:
#         print(f"Warning: Could not import SVMClassifierAutoSklearn: {e}")
#
#     try:
#         from fc_vgae import FCVGAEClassifier
#         classifiers['FCVGAEClassifier'] = FCVGAEClassifier
#     except ImportError as e:
#         print(f"Warning: Could not import FCVGAEClassifier: {e}")
#
#     return classifiers
#
# # 延迟加载的分类器字典
# _manual_classifiers = None
#
# # 第三方组件支持
# additional_components = ThirdPartyComponents(AutoSklearnClassificationAlgorithm)
# _addons["fc_classification"] = additional_components
#
#
# def add_classifier(classifier: Type[AutoSklearnClassificationAlgorithm]) -> None:
#     """添加新的分类器到组件列表中"""
#     additional_components.add_component(classifier)
#
#
# class FCClassifierChoice(AutoSklearnChoice):
#     """FC分类器选择类，模仿autosklearn的ClassifierChoice"""
#
#     def __init__(self, dataset_properties=None, **kwargs):
#         if dataset_properties is None:
#             dataset_properties = {}
#         super().__init__(dataset_properties, **kwargs)
#
#     @classmethod
#     def get_components(cls):
#         """获取所有可用的分类器组件"""
#         global _manual_classifiers
#
#         # 如果还没有加载手动分类器，则加载它们
#         if _manual_classifiers is None:
#             _manual_classifiers = _import_classifiers()
#
#         components = OrderedDict()
#         components.update(_classifiers)
#         components.update(_manual_classifiers)
#         components.update(additional_components.components)
#         return components
#
#     def get_available_components(
#         cls, dataset_properties=None, include=None, exclude=None
#     ):
#         """获取可用的分类器组件"""
#         if dataset_properties is None:
#             dataset_properties = {}
#
#         available_comp = cls.get_components()
#         components_dict = OrderedDict()
#
#         if include is not None and exclude is not None:
#             raise ValueError(
#                 "The argument include and exclude cannot be used together."
#             )
#
#         if include is not None:
#             for incl in include:
#                 if incl not in available_comp:
#                     raise ValueError(
#                         "Trying to include unknown component: " "%s" % incl
#                     )
#
#         for name in available_comp:
#             if include is not None and name not in include:
#                 continue
#             elif exclude is not None and name in exclude:
#                 continue
#
#             entry = available_comp[name]
#
#             # 避免无限循环
#             if entry == FCClassifierChoice:
#                 continue
#
#             if entry.get_properties()["handles_classification"] is False:
#                 continue
#             if (
#                 dataset_properties.get("multiclass") is True
#                 and entry.get_properties()["handles_multiclass"] is False
#             ):
#                 continue
#             if (
#                 dataset_properties.get("multilabel") is True
#                 and available_comp[name].get_properties()["handles_multilabel"] is False
#             ):
#                 continue
#             components_dict[name] = entry
#
#         return components_dict
#
#     def get_hyperparameter_search_space(
#         self,
#         feat_type: FEAT_TYPE_TYPE,
#         dataset_properties=None,
#         default=None,
#         include=None,
#         exclude=None,
#     ):
#         """获取超参数搜索空间"""
#         if dataset_properties is None:
#             dataset_properties = {}
#
#         if include is not None and exclude is not None:
#             raise ValueError(
#                 "The arguments include and " "exclude cannot be used together."
#             )
#
#         cs = ConfigurationSpace()
#
#         # 编译此问题的所有估计器对象列表
#         available_estimators = self.get_available_components(
#             dataset_properties=dataset_properties, include=include, exclude=exclude
#         )
#
#         if len(available_estimators) == 0:
#             raise ValueError("No classifiers found")
#
#         if default is None:
#             defaults = ["RandomForestClassifierAutoSklearn", "SVMClassifierAutoSklearn", "MLPClassifierAutoSklearn"] + list(
#                 available_estimators.keys()
#             )
#             for default_ in defaults:
#                 if default_ in available_estimators:
#                     if include is not None and default_ not in include:
#                         continue
#                     if exclude is not None and default_ in exclude:
#                         continue
#                     default = default_
#                     break
#
#         estimator = CategoricalHyperparameter(
#             "__choice__", list(available_estimators.keys()), default_value=default
#         )
#         cs.add_hyperparameter(estimator)
#         for estimator_name in available_estimators.keys():
#             estimator_configuration_space = available_estimators[
#                 estimator_name
#             ].get_hyperparameter_search_space(
#                 feat_type=feat_type, dataset_properties=dataset_properties
#             )
#             parent_hyperparameter = {"parent": estimator, "value": estimator_name}
#             cs.add_configuration_space(
#                 estimator_name,
#                 estimator_configuration_space,
#                 parent_hyperparameter=parent_hyperparameter,
#             )
#
#         self.configuration_space = cs
#         self.dataset_properties = dataset_properties
#         return cs
#
#     def predict_proba(self, X):
#         """预测概率"""
#         return self.choice.predict_proba(X)
#
#     def estimator_supports_iterative_fit(self):
#         """检查估计器是否支持迭代拟合"""
#         return hasattr(self.choice, "iterative_fit")
#
#     def get_max_iter(self):
#         """获取最大迭代次数"""
#         if self.estimator_supports_iterative_fit():
#             return self.choice.get_max_iter()
#         else:
#             raise NotImplementedError()
#
#     def get_current_iter(self):
#         """获取当前迭代次数"""
#         if self.estimator_supports_iterative_fit():
#             return self.choice.get_current_iter()
#         else:
#             raise NotImplementedError()
#
#     def iterative_fit(self, X, y, n_iter=1, **fit_params):
#         """迭代拟合"""
#         # 允许在choice对象上使用check_is_fitted
#         self.fitted_ = True
#         if fit_params is None:
#             fit_params = {}
#         return self.choice.iterative_fit(X, y, n_iter=n_iter, **fit_params)
#
#     def configuration_fully_fitted(self):
#         """检查配置是否完全拟合"""
#         return self.choice.configuration_fully_fitted()
#
#
# # 导出主要类和函数
# __all__ = [
#     'FCClassifierChoice',
#     'add_classifier',
#     'AdaBoostClassifierAutoSklearn',
#     'ARTClassifierAutoSklearn',
#     'GATClassifierAutoSklearn',
#     'GCNClassifierAutoSklearn',
#     'MLPClassifierAutoSklearn',
#     'RandomForestClassifierAutoSklearn',
#     'GraphSAGEClassifierAutoSklearn',
#     'SVMClassifierAutoSklearn',
#     'FCVGAEClassifier',
# ]
