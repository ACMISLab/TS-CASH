#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于fc_开头文件的超参数空间自动构建器

该模块参考autosklearn的架构设计，动态发现并构建fc_分类器的超参数搜索空间。
支持以下分类器：
- fc_ada: AdaBoost分类器
- fc_gat: GAT分类器  
- fc_gcn: GCN分类器
- fc_mlp: MLP分类器
- fc_rf: RandomForest分类器
- fc_sage: GraphSAGE分类器
- fc_svm: SVM分类器
- fc_vgae: FCVGAE分类器
"""

import os
import sys
import warnings
import importlib
# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath("../../"))

from typing import Optional, Dict, Any
from collections import OrderedDict

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter


from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm

warnings.filterwarnings("ignore")


class FCClassifierChoice:
    """
    FC分类器选择器 - 参考autosklearn的ClassifierChoice设计
    """
    
    def __init__(self):
        self.configuration_space = None
        self.dataset_properties = None
        self._components = None
    
    @classmethod
    def get_components(cls):
        """
        动态发现所有fc_分类器组件
        
        Returns:
            OrderedDict: 分类器名称到类的映射
        """
        if hasattr(cls, '_cached_components'):
            return cls._cached_components
            
        components = OrderedDict()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 扫描fc_开头的Python文件
        for filename in os.listdir(current_dir):
            if filename.startswith('fc_') and filename.endswith('.py'):
                module_name = filename[:-3]  # 移除.py扩展名
                
                try:
                    # 动态导入模块
                    module = importlib.import_module(module_name)
                    
                    # 查找AutoSklearnClassificationAlgorithm的子类
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            issubclass(attr, AutoSklearnClassificationAlgorithm) and
                            attr != AutoSklearnClassificationAlgorithm):
                            components[module_name] = attr
                            break
                            
                except Exception as e:
                    warnings.warn(f"无法导入模块 {module_name}: {str(e)}")
                    continue
        
        cls._cached_components = components
        return components
    
    def get_available_components(
        self, 
        dataset_properties=None, 
        include=None, 
        exclude=None
    ):
        """
        获取可用的分类器组件
        
        Args:
            dataset_properties: 数据集属性
            include: 包含的分类器列表
            exclude: 排除的分类器列表
            
        Returns:
            OrderedDict: 可用的分类器组件
        """
        if dataset_properties is None:
            dataset_properties = {}
            
        available_comp = self.get_components()
        components_dict = OrderedDict()
        
        if include is not None and exclude is not None:
            raise ValueError(
                "The argument include and exclude cannot be used together."
            )
            
        if include is not None:
            for incl in include:
                if incl not in available_comp:
                    raise ValueError(
                        f"Trying to include unknown component: {incl}"
                    )
        
        for name in available_comp:
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue
                
            entry = available_comp[name]
            
            # 检查分类器属性
            try:
                properties = entry.get_properties(dataset_properties)
                
                if not properties.get("handles_classification", True):
                    continue
                    
                if (dataset_properties.get("multiclass") is True and 
                    not properties.get("handles_multiclass", True)):
                    continue
                    
                if (dataset_properties.get("multilabel") is True and 
                    not properties.get("handles_multilabel", False)):
                    continue
                    
                components_dict[name] = entry
                
            except Exception as e:
                warnings.warn(f"检查分类器 {name} 属性时出错: {str(e)}")
                continue
                
        return components_dict
    
    def get_hyperparameter_search_space(
        self,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties=None,
        default=None,
        include=None,
        exclude=None,
        seed=42
    ):
        """
        构建超参数搜索空间
        
        Args:
            feat_type: 特征类型
            dataset_properties: 数据集属性
            default: 默认分类器
            include: 包含的分类器列表
            exclude: 排除的分类器列表
            seed: 随机种子
            
        Returns:
            ConfigurationSpace: 超参数配置空间
        """
        if dataset_properties is None:
            dataset_properties = {}
            
        if include is not None and exclude is not None:
            raise ValueError(
                "The arguments include and exclude cannot be used together."
            )
            
        cs = ConfigurationSpace(seed=seed)
        
        # 获取可用的分类器
        available_estimators = self.get_available_components(
            dataset_properties=dataset_properties, 
            include=include, 
            exclude=exclude
        )
        
        if len(available_estimators) == 0:
            raise ValueError("No classifiers found")
            
        # 设置默认分类器
        if default is None:
            defaults = ["fc_mlp", "fc_rf", "fc_svm"] + list(available_estimators.keys())
            for default_ in defaults:
                if default_ in available_estimators:
                    if include is not None and default_ not in include:
                        continue
                    if exclude is not None and default_ in exclude:
                        continue
                    default = default_
                    break
        
        # 添加分类器选择超参数
        estimator = CategoricalHyperparameter(
            "__choice__", 
            list(available_estimators.keys())[::-1],
            default_value=default
        )
        cs.add_hyperparameter(estimator)
        
        # 为每个分类器添加其超参数空间
        for estimator_name in available_estimators.keys():
            try:
                estimator_configuration_space = available_estimators[
                    estimator_name
                ].get_hyperparameter_search_space(
                    feat_type=feat_type, 
                    dataset_properties=dataset_properties,
                    seed=seed
                )
                
                parent_hyperparameter = {
                    "parent": estimator, 
                    "value": estimator_name
                }
                
                cs.add_configuration_space(
                    estimator_name,
                    estimator_configuration_space,
                    parent_hyperparameter=parent_hyperparameter,
                )
                
            except Exception as e:
                warnings.warn(
                    f"构建分类器 {estimator_name} 的超参数空间时出错: {str(e)}"
                )
                continue
        
        self.configuration_space = cs
        self.dataset_properties = dataset_properties
        return cs
    
    def get_classifier_info(self):
        """
        获取支持的分类器信息
        
        Returns:
            Dict: 分类器信息字典
        """
        components = self.get_components()
        info = {}
        
        for name, cls in components.items():
            try:
                # 获取超参数空间以计算超参数数量
                cs = cls.get_hyperparameter_search_space()
                hyperparams_count = len(cs.get_hyperparameters())
                
                # 获取分类器属性
                properties = cls.get_properties()
                
                info[name] = {
                    'name': cls.__name__,
                    'hyperparams': hyperparams_count,
                    'description': cls.__doc__.split('\n')[0] if cls.__doc__ else '无描述',
                    'handles_multiclass': properties.get('handles_multiclass', True),
                    'handles_multilabel': properties.get('handles_multilabel', False)
                }
                
            except Exception as e:
                info[name] = {
                    'name': cls.__name__,
                    'hyperparams': 0,
                    'description': f'获取信息时出错: {str(e)}',
                    'handles_multiclass': True,
                    'handles_multilabel': False
                }
                
        return info


class FCHyperparameterSpaceBuilder:
    """
    FC超参数空间构建器 - 兼容旧接口的包装器
    """
    
    def __init__(self, seed=42):
        self.seed = seed
        self.classifier_choice = FCClassifierChoice()
    
    def build_configuration_space(
        self, 
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties=None,
        include=None,
        exclude=None
    ):
        """
        构建包含所有fc_分类器的配置空间
        
        Args:
            feat_type: 特征类型
            dataset_properties: 数据集属性
            include: 包含的分类器列表
            exclude: 排除的分类器列表
            
        Returns:
            ConfigurationSpace: 包含所有分类器及其超参数的配置空间
        """
        return self.classifier_choice.get_hyperparameter_search_space(
            feat_type=feat_type,
            dataset_properties=dataset_properties,
            include=include,
            exclude=exclude,
            seed=self.seed
        )
    
    def get_classifier_info(self):
        """
        获取支持的分类器信息
        
        Returns:
            Dict: 分类器信息字典
        """
        return self.classifier_choice.get_classifier_info()


def demo_configuration_space():
    """
    演示配置空间构建
    """
    print("=" * 60)
    print("基于fc_开头文件的超参数空间构建器演示")
    print("=" * 60)
    
    # 创建构建器
    builder = FCHyperparameterSpaceBuilder(seed=42)
    
    # 构建配置空间
    try:
        cs = builder.build_configuration_space()
        print(f"配置空间构建成功!")
        print(f"超参数数量: {len(cs.get_hyperparameters())}")
        
        # 显示分类器信息
        classifier_info = builder.get_classifier_info()
        print(f"\n支持的分类器数量: {len(classifier_info)}")
        print("\n分类器详情:")
        
        total_hyperparams = 0
        for classifier_id, info in classifier_info.items():
            print(f"  {classifier_id}: {info['name']} ({info['hyperparams']}个超参数)")
            print(f"    描述: {info['description']}")
            print(f"    多类支持: {info['handles_multiclass']}, 多标签支持: {info['handles_multilabel']}")
            total_hyperparams += info['hyperparams']
        
        print(f"\n总超参数数量: {total_hyperparams + 1} (包含分类器选择)")
        
        # 采样配置
        print("\n采样配置示例:")
        config = cs.sample_configuration()
        print(f"选择的分类器: {config['__choice__']}")
        
        # 显示该分类器的超参数
        classifier_params = {k: v for k, v in config.get_dictionary().items() 
                           if k != '__choice__'}
        
        if classifier_params:
            print(f"\n{config['__choice__']}分类器的超参数:")
            for param, value in classifier_params.items():
                print(f"  {param}: {value}")
        
    except Exception as e:
        print(f"构建配置空间时出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    # demo_configuration_space()
    # 创建构建器
    builder = FCHyperparameterSpaceBuilder(seed=42)
    # cs = builder.build_configuration_space(include=["fc_vgae"])
    cs = builder.build_configuration_space()
    print(cs.sample_configuration(10))