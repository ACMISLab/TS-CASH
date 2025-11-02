#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析fc_classifier结果文件，找到每个算法在D1和D2数据集上的最高性能并排序
"""

import json
import os
from typing import Dict, List, Tuple
import ast

from pyutils.pickle.util_pickle import save_pkl


def load_algorithm_results(results_dir: str) -> Dict[str, Dict]:
    """
    加载所有算法的结果文件
    
    Args:
        results_dir: 结果文件目录路径
        
    Returns:
        Dict: 算法名称到结果数据的映射
    """
    algorithm_results = {}

    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            algorithm_name = filename.replace('.json', '')
            filepath = os.path.join(results_dir, filename)

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    algorithm_results[algorithm_name] = data
            except Exception as e:
                print(f"加载文件 {filename} 时出错: {str(e)}")
                continue

    return algorithm_results


def extract_best_performance(algorithm_results: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    提取每个算法在D1和D2数据集上的最佳性能
    
    Args:
        algorithm_results: 算法结果数据
        
    Returns:
        Dict: 包含每个算法在各数据集上最佳性能的字典
    """
    best_performances = {}

    for algorithm_name, results in algorithm_results.items():
        best_performances[algorithm_name] = {
            'D1': {'f1': 0, 'pre': 0, 'recall': 0, 'config': None},
            'D2': {'f1': 0, 'pre': 0, 'recall': 0, 'config': None}
        }

        for config_str, metrics in results.items():
            try:
                # 解析配置字符串
                config = ast.literal_eval(config_str)
                dataset_name = config.get('dataset_name', '')

                if dataset_name in ['D1', 'D2']:
                    current_f1 = metrics.get('f1', 0)

                    # 如果当前F1分数更高，更新最佳性能
                    if current_f1 > best_performances[algorithm_name][dataset_name]['f1']:
                        best_performances[algorithm_name][dataset_name] = {
                            'f1': current_f1,
                            'pre': metrics.get('pre', 0),
                            'recall': metrics.get('recall', 0),
                            'config': config
                        }

            except Exception as e:
                print(f"解析配置 {config_str} 时出错: {str(e)}")
                continue

    return best_performances


def rank_algorithms(best_performances: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    根据F1分数对算法进行排序
    
    Args:
        best_performances: 最佳性能数据
        
    Returns:
        Dict: 排序后的结果
    """
    ranking_results = {
        'D1': {'alg_ranking': [], 'alg_name': []},
        'D2': {'alg_ranking': [], 'alg_name': []}
    }

    for dataset in ['D1', 'D2']:
        # 创建算法和F1分数的列表
        algorithm_scores = []

        for algorithm_name, performances in best_performances.items():
            f1_score = performances[dataset]['f1']
            algorithm_scores.append((algorithm_name, f1_score))

        # 按F1分数降序排序
        algorithm_scores.sort(key=lambda x: x[1], reverse=True)

        # 提取排序后的算法名称和分数
        for i, (algorithm_name, f1_score) in enumerate(algorithm_scores):
            ranking_results[dataset]['alg_name'].append(algorithm_name)
            ranking_results[dataset]['alg_ranking'].append(f1_score)

    return ranking_results


def print_detailed_results(best_performances: Dict[str, Dict], ranking_results: Dict[str, Dict]):
    """
    打印详细的结果信息
    
    Args:
        best_performances: 最佳性能数据
        ranking_results: 排序结果
    """
    print("=" * 80)
    print("算法性能分析结果")
    print("=" * 80)

    for dataset in ['D1', 'D2']:
        print(f"\n{dataset} 数据集排序结果:")
        print("-" * 50)

        for i, algorithm_name in enumerate(ranking_results[dataset]['alg_name']):
            f1_score = ranking_results[dataset]['alg_ranking'][i]
            performance = best_performances[algorithm_name][dataset]

            print(f"{i + 1:2d}. {algorithm_name:15s} - F1: {f1_score:.4f}, "
                  f"Precision: {performance['pre']:.4f}, "
                  f"Recall: {performance['recall']:.4f}")

    print("\n" + "=" * 80)


def save_results(ranking_results: Dict[str, Dict], output_file: str):
    """
    保存排序结果到JSON文件
    
    Args:
        ranking_results: 排序结果
        output_file: 输出文件路径
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ranking_results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")


def ranking_algs():
    """
    主函数
    """
    # 设置路径
    results_dir = "/Volumes/sw_data/phd_paper_data/experiments/ts_cash_ms/deps/microservices/fc_classifier/results"
    output_file = "/Volumes/sw_data/phd_paper_data/experiments/ts_cash_ms/deps/microservices/fc_classifier/algorithm_ranking.json"

    print("开始分析算法性能...")

    # 加载算法结果
    print("1. 加载算法结果文件...")
    algorithm_results = load_algorithm_results(results_dir)
    print(f"   成功加载 {len(algorithm_results)} 个算法的结果")

    # 提取最佳性能
    print("2. 提取每个算法的最佳性能...")
    best_performances = extract_best_performance(algorithm_results)

    # 算法排序
    print("3. 根据F1分数对算法进行排序...")
    ranking_results = rank_algorithms(best_performances)

    # 打印详细结果
    print_detailed_results(best_performances, ranking_results)

    # 保存结果
    save_results(ranking_results, output_file)

    print("\n分析完成!")

    # 输出最终的JSON格式结果
    print("\n最终排序结果 (JSON格式):")
    print(json.dumps(ranking_results, indent=2, ensure_ascii=False))
    save_pkl(ranking_results, "ms_alg_ranking.pkl")


if __name__ == "__main__":
    ranking_algs()