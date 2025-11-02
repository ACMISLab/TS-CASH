#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本用于给Excel文件添加max_prec, max_recall, max_f1列
从configs_and_metrics列中提取RunValueMS的最大值
"""

import pandas as pd
import re
import ast
from typing import List, Tuple, Dict, Any


def parse_runvalue_ms(text: str) -> List[Dict[str, float]]:
    """
    解析configs_and_metrics列中的RunValueMS数据
    
    Args:
        text: 包含RunValueMS数据的字符串
        
    Returns:
        包含precision, recall, f1值的字典列表
    """
    runvalue_list = []
    
    try:
        # 将text字符串还原为原始对象
        # 由于text包含Python对象的字符串表示，我们需要安全地解析它
        
        # 首先尝试使用eval（在受控环境中）
        # 为了安全，我们只允许特定的名称
        safe_dict = {
            '__builtins__': {},
            'Configuration': type('Configuration', (), {}),
            'RunValueMS': type('RunValueMS', (), {}),
            'True': True,
            'False': False,
            'None': None
        }
        
        # 创建一个简单的RunValueMS类来解析数据
        class RunValueMS:
            def __init__(self, default=None, roc_auc=None, f1=None, accuracy=None, 
                        recall=None, log_loss=None, precision=None, elapsed_seconds=None, 
                        error_msg=None, run_job=None, exp_conf=None, is_error=None):
                self.default = default
                self.roc_auc = roc_auc
                self.f1 = f1
                self.accuracy = accuracy
                self.recall = recall
                self.log_loss = log_loss
                self.precision = precision
                self.elapsed_seconds = elapsed_seconds
                self.error_msg = error_msg
                self.run_job = run_job
                self.exp_conf = exp_conf
                self.is_error = is_error
        
        class Configuration:
            def __init__(self, values=None):
                self.values = values or {}
        
        # 更新安全字典
        safe_dict['RunValueMS'] = RunValueMS
        safe_dict['Configuration'] = Configuration
        
        # 尝试解析字符串为Python对象
        parsed_data = eval(text, safe_dict)
        # 从解析的数据中提取RunValueMS对象
        if isinstance(parsed_data, list):
            for item in parsed_data:
                if isinstance(item, tuple) and len(item) >= 2:
                    # 第二个元素应该是RunValueMS对象
                    runvalue = item[1]
                    if hasattr(runvalue, 'precision') and hasattr(runvalue, 'recall') and hasattr(runvalue, 'f1'):
                        precision = runvalue.precision
                        recall = runvalue.recall
                        f1 = runvalue.f1
                        
                        # 过滤掉-1值（表示无效值）
                        if precision != -1 and recall != -1 and f1 != -1:
                            runvalue_list.append({
                                'precision': precision,
                                'recall': recall,
                                'f1': f1
                            })
        
    except Exception as e:
        print(f"解析configs_and_metrics时出错: {e}")
        # 回退到正则表达式方法
        return parse_runvalue_ms_regex(text)
    
    return runvalue_list

def parse_runvalue_ms_regex(text: str) -> List[Dict[str, float]]:
    """
    使用正则表达式解析RunValueMS数据的备用方法
    
    Args:
        text: 包含RunValueMS数据的字符串
        
    Returns:
        包含precision, recall, f1值的字典列表
    """
    runvalue_list = []
    
    # 使用正则表达式匹配RunValueMS模式
    pattern = r'RunValueMS\([^)]+\)'
    matches = re.findall(pattern, text)
    
    for match in matches:
        try:
            # 提取RunValueMS中的参数
            # 匹配precision, recall, f1的值
            precision_match = re.search(r'precision=([0-9.-]+)', match)
            recall_match = re.search(r'recall=([0-9.-]+)', match)
            f1_match = re.search(r'f1=([0-9.-]+)', match)
            
            if precision_match and recall_match and f1_match:
                precision = float(precision_match.group(1))
                recall = float(recall_match.group(1))
                f1 = float(f1_match.group(1))
                
                # 过滤掉-1值（表示无效值）
                if precision != -1 and recall != -1 and f1 != -1:
                    runvalue_list.append({
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    })
        except (ValueError, AttributeError) as e:
            print(f"解析RunValueMS时出错: {e}")
            continue
    
    return runvalue_list


def calculate_max_metrics(configs_and_metrics: str) -> Tuple[float, float, float]:
    """
    计算configs_and_metrics中所有RunValueMS的最大precision, recall, f1值
    
    Args:
        configs_and_metrics: 包含配置和指标的字符串
        
    Returns:
        (max_precision, max_recall, max_f1)的元组
    """
    if pd.isna(configs_and_metrics) or not configs_and_metrics:
        return -1.0, -1.0, -1.0

    runvalue_list = parse_runvalue_ms(str(configs_and_metrics))

    if not runvalue_list:
        return -1.0, -1.0, -1.0

    max_precision = max(item['precision'] for item in runvalue_list)
    max_recall = max(item['recall'] for item in runvalue_list)
    max_f1 = max(item['f1'] for item in runvalue_list)

    return max_precision, max_recall, max_f1


def process_excel_file(file_path: str) -> None:
    """
    处理Excel文件，添加max_prec, max_recall, max_f1列
    
    Args:
        file_path: Excel文件路径
    """
    print(f"正在读取文件: {file_path}")

    # 读取Excel文件
    try:
        df = pd.read_excel(file_path)
        print(f"成功读取文件，共{len(df)}行数据")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 检查是否存在configs_and_metrics列
    if 'configs_and_metrics' not in df.columns:
        print("错误: 文件中未找到configs_and_metrics列")
        print(f"可用列: {list(df.columns)}")
        return

    print("正在计算最大指标值...")

    # 计算最大指标值
    max_metrics = df['configs_and_metrics'].apply(calculate_max_metrics)

    # 分离结果到三个新列
    df['max_prec'] = [metrics[0] for metrics in max_metrics]
    df['max_recall'] = [metrics[1] for metrics in max_metrics]
    df['max_f1'] = [metrics[2] for metrics in max_metrics]

    print("新列添加完成:")
    print(f"max_prec: {df['max_prec'].describe()}")
    print(f"max_recall: {df['max_recall'].describe()}")
    print(f"max_f1: {df['max_f1'].describe()}")

    # 保存更新后的文件
    columns=["dataset", "metric", "n_high_performing_model", "hpo_opt_method", "max_prec", "max_f1",
                                 "max_recall","#instances","configs_and_metrics"]
    df.to_excel(file_path+"_update.xlsx", index=False,columns=columns)



def main():
    """
    主函数
    """
    file_path = '/Volumes/sw_data/phd_paper_data/experiments/ts_cash_ms/ts_hpo_ms/results/v2.xlsx'

    print("开始处理Excel文件...")
    process_excel_file(file_path)
    print("处理完成！")


if __name__ == '__main__':
    main()
