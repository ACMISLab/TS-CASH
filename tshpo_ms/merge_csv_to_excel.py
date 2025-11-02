#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并CSV文件到Excel的脚本
将指定目录中的所有CSV文件合并为一个Excel文件
"""

import os
import pandas as pd
from pathlib import Path
import glob


def merge_csv_to_excel(input_dir, output_file):
    """
    将指定目录中的所有CSV文件合并为一个Excel文件
    
    Args:
        input_dir (str): 包含CSV文件的目录路径
        output_file (str): 输出Excel文件路径
    """
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"错误：目录 {input_dir} 不存在")
        return

    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    if not csv_files:
        print(f"错误：在目录 {input_dir} 中没有找到CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件")

    # 读取并合并所有CSV文件
    all_dataframes = []

    for csv_file in csv_files:
        try:
            print(f"正在处理: {os.path.basename(csv_file)}")
            df = pd.read_csv(csv_file)

            # 添加文件名作为标识列
            df['source_file'] = os.path.basename(csv_file)

            all_dataframes.append(df)

        except Exception as e:
            print(f"读取文件 {csv_file} 时出错: {e}")
            continue

    if not all_dataframes:
        print("错误：没有成功读取任何CSV文件")
        return

    # 合并所有数据框
    print("正在合并数据...")
    merged_df = pd.concat(all_dataframes, ignore_index=True)

    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存为Excel文件
    print(f"正在保存到: {output_file}")
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 将合并的数据保存到主工作表
            merged_df.to_excel(writer, sheet_name='合并数据', index=False)

            # 可选：为每个原始文件创建单独的工作表
            for i, (csv_file, df) in enumerate(zip(csv_files, all_dataframes)):
                sheet_name = f"文件{i + 1}_{os.path.splitext(os.path.basename(csv_file))[0][:20]}"
                # Excel工作表名称限制为31个字符
                if len(sheet_name) > 31:
                    sheet_name = sheet_name[:31]
                df_without_source = df.drop('source_file', axis=1)
                df_without_source.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"成功！合并了 {len(all_dataframes)} 个文件，共 {len(merged_df)} 行数据")
        print(f"输出文件: {output_file}")

    except Exception as e:
        print(f"保存Excel文件时出错: {e}")


def main(input_directory, output_file):
    # 设置输入和输出路径

    print("开始合并CSV文件到Excel...")
    print(f"输入目录: {input_directory}")
    print(f"输出文件: {output_file}")
    print("-" * 50)

    merge_csv_to_excel(input_directory, output_file)


if __name__ == "__main__":
    input_directory = "./results/tshpo/tshpo"
    output_file = "./results/tshpo/merged_results_tshpo.xlsx"
    main(input_directory, output_file)
    input_directory = "./results/100baseline"
    output_file = "./results/100baseline.xlsx"
    main(input_directory, output_file)
