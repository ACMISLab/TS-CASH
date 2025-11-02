#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2025-08-30 07:31
# @Author  : xxx@163.com
# @File    : pre_evaluate.py
# @Description: 自动评估所有fc开头的分类器
import os.path
import sys

sys.path.append("/Users/xxx/Research/dev_libs")
sys.path.append("/Volumes/sw_data/phd_paper_data/experiments/ts_cash_ms")
sys.path.append("/Users/xxx/Research/dev_libs")
sys.path.append("/Volumes/sw_data/phd_paper_data/experiments/ts_cash_ms")
sys.path.append("/remote-home4/cs_acmis_xxx/tshpo_ms")
from analyze_results import ranking_algs
from fc_gcn import GCNClassifierAutoSklearn

from pytorch_lightning import seed_everything

from fc_ada import AdaBoostClassifierAutoSklearn
from fc_mlp import MLPClassifierAutoSklearn
from fc_rf import RandomForestClassifierAutoSklearn
from fc_sage import GraphSAGEClassifierAutoSklearn
from fc_svm import SVMClassifierAutoSklearn
from fcvgae.kvdb_json import KVDBJson
from fc_vgae import FCVGAEClassifier
from fc_gat import GATClassifierAutoSklearn

# click
import click


@click.command()
@click.option("--debug", is_flag=True, default=False, help="是否开启调试模式")
def main(debug):
    # 定义所有fc开头的分类器
    fc_classifiers = {

        "GCNClassifierAutoSklearn": {
            "class": GCNClassifierAutoSklearn,
            "db_file": "fc_gcn.json"
        },
        "GATClassifierAutoSklearn": {
            "class": GATClassifierAutoSklearn,
            "db_file": "fc_gat.json"
        },
        "GraphSageClassifierAutoSklearn": {
            "class": GraphSAGEClassifierAutoSklearn,
            "db_file": "fc_sage.json"
        },
        "RandomForestClassifierAutoSklearn": {
            "class": RandomForestClassifierAutoSklearn,
            "db_file": "fc_rf.json"
        },
        "SVMClassifierAutoSklearn": {
            "class": SVMClassifierAutoSklearn,
            "db_file": "fc_svm.json"
        },
        "MLPClassifierAutoSklearn": {
            "class": MLPClassifierAutoSklearn,
            "db_file": "fc_mlp.json"
        },
        "AdaBoostClassifier": {
            "class": AdaBoostClassifierAutoSklearn,
            "db_file": "fc_ada.json"
        },
        "FCVGAEClassifier": {
            "class": FCVGAEClassifier,
            "db_file": "fc_vgae.json"
        },
        # "ARTClassifierAutoSklearn": {
        #     "class": ARTClassifierAutoSklearn,
        #     "db_file": "fc_art.json"
        # },
    }

    # 创建results目录
    os.makedirs("results", exist_ok=True)

    # 固定随机种子
    seed_everything(42)
    train_ratio = 0.1
    debug = True
    num_configs = 30  # 配置次数

    # 遍历所有fc分类器
    for classifier_name, classifier_info in fc_classifiers.items():
        print(f"正在评估 {classifier_name}...")

        classifier_class = classifier_info["class"]
        db_file = classifier_info["db_file"]

        # 获取超参数搜索空间
        cs = classifier_class.get_hyperparameter_search_space()
        configs = cs.sample_configuration(num_configs)
        db = KVDBJson(os.path.join("results", db_file))

        for dataset_name in ["D1", "D2"]:
            print(f"  数据集: {dataset_name}")
            for i, config in enumerate(configs):
                keys = config.get_dictionary()
                keys["dataset_name"] = dataset_name
                keys["train_ratio"] = train_ratio
                keys["debug"] = debug

                if db.query(keys) is not None:
                    print(f"✅✅✅✅ 配置 {i + 1}/{num_configs}: 已存在，跳过")
                    continue

                print(f"    配置 {i + 1}/{num_configs}: 开始训练...")
                # 创建分类器实例
                classifier = classifier_class(
                    debug=debug,
                    dataset_name=dataset_name,
                    train_ratio=train_ratio,
                    **config
                )

                pre, recall, f1 = classifier.fit(X=None, y=None)
                db.add(keys, {"pre": pre, "recall": recall, "f1": f1})
                print(f"    配置 {i + 1}/{num_configs}: 完成 - P:{pre:.4f}, R:{recall:.4f}, F1:{f1:.4f}")
                if debug:
                    break

    print("所有fc分类器评估完成！")
    # ranking_algs()
    print("@info 请运行analyze_results.py, 生成 ms_alg_ranking.pkl 文件")


if __name__ == '__main__':
    main()
