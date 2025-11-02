import sys
import os
import warnings
import logging
from io import StringIO

from lightning import seed_everything

from classic.main import load_classic_fault_type_data

# 设置环境变量以抑制警告
os.environ['LOKY_MAX_CPU_COUNT'] = '16'  # 设置CPU核心数，避免检测警告
os.environ['DASK_DATAFRAME__QUERY_PLANNING'] = 'False'  # 抑制dask警告
os.environ['PYTHONWARNINGS'] = 'ignore'  # 抑制Python警告

# 抑制所有警告
warnings.filterwarnings('ignore')

# 设置日志级别，减少输出
logging.getLogger('joblib').setLevel(logging.ERROR)
logging.getLogger('dask').setLevel(logging.ERROR)
logging.getLogger('autogluon').setLevel(logging.WARNING)


# 创建一个静默的stderr重定向器
class SilentStderr:
    def __init__(self):
        self.buffer = StringIO()

    def __enter__(self):
        self.original_stderr = sys.stderr
        sys.stderr = self.buffer
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self.original_stderr


from pyutils.kvdb.kvdb_json import KVDBJson

"""
pip install -e D:\phd_paper_all\Research\dev_libs
"""
import click
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, \
    recall_score
import time

DEBUG = False

DATA_HOME = "/Volumes/18212722133/experiments/ts_cash_ms/deps/microservices/fc_classifier/classic/data"


@click.command()
@click.option('--dataset_name', default='D1', help='数据集名称,one of d1,d2')
@click.option('--fold_index', default=0, help='折数索引')
@click.option('--seed', default=42, help='随机种子')
@click.option('--eval_metric', default="f1_weighted", help='评估指标,precision_weighted f1_weighted recall_weighted')
def main(dataset_name, fold_index, seed, eval_metric):
    """AutoGluon优化脚本"""
    assert eval_metric in ['precision_weighted', 'f1_weighted', 'recall_weighted']
    # 使用seed模拟fold index的情况
    seed_everything(seed + fold_index)
    print(f"参数设置: dataset_name={dataset_name}, fold_index={fold_index}, seed={seed}")
    db = KVDBJson("autogluon.json")
    _key = {"k": f"{dataset_name}_{fold_index}_{seed}_{eval_metric}"}
    X_train, y_train, ts_train = load_classic_fault_type_data(
        os.path.join(DATA_HOME, f"{dataset_name.upper()}/chunk_train.pkl"))
    X_test, y_test, ts_test = load_classic_fault_type_data(
        os.path.join(DATA_HOME, f"{dataset_name.upper()}/chunk_test.pkl"))

    if db.is_exist(_key) and DEBUG is False:
        print("实验结果已存在")
        sys.exit()

    print(
        f"数据形状: X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

    # 将数据转换为DataFrame格式
    train_data = pd.DataFrame(X_train)
    train_data['target'] = y_train

    test_data = pd.DataFrame(X_test)
    test_data['target'] = y_test

    print(f"训练数据形状: {train_data.shape}")
    print(f"测试数据形状: {test_data.shape}")
    print(f"目标变量分布: {np.unique(y_train, return_counts=True)}")

    predictor = TabularPredictor(
        label='target',
        eval_metric=eval_metric,
        path=f'./autogluon_models/{_key}'  # 模型保存路径
    )

    # 设置训练参数（不使用交叉验证）
    train_params = {
        'time_limit': None,  # 不设置时间限制，使用迭代次数控制
        'num_bag_folds': 0,  # 不使用交叉验证
        'num_bag_sets': 1,
        'num_stack_levels': 0,  # 不使用交叉验证时必须为0,
        # 'fit_strategy': 'parallel'
    }

    print("开始训练AutoGluon模型...")
    start_time = time.time()

    # 训练模型（静默stderr以隐藏警告）
    with SilentStderr():
        predictor.fit(
            train_data=train_data,
            **train_params
        )

    training_time = time.time() - start_time
    print(f"训练完成，耗时: {training_time:.2f}秒")

    # 在测试集上进行预测
    print("在测试集上进行预测...")
    predictions = predictor.predict(test_data.drop('target', axis=1))
    prediction_probs = predictor.predict_proba(test_data.drop('target', axis=1))

    # 计算评估指标
    if len(prediction_probs.shape) > 1 and prediction_probs.shape[1] > 1:
        probs = prediction_probs.iloc[:, 1]  # 取正类概率
    else:
        probs = prediction_probs

    # 'balanced_accuracy','f1_weighted','precision_weighted'
    if eval_metric == "precision_weighted":
        perf = precision_score(y_test, predictions, average="weighted")
    elif eval_metric == "f1_weighted":
        perf = f1_score(y_test, predictions, average="weighted")
    elif eval_metric == "recall_weighted":
        perf = recall_score(y_test, predictions, average="weighted")
    else:
        raise RuntimeError("Unknown evaluation metric")

    print(f"\n=== 评估结果 ===")
    print(f"{eval_metric}: {perf:.4f}")

    # 显示模型排行榜
    print(f"\n=== 模型排行榜 ===")
    leaderboard = predictor.leaderboard(test_data, silent=True)
    print(leaderboard)

    # 显示最佳模型信息
    print(f"\n=== 最佳模型信息 ===")
    best_model = leaderboard.iloc[0]['model']  # 从排行榜获取最佳模型
    print(f"最佳模型: {best_model}")
    print(f"最佳模型测试分数: {leaderboard.iloc[0]['score_test']:.4f}")
    print(f"模型数量: {len(predictor.model_names())}")

    print(f"\n=== 优化完成 ===")
    print(f"使用的评估指标: {eval_metric}")
    print(f"总训练时间: {training_time:.2f}秒")
    db.add(_key, {
        'dataset_name': dataset_name,
        'fold_index': fold_index,
        'seed': seed,
        'model': best_model,
        'score_test': leaderboard.iloc[0]['score_test'],
        'value': perf,
        'training_time': training_time,
        'eval_metric': eval_metric
    })


if __name__ == '__main__':
    main()
