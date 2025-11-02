import sys
import os
import warnings
import logging
from io import StringIO

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
logging.getLogger('flaml').setLevel(logging.WARNING)


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
from flaml import AutoML
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, balanced_accuracy_score, \
    recall_score
import time

DEBUG = False

DATA_HOME = "/Volumes/18212722133/experiments/ts_cash_ms/deps/microservices/fc_classifier/classic/data"


def custom_metric_function(y_true, y_pred, eval_metric):
    """
    自定义评估函数，支持balanced_accuracy、f1_weighted、precision_weighted指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        eval_metric: 评估指标名称
    
    Returns:
        评估指标值
    """
    if eval_metric == "precision_weighted":
        return precision_score(y_true, y_pred, average="weighted")
    elif eval_metric == "f1_weighted":
        return f1_score(y_true, y_pred, average="weighted")
    elif eval_metric == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    else:
        raise RuntimeError("Unknown evaluation metric")


# if eval_metric == "precision_weighted":
#     perf = precision_score(y_test, predictions,average="weighted")
# elif eval_metric == "f1_weighted":
#     perf = f1_score(y_test, predictions,average="weighted")
# elif eval_metric == "balanced_accuracy":
#     perf = balanced_accuracy_score(y_test, predictions)
# else:
#     raise RuntimeError("Unknown evaluation metric")
def custom_metric_precision(
        X_val, y_val, estimator, labels,
        X_train, y_train, weight_val=None, weight_train=None,
        *args, ):
    from sklearn.metrics import log_loss
    import time

    start = time.time()
    y_pred = estimator.predict_proba(X_val).argmax(axis=-1)
    pred_time = (time.time() - start) / len(X_val)
    val_loss = precision_score(y_val, y_pred, average="weighted")
    y_pred = estimator.predict_proba(X_train).argmax(axis=-1)
    train_loss = precision_score(y_train, y_pred, average="weighted")
    return val_loss, {
        "val_loss": val_loss,
        "train_loss": train_loss,
        "pred_time": pred_time,
    }


def custom_metric_recall(
        X_val, y_val, estimator, labels,
        X_train, y_train, weight_val=None, weight_train=None,
        *args, ):
    from sklearn.metrics import log_loss
    import time

    start = time.time()
    y_pred = estimator.predict_proba(X_val).argmax(axis=-1)
    pred_time = (time.time() - start) / len(X_val)
    val_loss = recall_score(y_val, y_pred, average="weighted")
    y_pred = estimator.predict_proba(X_train).argmax(axis=-1)
    train_loss = recall_score(y_train, y_pred, average="weighted")
    return val_loss, {
        "val_loss": val_loss,
        "train_loss": train_loss,
        "pred_time": pred_time,
    }


def custom_metric_f1(
        X_val, y_val, estimator, labels,
        X_train, y_train, weight_val=None, weight_train=None,
        *args, ):
    from sklearn.metrics import log_loss
    import time

    start = time.time()
    y_pred = estimator.predict_proba(X_val).argmax(axis=-1)
    pred_time = (time.time() - start) / len(X_val)
    val_loss = f1_score(y_val, y_pred, average="weighted")
    y_pred = estimator.predict_proba(X_train).argmax(axis=-1)
    train_loss = f1_score(y_train, y_pred, average="weighted")
    return val_loss, {
        "val_loss": val_loss,
        "train_loss": train_loss,
        "pred_time": pred_time,
    }


@click.command()
@click.option('--dataset_name', default='d1', help='数据集名称,D1 or d2')
@click.option('--fold_index', default=0, help='折数索引')
@click.option('--seed', default=42, help='随机种子')
@click.option('--eval_metric', default="precision_weighted", help='precision_weighted,f1_weighted,custom_metric_recall')
def main(dataset_name, fold_index, seed, eval_metric):
    """FLAML优化脚本"""
    print(f"参数设置: dataset_name={dataset_name}, fold_index={fold_index}, seed={seed}")
    db = KVDBJson("flaml.json")
    _key = {"k": f"{dataset_name}_{fold_index}_{seed}_{eval_metric}"}
    if db.is_exist(_key) and DEBUG is False:
        print("实验结果已存在")
        sys.exit()
    # 加载数据
    X_train, y_train, ts_train = load_classic_fault_type_data(
        os.path.join(DATA_HOME, f"{dataset_name.upper()}/chunk_train.pkl"))
    X_test, y_test, ts_test = load_classic_fault_type_data(
        os.path.join(DATA_HOME, f"{dataset_name.upper()}/chunk_test.pkl"))
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

    # 创建FLAML AutoML预测器
    print("开始训练FLAML模型...")

    # 创建FLAML AutoML实例
    automl = AutoML()

    # 准备训练数据
    X_train_flaml = train_data.drop(columns=['target'])
    y_train_flaml = train_data['target']

    # 训练模型
    start_time = time.time()

    if eval_metric == "precision_weighted":
        flaml_metric = custom_metric_precision
    elif eval_metric == "f1_weighted":
        flaml_metric = custom_metric_f1
    elif eval_metric == "recall_weighted":
        flaml_metric = custom_metric_recall
    else:
        raise RuntimeError("Unknown evaluation metric")

    automl.fit(
        X_train=X_train_flaml,
        y_train=y_train_flaml,
        task='classification',
        metric=flaml_metric,
        max_iter=100,
        # time_budget=time_limit,
        # estimator_list=['lgbm', 'rf', 'xgboost', 'extra_tree', 'lrl1'],
        seed=seed,

    )
    training_time = time.time() - start_time
    print(f"训练完成，耗时: {training_time:.2f}秒")

    # 在测试集上进行预测
    print("在测试集上进行预测...")
    X_test_flaml = test_data.drop('target', axis=1)
    predictions = automl.predict(X_test_flaml)
    prediction_probs = automl.predict_proba(X_test_flaml)

    # 计算评估指标
    if len(prediction_probs.shape) > 1 and prediction_probs.shape[1] > 1:
        probs = prediction_probs[:, 1]  # 取正类概率
    else:
        probs = prediction_probs

    # 使用自定义评估函数计算最终性能指标
    # 对于其他标准指标，保持原有逻辑
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

    # 显示最佳模型信息
    print(f"\n=== 最佳模型信息 ===")
    best_model = automl.best_estimator
    print(f"最佳模型: {best_model}")
    print(f"最佳模型配置: {automl.best_config}")
    print(f"最佳损失: {automl.best_loss:.4f}")

    print(f"\n=== 优化完成 ===")
    print(f"使用的评估指标: {eval_metric}")
    print(f"总训练时间: {training_time:.2f}秒")
    db.add(_key, {
        'dataset_name': dataset_name,
        'fold_index': fold_index,
        'seed': seed,
        'model': best_model,
        'best_config': automl.best_config,
        'best_loss': automl.best_loss,
        'value': perf,
        'training_time': training_time,
        'eval_metric': eval_metric
    })


if __name__ == '__main__':
    main()
