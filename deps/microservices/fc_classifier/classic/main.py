import sys
import os

sys.path.append(os.path.abspath("../../"))

from sklearn.neural_network import MLPClassifier

from fcvgae.kvdb_json import KVDBJson
from fcvgae.libs import eval_predict_failure_type, D1, D2, load_pkl

import argparse
import warnings

warnings.filterwarnings("ignore")
import dgl.nn.pytorch
import pandas as pd
import numpy as np
import pytorch_lightning
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC


def load_data(path, label_col):
    """
    从 CSV 文件读取数据，并拆分特征与标签
    path: CSV 文件路径
    label_col: 标签所在列名
    返回 X, y（均为 DataFrame/Series 或 ndarray）
    """

    df = pd.read_csv(path)
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y


def train_random_forest(X_train, y_train, params=None):
    """
    训练 RandomForestClassifier
    params: 可选的超参数字典，例如
      {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'random_state': 42
      }
    返回训练好的 model
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'random_state': 42,
            'n_jobs': -1
        }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    """
    对测试集进行预测并输出评估报告
    """
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


def grid_search_cv(X, y, param_grid, cv=5):
    """
    使用 GridSearchCV 寻找最优超参数
    param_grid: 字典，参见 scikit-learn 文档
    cv: 交叉验证折数
    返回最优模型和最优参数
    """
    base = RandomForestClassifier(random_state=42, n_jobs=-1)
    gs = GridSearchCV(base, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    gs.fit(X, y)
    print("Best params:", gs.best_params_)
    print("Best CV score:", gs.best_score_)
    return gs.best_estimator_


def load_classic_fault_type_data(file):
    pkil_data = load_pkl(file)
    # 最后一列是failure_type_id
    feas = []
    labels = []
    ts_arr = []
    for d in pkil_data:
        ts, graph, fea, fault_ms_id, failure_type_id = d
        pooler = dgl.nn.pytorch.AvgPooling()
        fea = pooler(graph, fea)
        feas.append(fea[0].numpy())
        labels.append(failure_type_id)
        ts_arr.append(ts)

    return np.asarray(feas), np.asarray(labels), np.asarray(ts_arr)


if __name__ == "__main__":
    # ===== 1. 载入数据 =====
    # 假设你的数据在 data.csv，标签列名叫“target”
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--dataset_name', default="D1", type=str, help="dataset name, D1 or D2")
    parser.add_argument("-m", "--model_name", type=str, default="RF", help="model name, SVM or RF or ADABOOST")
    args = parser.parse_args()
    assert args.dataset_name in ["D1", "D2"]
    pytorch_lightning.seed_everything(42)
    if args.dataset_name == "D1":
        loader = D1()

    else:
        loader = D2()
    X_train, y_train, ts_train = load_classic_fault_type_data(f"data/{loader.DATA_NAME}/chunk_train.pkl")
    X_test, y_test, ts_test = load_classic_fault_type_data(f"data/{loader.DATA_NAME}/chunk_test.pkl")

    if args.model_name == "RF":
        base = RandomForestClassifier(random_state=42, n_jobs=-1)
    elif args.model_name == "SVM":
        base = SVC(random_state=42)
    elif args.model_name == "ADABOOST":
        base = AdaBoostClassifier(random_state=42)
    elif args.model_name == "MLP":
        base = MLPClassifier(hidden_layer_sizes=(128, 64))
    else:
        raise ValueError("model_name must be RF or SVM or ADABOOST")

    base.fit(X_train, y_train)
    predict = base.predict(X_test)
    predict_df = pd.DataFrame(list(zip(ts_test, predict)), columns=["timestamp", "predict"])
    prec, recall, f1 = eval_predict_failure_type(loader, predict_df, desc=vars(args))
    fdb = KVDBJson()
    fdb.add(vars(args), {
        "prec": prec,
        "rec": recall,
        "f1": f1
    })
