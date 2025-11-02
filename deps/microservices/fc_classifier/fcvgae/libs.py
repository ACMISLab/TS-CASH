import argparse
import logging
import pickle
import os
import sys
import time
from pathlib import Path
import pytorch_lightning
import torch
import torch.nn as nn
import dgl
from pytorch_lightning.callbacks import EarlyStopping
from deps.microservices.fc_classifier.fcvgae.mlp import MLPClassifier
import typing
import numpy as np
import pandas as pd

from fc_config import get_ms_data_home

PROJECT_HOME = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
DATA_HOME = get_ms_data_home()

def load_pkl(file="test_graph.pkl"):
    """
    加载 pickle.dump 导出的文件

    Parameters
    ----------
    file : str
        文件路径

    Returns
    -------

    """
    print(f"\n========load {file}==========\n")
    with open(file, 'rb') as f:
        g = pickle.load(f)
    return g


def save_pkl(obj, file="test_graph.pkl"):
    """
    将python对象保存为.pkl 文件, 然后可以用load_pkl 导入

    Parameters
    ----------
    file : str
        文件路径

    Returns
    -------

    """
    with open(file, 'wb') as f:
        pickle.dump(obj, f)
    print(f"save pkl to {file}")
    assert os.path.exists(file)
    return os.path.abspath(file)


class DatasetAPI:
    def __init__(self, data_name, input_dim, feat_span, type_hash, type_dict, data_home=None, window_size=10):
        if data_home is None:
            data_home = os.path.join(DATA_HOME, data_name)
        self.DATA_HOME = data_home
        self.DATA_NAME = data_name
        self.TypeHash = type_hash
        self.TypeDict = type_dict
        self.feat_span = feat_span
        self.window_size = window_size
        self._cases = None
        self.input_dim = input_dim

    def get_input_dim(self):
        return self.input_dim

    def get_checkpoint_dir(self):
        checkpoint_home = os.path.join(PROJECT_HOME, "checkpoints", self.DATA_NAME)
        os.makedirs(checkpoint_home, exist_ok=True)
        return checkpoint_home

    def load_ms_id(self):
        """微服务名称与id的对应关系
        {'adservice-0': 0, 'adservice-1': 1, 'adservice-2': 2, 'adservice2-0': 3, 'cartservice-0': 4, 'cartservice-1': 5, 'cartservice-2': 6, 'cartservice2-0': 7, 'checkoutservice-0': 8, 'checkoutservice-1': 9, 'checkoutservice-2': 10, 'checkoutservice2-0': 11, 'currencyservice-0': 12, 'currencyservice-1': 13, 'currencyservice-2': 14, 'currencyservice2-0': 15, 'emailservice-0': 16, 'emailservice-1': 17, 'emailservice-2': 18, 'emailservice2-0': 19, 'frontend-0': 20, 'frontend-1': 21, 'frontend-2': 22, 'frontend2-0': 23, 'paymentservice-0': 24, 'paymentservice-1': 25, 'paymentservice-2': 26, 'paymentservice2-0': 27, 'productcatalogservice-0': 28, 'productcatalogservice-1': 29, 'productcatalogservice-2': 30, 'productcatalogservice2-0': 31, 'recommendationservice-0': 32, 'recommendationservice-1': 33, 'recommendationservice-2': 34, 'recommendationservice2-0': 35, 'shippingservice-0': 36, 'shippingservice-1': 37, 'shippingservice-2': 38, 'shippingservice2-0': 39, 'node-1': 40, 'node-2': 41, 'node-3': 42, 'node-4': 43, 'node-5': 44, 'node-6': 45}

        """
        return load_pkl(os.path.join(self.DATA_HOME, "graphs_info/node_hash.pkl"))

    def get_node_hash(self):
        return self.load_ms_id()

    def get_naive_model_path(self):
        stem = Path(sys.argv[0]).stem
        res_dir = f'res/{stem}'
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        naive_model_path = f'{res_dir}/naive_model.pkl'
        return naive_model_path

    def device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    def load_case(self):
        if self._cases is None:
            self._cases = pd.read_csv(os.path.join(self.DATA_HOME, "cases.csv"))
        return self._cases

    def load_train_test_cases(self, train_rate=0.6):
        """equal to DeepHunt: models.evaluation.get_eval_df)
        """
        cases = self.load_case()
        n_cases = len(cases)
        split_pos = int(n_cases * train_rate)
        fd_cases, test_cases = \
            cases.iloc[: split_pos], cases.iloc[split_pos:]
        return fd_cases, test_cases

    def load_samples(self):
        filedir = self.DATA_HOME
        # print(f"load from file: {filedir}")
        with open(os.path.join(filedir, 'samples/test_samples.pkl'), 'rb') as f:
            test_samples = pickle.load(f)

        with open(os.path.join(filedir, 'samples/train_samples.pkl'), 'rb') as f:
            train_samples = pickle.load(f)
        return train_samples, test_samples

    def load_all_samples(self):
        filedir = self.DATA_HOME
        # print(f"load from file: {filedir}")
        with open(os.path.join(filedir, 'samples/samples.pkl'), 'rb') as f:
            samples = pickle.load(f)

        return samples

    def get_fet_span(self):
        return self.feat_span

    def failuretypename2id(self, case_failure_type):
        """故障类型名称转为故障id"""
        return self.TypeHash[case_failure_type]

    def msname2id(self, ms_name):
        """微服名称转为微服务id"""
        try:
            return self.load_ms_id()[ms_name]
        except:
            return self.load_ms_id()[ms_name + "-0"]

    def get_fault_ms_id(self, start_ts, end_ts):
        casedf = self.load_case()
        fdf = casedf.query(f"timestamp>={start_ts} and timestamp <= {end_ts}")
        if fdf.shape[0] == 1:
            ms_name = fdf['cmdb_id'].item()

            # the service is not in the ms_id list, so we set to the first instance
            try:
                return self.load_ms_id()[ms_name]
            except:
                return self.load_ms_id()[ms_name + "-0"]

        elif fdf.shape[0] > 1:
            # print("pass for multitype")
            if fdf['cmdb_id'].unique().shape[0] == 1:
                return self.load_ms_id()[fdf['cmdb_id'].unique()[0]]
            else:
                return None
        else:
            return None

    def id_to_ms(self, id):
        """"""
        for k, v in self.load_ms_id().items():
            if v == id:
                return k


class D1(DatasetAPI):
    n_class = 5

    def __init__(self):
        TypeHash = {'Kubernetes Container CPU Load': 0,
                    'Kubernetes Container Memory Load': 0,
                    'Kubernetes Container Process Termination': 0,
                    'Kubernetes Container Read I/O Load': 0,
                    'Kubernetes Container Write I/O Load': 0,

                    'Kubernetes Container Network Latency': 1,
                    'Kubernetes Container Network Packet Loss': 1,
                    'Kubernetes Container Network Resource Packet Corruption': 1,
                    'Kubernetes Container Network Resource Packet Duplication': 1,

                    'Node CPU Failure': 2,
                    'Node CPU Spiking': 2,

                    'Node Disk Read I/O Consumption': 3,
                    'Node Disk Space Consumption': 3,
                    'Node Disk Write I/O Consumption': 3,

                    'Node Memory Consumption': 4}

        TypeDict = {0: 'Container Hardware', 1: 'Container Network', 2: 'Node CPU', 3: 'Node Disk', 4: 'Node Memory'}
        feat_span = {
            "metric": [(0, 52), (73, 129)],
            "trace": [(53, 61)],
            "log": [(62, 72)],
        }
        super().__init__("D1", input_dim=130, feat_span=feat_span, type_hash=TypeHash, type_dict=TypeDict)


class D2(DatasetAPI):
    n_class = 6

    def __init__(self):
        TypeHash = {'High Memory Usage': 0,
                    'High JVM CPU Load': 1,
                    'JVM Out of Memory Heap': 2,
                    'High Disk I/O Read Usage': 5,
                    'High CPU Usage': 3,
                    'Network Latency': 4,
                    'Network Packet Loss': 4,
                    'High Disk Space Usage': 5}
        TypeDict = {0: 'MEM',
                    1: 'JVM;CPU',
                    2: 'JVM;MEM',
                    3: 'CPU',
                    4: 'Network',
                    5: 'Disk'}
        feat_span = {
            "metric": [(0, 255)],
            "trace": [(256, 257)],
            "log": [(258, 265)],
        }

        super().__init__("D2", input_dim=266, feat_span=feat_span, type_hash=TypeHash, type_dict=TypeDict)


from torch.utils.data import DataLoader


# data loader start
def collate(samples):
    timestamps, graphs, feats = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return timestamps, batched_graph, torch.cat(feats, dim=0)


def create_dataloader(samples, batch_size, shuffle=True):
    dataset = list(samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    return dataloader


# data loader end


# reconstruction start
def process_top_rc_and_label(cases, filter_error, type_hash=None, before=60, after=300):
    avg_reconstruction_error, failure_label = [], []
    for _, case in cases.iterrows():
        agg_df = filter_error[(filter_error['timestamp'] >= (case['timestamp'] - before)) & (
                filter_error['timestamp'] < (case['timestamp'] + after))]
        avg_reconstruction_error.append(list(agg_df.mean()[:-1]))  # mean
        if type_hash:
            failure_label.append(type_hash[case['failure_type']])
        else:
            failure_label.append(case['failure_type'])
    return avg_reconstruction_error, failure_label


def filter_reconstruction_error(model, dataloader, filter_ms_strategy=None):
    """
    Calculate Reconstruction Error

    Parameters
    ----------
    model: the trained model
    dataloader: the test dataloader
    filter_ms_strategy: the top microservice (int) or filter by 3sigma (None)

    Returns
    -------
    reconstruction for each graph MIG

    """
    mse = nn.MSELoss(reduction='none')
    graph_rc = []

    model.eval()
    with torch.no_grad():
        for batch_index, batch_samples in enumerate(dataloader):
            batch_ts, graphs, inputs = batch_samples
            x_hat, mu, logvar = model(graphs, inputs)

            # z, h = model(graphs, inputs) ##z.shape=(128,46,32), h.shape=(128,46,130). z是经过三个神经网络提取后的特征, h 是输入的预测 (经过了一个MLP)
            loss = mse(inputs, x_hat)  # 128,46,130

            # 保证每个子图（每个样本）的节点数对应，最终 loss.shape=(batch_size, n_nodes, n_features)。
            loss = torch.stack(loss.split(graphs.batch_num_nodes().tolist()),
                               dim=0)  # 转为(batch_size, n_instance, n_fea)

            # loss.shape=(128,46,130)=(batch_size, n_instance, n_featues)
            error_of_each_ms = torch.sum(loss, dim=-1)
            mask_arr = []
            if filter_ms_strategy is None:
                # filter by 3sigma
                mu_of_each_graph = error_of_each_ms.mean(dim=-1)
                std_of_each_graph = error_of_each_ms.std(dim=-1)
                threshold_of_each_graph = mu_of_each_graph + std_of_each_graph * 3
                filter_error_of_ms_index = []

                for _threshold, _graph in zip(threshold_of_each_graph, error_of_each_ms):
                    _t = _graph > _threshold
                    _n_ms = torch.sum(_t).item()
                    if _n_ms < 3:
                        _top_value, _top_index = torch.topk(_graph, k=3)
                    else:
                        _top_value, _top_index = torch.topk(_graph, k=_n_ms)
                    _tmp_mask = torch.zeros(error_of_each_ms.shape[1])
                    _tmp_mask.scatter_(0, _top_index, 1)
                    mask_arr.append(_tmp_mask)

                mask = torch.stack(mask_arr).unsqueeze(-1)
            else:
                # filter by top N
                _, topk_indices = torch.topk(error_of_each_ms, k=filter_ms_strategy, dim=-1)

                mask = torch.zeros_like(error_of_each_ms)
                mask = mask.scatter_(1, topk_indices, 1).unsqueeze(-1)  # 128,46,1

            # remove the rc errror of the ms that are not selected
            filter_rc = loss * mask

            # get the reconstruction error for each graph on metric level
            filter_ms_error = torch.sum(filter_rc, dim=1)  # (batch_size,n_channel)=(128,130)

            tmp_df = pd.DataFrame(filter_ms_error.detach().numpy())
            tmp_df['timestamp'] = batch_ts
            graph_rc.append(tmp_df)
    return pd.concat(graph_rc, ignore_index=True)
    # return graph_rc.reset_index(
    #     drop=True)  # system_level_deviation_df.shape=(1750,131), 131中有一列是时间戳, 剩余130列是每个channel


def save_reconstruction_error(model, test_samples, cases, type_hash, rep_file=f"{time.time()}.pkl",
                              filter_ms_strategy=None,
                              before=60,
                              after=300):
    # cases: 标注好的故障样本. columns=['timestamp', 'level', 'cmdb_id', 'failure_type']
    obj = get_reconstruction_error(model, test_samples, cases, type_hash=type_hash,
                                   filter_ms_strategy=filter_ms_strategy, before=before, after=after)
    rep_file = f"res/{rep_file}"
    save_pkl(obj, rep_file)
    return rep_file


def get_reconstruction_error(model, test_samples, cases, type_hash, filter_ms_strategy=None,
                             before=60,
                             after=300):
    # cases: 标注好的故障样本. columns=['timestamp', 'level', 'cmdb_id', 'failure_type']
    dataloader = create_dataloader(test_samples, batch_size=128, shuffle=False)
    top_rc = filter_reconstruction_error(model, dataloader, filter_ms_strategy)
    failure_rc, failure_label = process_top_rc_and_label(cases, top_rc, type_hash, before, after)
    return {
        "fr": failure_rc,
        "label": failure_label
    }


# reconstruction end


# failure classification d1
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, recall_score, f1_score, precision_score


class PytorchLightingUtil:
    @staticmethod
    def merage_predict_two(predict_list: list):
        """合并predict_step步骤中返回两个样本的情况, 例如

        def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
            x, y = batch
            logits = self(x)
            predict_class = torch.argmax(logits, dim=1)
            return (predict_class,y)

        """
        res1 = []
        res2 = []
        for _p in predict_list:
            res1.append(_p[0])
            res2.append(_p[1])

        return torch.concat(res1), torch.concat(res2)


def eval_d1(rep_pkl="off_v21_pl.py.rep.pkl"):
    data = load_pkl(rep_pkl)
    # X = torch.tensor(data['fr'])  # shape: (n_samples, n_features)
    # y = torch.tensor(data['label'])  # shape: (n_samples,)
    X = data['fr']  # shape: (n_samples, n_features)
    y = data['label']  # shape: (n_samples,)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.7,
        random_state=42,
        stratify=y
    )
    # X_train, y_train=balance_with_augmentation(X_train, y_train)
    batch_size = 512
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    input_dim = len(X_train[0])
    num_classes = 6

    device = "gpu" if torch.cuda.is_available() else "cpu"
    cfg = {'hidden_dims': (128, 64), 'dropout': 0.3, 'conv_channels': 32, 'kernel_size': 3}
    hidden_dims = cfg["hidden_dims"]
    dropout = cfg["dropout"]
    model = MLPClassifier(input_dim, hidden_dims, num_classes, dropout=dropout)

    early_stop_callback = EarlyStopping(
        monitor="train_loss",
        patience=50,
        min_delta=1e-4,
        verbose=True
    )
    trainer = pytorch_lightning.Trainer(
        max_epochs=1000,
        min_epochs=200,
        fast_dev_run=False,
        callbacks=[early_stop_callback],
        accelerator=device,
        deterministic=True,
        devices=[0] if device == "gpu" else None,
        enable_checkpointing=False
    )
    trainer.fit(model=model, train_dataloaders=train_loader)
    res = trainer.predict(model=model, dataloaders=test_loader)
    y_pred, y_true = PytorchLightingUtil.merage_predict_two(res)
    acc = round(precision_score(y_true, y_pred, average="weighted"), 4)
    recall = round(recall_score(y_true, y_pred, average="weighted"), 4)
    f1 = round(f1_score(y_true, y_pred, average="weighted"), 4)

    print("==== Failrue Classification ====")
    print("precision:", acc)
    print("Recall (micro):", recall)
    print("F1 (micro):", f1)
    print("Classification Report:\n", classification_report(y_true, y_pred, digits=4))


# #  "{'dataset': 'D1', 'debug': 0, 'group': '', 'mlp_batch_size': 64, 'mlp_drop_rate': 0.1, 'mlp_hidden_dims': [128, 64], 'mlp_test_ratio': 0.4, 'vgae_batch_size': 64, 'vgae_drop_rate': 0.3, 'vgae_hidden_dim': 64, 'vgae_latent_dim': 8, 'vgae_num_layers': 2}": {
#  #    "f1": 0.964
#  #  },
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="The dataset name", default="D1")
    parser.add_argument("--debug", type=int, default=0, help="0:False, 1: True")
    parser.add_argument('--group', type=str, default="")

    parser.add_argument('--vgae_batch_size', type=int, default=64)
    parser.add_argument('--vgae_hidden_dim', type=int, default=64)
    parser.add_argument('--vgae_latent_dim', type=int, default=8)
    parser.add_argument('--vgae_drop_rate', type=float, default=0.3)
    parser.add_argument('--vgae_num_layers', type=int, default=2)
    #  --coordinates 64,32
    parser.add_argument('--mlp_hidden_dims', type=str, default="128,64")
    parser.add_argument('--mlp_drop_rate', type=float, default=0.1)
    parser.add_argument('--mlp_batch_size', type=int, default=64)
    parser.add_argument('--mlp_test_ratio', type=float, default=0.4)

    args = parser.parse_args()
    if args.mlp_hidden_dims:
        args.mlp_hidden_dims = [int(i) for i in str(args.mlp_hidden_dims).split(",")]

    return args


def get_class_weight(y_train):
    class_counts = np.bincount(y_train)
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    return class_weights


# helper

import ast
import os
import sqlite3


def _sort_dict_key(sort_d):
    """对dict的key进行排序"""
    return {key: sort_d[key] for key in sorted(sort_d.keys())}


def _round_dict_value(query_key, decimal=6):
    """标准化dict的key"""
    ret = {}
    for key, value in query_key.items():
        if isinstance(value, (float, int)):
            value = float(round(value, decimal))
        ret.update({key: value})
    return ret


class KVDBSqlite:
    """一个基于sqlite的kvdb，插入和查询都是key"""

    def __init__(self, dbfile=f"kvdb_{os.uname().nodename}.sqlite"):
        self.dbfile = dbfile
        self.conn = sqlite3.connect(self.dbfile)
        self.cursor = self.conn.cursor()
        self.cursor.execute("CREATE TABLE IF NOT EXISTS kvdb_main (key TEXT PRIMARY KEY, value TEXT)")
        self.conn.commit()

    def add(self, k, v, update=False):
        # 如果key已经存在，提示后更新value
        """

        Parameters
        ----------
        k :
        v :
        update : bool，default False
            是否更新现有的值

        Returns
        -------

        """
        # assert isinstance(k, dict)
        # assert isinstance(v, dict)
        _insert_key = self.prepare_query_key(k)
        _insert_value = self.prepare_query_key(v)
        if self.query(k):

            if update is True:
                self.cursor.execute("INSERT OR REPLACE INTO kvdb_main (key, value) VALUES (?, ?)",
                                    (_insert_key, _insert_value))
                self.conn.commit()
            else:
                logging.info("skip since key already exists, you can set update=True to overwrite current value")
        else:
            logging.debug(f"add new data {k}: {v}")
            self.cursor.execute("INSERT OR REPLACE INTO kvdb_main (key, value) VALUES (?, ?)",
                                (_insert_key, _insert_value))
            self.conn.commit()

    def query(self, k):
        assert isinstance(k, dict)
        query_key = self.prepare_query_key(k)

        self.cursor.execute("SELECT value FROM kvdb_main WHERE key=?", (query_key,))
        result = self.cursor.fetchone()
        if result:
            return ast.literal_eval(result[0])
        else:
            return None

    def prepare_query_key(self, k):
        if isinstance(k, dict):
            query_key = _sort_dict_key(k)
            query_key = _round_dict_value(query_key)
            return str(query_key)

    def query_all(self):

        self.cursor.execute("SELECT key,value FROM kvdb_main")
        result = self.cursor.fetchall()
        ret_arr = []
        for r in result:
            ret_arr.append([ast.literal_eval(r[0]), ast.literal_eval(r[1])])
        return ret_arr

    def to_dataframp(self) -> pd.DataFrame:
        """转为numpy的dataframp"""
        datas = self.query_all()
        res = []
        for key, val in datas:
            key.update(val)
            res.append(key)
        df = pd.DataFrame(res)
        return df

    def to_csv(self, csv_file_name):
        self.to_dataframp().to_csv(csv_file_name)
        return csv_file_name


# baseline start


def eval_predict_failure_type(loader: typing.Union[D1, D2], predict_df, window_size=10, predict_key="failure_type",
                              desc=""):
    """predict_key: one of cmdb_id or failure_type
    predict_df.columns=[timestamp,predict_label]
    predict_label in {1,2,...,N}
    """
    _, test_case = loader.load_train_test_cases()
    # predict_df = pd.DataFrame(np.concatenate(predict, axis=1).T, columns=['timestamp', "predict"])
    label_true_arr = []
    predict_label_arr = []
    for i, case in test_case.iterrows():
        ts = case['timestamp']
        if predict_key == "failure_type":
            label_true_id = loader.failuretypename2id(case[predict_key])
        elif predict_key == "cmdb_id":
            label_true_id = loader.msname2id(case[predict_key])
        else:
            raise Exception("predict_key must be one of cmdb_id or failure_type")

        start_ts = case['timestamp'] - (window_size * 60 / 2)  # case['timestamp']=1651542023
        end_ts = case['timestamp'] + (window_size * 60 / 2)  # end_ts-start_ts=600
        # temporal dependency
        # 对应原文中的 4.4.1 Calculate Reconstruction Error.
        series_res = predict_df[(predict_df['timestamp'] >= start_ts) & (predict_df['timestamp'] < end_ts)].set_index(
            'timestamp')
        predict_label_id = series_res['predict'].value_counts().nlargest(1).index[0]
        label_true_arr.append(label_true_id)
        predict_label_arr.append(predict_label_id)

    prec = round(precision_score(label_true_arr, predict_label_arr, average="weighted"), 4)
    recall = round(recall_score(label_true_arr, predict_label_arr, average="weighted"), 4)
    f1 = round(f1_score(label_true_arr, predict_label_arr, average="weighted"), 4)

    print(f"[{desc}] classification Report:\n", classification_report(label_true_arr, predict_label_arr))
    return prec, recall, f1


def eval_predict_failure_type_v2(loader: typing.Union[D1, D2], predict_df, window_size=10, predict_key="failure_type",
                                 desc=""):
    """predict_key: one of cmdb_id or failure_type
    predict_df.columns=[timestamp,predict_label]
    predict_label in {1,2,...,N}
    """
    _, test_case = loader.load_train_test_cases(train_rate=0.6)
    # predict_df = pd.DataFrame(np.concatenate(predict, axis=1).T, columns=['timestamp', "predict"])
    label_true_arr = []
    predict_label_arr = []
    for i, case in test_case.iterrows():
        ts = case['timestamp']
        label_true_id = loader.failuretypename2id(case[predict_key])

        start_ts = ts - (window_size * 60 / 2)  # case['timestamp']=1651542023
        end_ts = case['timestamp'] + (window_size * 60 / 2)  # end_ts-start_ts=600
        # temporal dependency
        # 对应原文中的 4.4.1 Calculate Reconstruction Error.
        series_res = predict_df[(predict_df['timestamp'] >= start_ts) & (predict_df['timestamp'] < end_ts)].set_index(
            'timestamp')
        predict_label_id = series_res['predict'].value_counts().nlargest(1).index[0]
        label_true_arr.append(label_true_id)
        predict_label_arr.append(predict_label_id)

    prec = round(precision_score(label_true_arr, predict_label_arr, average="weighted"), 4)
    recall = round(recall_score(label_true_arr, predict_label_arr, average="weighted"), 4)
    f1 = round(f1_score(label_true_arr, predict_label_arr, average="weighted"), 4)

    print(f"[{desc}] classification Report:\n", classification_report(label_true_arr, predict_label_arr))
    return prec, recall, f1


# baseline end


#
def prepare_mlp_data(rc_data, args):
    X = rc_data['fr']  # shape: (n_samples, n_features)
    y = rc_data['label']  # shape: (n_samples,)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.mlp_test_ratio,
        random_state=42,
        stratify=y
    )

    batch_size = args.mlp_batch_size
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    class_weight = get_class_weight(y_test)
    return train_loader, test_loader, class_weight


def save_rc_normal_vs_abnormal(model, test_samples, cases, args):
    """导出重构误差,用于验证我们的第一个假设： 即正常数据的重构误差较小，异常数据的重构误差较大"""
    dataloader = create_dataloader(test_samples, batch_size=128, shuffle=False)
    mse = nn.MSELoss(reduction='none')
    model.eval()
    losses = []
    times = []
    with torch.no_grad():
        for batch_index, batch_samples in enumerate(dataloader):
            batch_ts, graphs, inputs = batch_samples
            x_hat, mu, logvar = model(graphs, inputs)

            # z, h = model(graphs, inputs) ##z.shape=(128,46,32), h.shape=(128,46,130). z是经过三个神经网络提取后的特征, h 是输入的预测 (经过了一个MLP)
            loss = mse(inputs, x_hat)  # 128,46,130
            # 保证每个子图（每个样本）的节点数对应，最终 loss.shape=(batch_size, n_nodes, n_features)。
            loss = torch.stack(loss.split(graphs.batch_num_nodes().tolist()),
                               dim=0)  # 转为(batch_size, n_instance, n_fea)

            loss_of_each_graph = loss.mean(dim=-1).mean(-1)
            times.append(batch_ts)
            losses.append(loss_of_each_graph.detach().numpy())

    save_pkl({
        "time": times,
        "losses": losses,
        "cases": cases
    }, f"rc_{args.dataset}_view_d1.pkl")
