"""
将数据转为传统机器学习的
"""
import copy
import json
import os
import sys

sys.path.append(os.path.abspath("../../"))
from microservices.fc_classifier.fcvgae.libs import save_pkl, D1, D2
import typing
import torch
from pytorch_lightning import seed_everything

window_size = 10

seed_everything(42)


def get_data(graph_node_data, fet_span):
    """合并fet_span指定的多列数据: 例如将0:52和73:129列合并"""
    # fet_span=[(0, 52), (73, 129)]
    data = []
    for index in fet_span:
        data.append(graph_node_data[:, index[0]:index[1]])
    return torch.concat(data, dim=1)


def process_label(loader: typing.Union[D1, D2], window_size=10, process_type="train"):
    """只处理有label的数据
    process_type: one of train, test
    """
    samples = loader.load_all_samples()
    if process_type == "train":
        process_cases, _ = loader.load_train_test_cases()
    else:
        _, process_cases = loader.load_train_test_cases()

    baseline = []
    for case_id, case in process_cases.iterrows():
        cnt_ts = case['timestamp']
        start_ts = case['timestamp'] - (window_size * 60 / 2)  # case['timestamp']=1651542023
        end_ts = case['timestamp'] + (window_size * 60 / 2)  # end_ts-start_ts=600

        # 开始时间,结束时间, 当前时间,故障微服务的名称,故障的分类(cpu/内存/磁盘)
        baseline.append([start_ts, end_ts, cnt_ts, case['cmdb_id'], case['failure_type']])
        # temporal dependency
        # 对应原文中的 4.4.1 Calculate Reconstruction Error.
        # series_res = loss_df[(loss_df['timestamp'] >= start_ts) & (loss_df['timestamp'] < end_ts)].set_index(
        #     'timestamp').mean()

    result_array = []
    for sample in samples:
        _ts = sample[0]
        for _baseline in baseline:
            if _ts >= _baseline[0] and _ts <= _baseline[1]:
                case_ms_name = _baseline[3]
                case_failure_type = _baseline[4]

                fault_ms_id = loader.msname2id(case_ms_name)
                failure_type_id = loader.failuretypename2id(case_failure_type)
                assert fault_ms_id is not None
                tmp_sample = list(copy.deepcopy(sample))
                tmp_sample.append(fault_ms_id)
                tmp_sample.append(failure_type_id)
                result_array.append(tmp_sample)
    return result_array


def get_num(log_span):
    sum = 0
    for item in log_span:
        sum += item[1] - item[0]
    return sum


def process_meta_json(*, loader: typing.Union[D1, D2], train_data, test_data, graph_adj_list, n_nodes):
    """
    graph_adj_list: 一个二维数组,表示邻接表
    [[1, 1, 1, 1, 1, 1, 1, 1, 10, 10, 10, 2, 0, 0, 7, 7, 7, 5, 3, 11, 11, 11, 11, 11],
    [1, 10, 6, 2, 7, 8, 5, 3, 10, 2, 0, 2, 0, 5, 7, 4, 9, 5, 3, 1, 10, 11, 0, 5]]


    """
    log_span = loader.get_fet_span()['log']
    metric_span = loader.get_fet_span()['metric']
    trace_span = loader.get_fet_span()['trace']

    data = {
        'chunk_lenth': window_size,
        'chunk_num': len(train_data) + len(test_data),
        'edges': graph_adj_list,  # 邻接表,一个(2, |E|)的数组
        'event_num': get_num(log_span),  # log 的事件列表
        'metric_num': get_num(metric_span),  # 指标的数量
        "trace_num": get_num(trace_span),  # trace 的数量
        'node_num': n_nodes  # 节点的数量
    }
    return data


def process_data(dataset="D1", fd_num=0.3):
    """dataset: D1 or D2
    fd_num: 用来训练的有监督的表情, 为了保证和DeepHunt的评估条件一样
    inspired by models.evaluation.get_eval_df
    """

    if dataset == "D1":
        loader = D1()

    else:
        loader = D2()

    train_case, _ = loader.load_train_test_cases(train_rate=0.6)
    train_data = process_label(loader, window_size=10, process_type="train")
    test_data = process_label(loader, window_size=10, process_type="test")

    # train_samples = loader.load_all_samples()
    graph = train_data[0][1]  # type: dgl.DGLHeteroGraph
    src, dst = graph.edges()  # 或者你自己的 etype
    src_list, dst_list = src.tolist(), dst.tolist()
    adj = [src_list, dst_list]  # 图的邻接表
    n_nodes = len(graph.nodes())  # 图中节点的数量

    meta = process_meta_json(loader=loader, train_data=train_data, test_data=test_data, graph_adj_list=adj,
                             n_nodes=n_nodes)
    meta["TypeHash"] = loader.TypeHash
    meta["TypeDict"] = loader.TypeDict

    if not os.path.exists(f"data/{loader.DATA_NAME}"):
        os.makedirs(f"data/{loader.DATA_NAME}")
    with open(f"data/{loader.DATA_NAME}/metadata.json", "w") as f:
        json.dump(meta, f)

    save_pkl(train_data, f"data/{loader.DATA_NAME}/chunk_train.pkl")
    save_pkl(test_data, f"data/{loader.DATA_NAME}/chunk_test.pkl")


if __name__ == '__main__':
    process_data("D1")
    process_data("D2")
