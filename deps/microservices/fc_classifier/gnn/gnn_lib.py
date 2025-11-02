import dataclasses
import os
import sys
from microservices.fc_classifier.fcvgae.libs import load_pkl
from pyutils.util_sys import is_macos

sys.path.append(os.path.abspath("../../"))
from collections import Counter

from dgl.nn.pytorch import AvgPooling, GATConv, GraphConv, SAGEConv
from torch.utils.data import WeightedRandomSampler

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

pl.seed_everything(42)


def load_gnn_data(data_name):
    if is_macos():
        GNN_DATA_HOME = "/Volumes/sw_data/phd_paper_data/ms_dataset/gnn"
    else:
        GNN_DATA_HOME = "/remote-home4/cs_acmis_xxx/tshpo_ms/ms_dataset/gnn"
    train_samples = load_pkl(os.path.join(GNN_DATA_HOME, f"{data_name}/chunk_train.pkl"))
    test_samples = load_pkl(os.path.join(GNN_DATA_HOME, f"{data_name}/chunk_test.pkl"))
    print("train_samples: ", len(train_samples))
    print("test_samples: ", len(test_samples))
    return train_samples, test_samples


class GraphDataUtil:
    @staticmethod
    def load_samples(file="data/D1/samples/samples_with_label.pkl"):
        """根据modal_type 自动选择对应列的特征,目标是分别训练log,metric,traces的vae模型"""

        samples = load_pkl(file)
        # samples: list  of tuple
        # each item in samples:
        # (timestamp, graph,     node-features, is_label, label_cls)
        # (时间戳,     图结构(V,E) 节点特征         是否有标签  标签的分类)
        test_rate = 0.4
        n_train = int(len(samples) * (1 - test_rate))
        train_samples, test_samples = samples[:n_train], samples[n_train:]
        return train_samples, test_samples

    @staticmethod
    def collate_cls(samples):
        timestamps, graphs, feats, is_label, label_cls = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return (timestamps, batched_graph, torch.cat(feats, dim=0), is_label, label_cls)

    @staticmethod
    def create_dataloader(samples, batch_size, shuffle=True, with_sampler=False):
        """
        train_samples, test_samples = GraphDataUtil.load_samples()
        print("train, test samples: ",len(train_samples), len(test_samples))
        data_loader=GraphDataUtil.create_dataloader(train_samples, 128)
        for batch in data_loader:
            print(batch)

        -------

        """
        label_list = [item[-1] for item in samples]
        if with_sampler:
            class_counts = Counter(label_list)  # e.g. {0:80, 1:20}
            class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
            # 给每个样本分配 weight
            sample_weights = [class_weights[lbl] for lbl in label_list]
            sample_weights = torch.tensor(sample_weights, dtype=torch.double)
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(label_list),
                replacement=True
            )
            dataloader = torch.utils.data.DataLoader(samples, batch_size=batch_size, shuffle=False,
                                                     collate_fn=GraphDataUtil.collate_cls, sampler=sampler)
        else:
            dataloader = torch.utils.data.DataLoader(samples, batch_size=batch_size, shuffle=shuffle,
                                                     collate_fn=GraphDataUtil.collate_cls)
        return dataloader


@dataclasses.dataclass
class GCNConf:
    n_node: int  # 图节点数量
    n_fea: int  # 图节点的特征数量
    n_class: int  # 分类的数量
    lr: float  # 学习率


# 1) 定义模型
class GATClassifier(pl.LightningModule):
    def __init__(self,
                 in_channels=128,  # 节点特征维度
                 hidden_channels=256,  # 隐藏层维度
                 num_classes=10,  # 图的类别数，自行修改
                 num_layers=3,
                 dropout=0.5,
                 lr=0.001,
                 num_heads=3
                 ):
        super(GATClassifier, self).__init__()
        self.lr = lr
        self.dropout = nn.Dropout(dropout)
        self.input_conv = GATConv(in_channels, hidden_channels, num_heads=3)
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, num_heads=3))
        self.classifiler = nn.Linear(hidden_channels, num_classes)
        self.pool = AvgPooling()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        graph = batch[1]
        features = batch[2]
        h = F.leaky_relu(self.input_conv(graph, features))
        h = self.dropout(h)
        for conv in self.convs:
            h = F.leaky_relu(conv(graph, h))
            h = self.dropout(h)
        # Compute mean and log variance
        # 全图池化，把每个图内的节点向量求平均
        hg = self.pool(graph, h)
        fault_type_label = batch[4]

        logits = self.classifiler(hg)

        predict = F.log_softmax(logits, dim=1)  # (128,5)
        predict = predict.mean(dim=(1, len(predict.shape) - 2))
        return (batch[0], torch.argmax(predict, dim=1))

    def training_step(self, batch, batch_index):
        # x: [总节点数, in_channels]
        # edge_index: [2, 总边数]
        # batch: [总节点数]，标记每个节点所属哪个图
        graph = batch[1]
        features = batch[2]
        h = F.leaky_relu(self.input_conv(graph, features))
        h = self.dropout(h)
        for conv in self.convs:
            h = F.leaky_relu(conv(graph, h))
            h = self.dropout(h)
        # Compute mean and log variance
        # 全图池化，把每个图内的节点向量求平均
        hg = self.pool(graph, h)

        logits = self.classifiler(hg)
        root_case_ms_label = batch[3]  # (batch_size,)
        fault_type_label = batch[4]  # (batch_size,)

        predict = F.log_softmax(logits, dim=1)  # GATConv:(128,3,3,5), GraphConv:(128,5)
        predict = predict.mean(dim=(1, len(predict.shape) - 2)).to(self.device)
        labels = torch.tensor(fault_type_label).to(self.device)  # (128,)
        loss = self.criterion(predict, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


@dataclasses.dataclass
class GCNConf:
    n_node: int  # 图节点数量
    n_fea: int  # 图节点的特征数量
    n_class: int  # 分类的数量
    lr: float  # 学习率


# 1) 定义模型
class GCNClassifier(pl.LightningModule):
    def __init__(self,
                 in_channels=128,  # 节点特征维度
                 hidden_channels=256,  # 隐藏层维度
                 num_classes=10,  # 图的类别数，自行修改
                 num_layers=3,
                 dropout=0.5,
                 lr=0.001,
                 ):
        super(GCNClassifier, self).__init__()
        self.lr = lr
        self.dropout = nn.Dropout(dropout)
        self.input_conv = GraphConv(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_channels, hidden_channels))
        self.classifiler = nn.Linear(hidden_channels, num_classes)
        self.pool = AvgPooling()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        graph = batch[1]
        features = batch[2]
        h = F.leaky_relu(self.input_conv(graph, features))
        h = self.dropout(h)
        for conv in self.convs:
            h = F.leaky_relu(conv(graph, h))
            h = self.dropout(h)
        # Compute mean and log variance
        # 全图池化，把每个图内的节点向量求平均
        hg = self.pool(graph, h)
        fault_type_label = batch[4]

        logits = self.classifiler(hg)

        predict = F.log_softmax(logits, dim=1)  # (128,5)
        return (batch[0], torch.argmax(predict, dim=1))

    def training_step(self, batch, batch_index):
        # x: [总节点数, in_channels]
        # edge_index: [2, 总边数]
        # batch: [总节点数]，标记每个节点所属哪个图
        graph = batch[1]
        features = batch[2]
        h = F.leaky_relu(self.input_conv(graph, features))
        h = self.dropout(h)
        for conv in self.convs:
            h = F.leaky_relu(conv(graph, h))
            h = self.dropout(h)
        # Compute mean and log variance
        # 全图池化，把每个图内的节点向量求平均
        hg = self.pool(graph, h)

        logits = self.classifiler(hg)
        fault_type_label = batch[4]  # (batch_size,)

        predict = F.log_softmax(logits, dim=1).to(self.device)  # (128,5)
        labels = torch.tensor(fault_type_label).to(self.device)  # (128,)
        loss = self.criterion(predict, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# 1) 定义模型
class GraphSageClassifier(pl.LightningModule):
    def __init__(self,
                 in_channels=128,  # 节点特征维度
                 hidden_channels=256,  # 隐藏层维度
                 num_classes=10,  # 图的类别数，自行修改
                 num_layers=3,
                 dropout=0.5,
                 lr=0.001,
                 num_heads=3
                 ):
        super(GraphSageClassifier, self).__init__()
        self.lr = lr
        self.dropout = nn.Dropout(dropout)
        self.input_conv = SAGEConv(in_channels, hidden_channels, aggregator_type="mean")
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggregator_type="mean"))
        self.classifiler = nn.Linear(hidden_channels, num_classes)
        self.pool = AvgPooling()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        graph = batch[1]
        features = batch[2]
        h = F.leaky_relu(self.input_conv(graph, features))
        h = self.dropout(h)
        for conv in self.convs:
            h = F.leaky_relu(conv(graph, h))
            h = self.dropout(h)
        # Compute mean and log variance
        # 全图池化，把每个图内的节点向量求平均
        hg = self.pool(graph, h)
        fault_type_label = batch[4]

        logits = self.classifiler(hg)

        predict = F.log_softmax(logits, dim=1)  # (128,5)
        return (batch[0], torch.argmax(predict, dim=1))

    def training_step(self, batch, batch_index):
        # x: [总节点数, in_channels]
        # edge_index: [2, 总边数]
        # batch: [总节点数]，标记每个节点所属哪个图
        graph = batch[1]
        features = batch[2]
        h = F.leaky_relu(self.input_conv(graph, features))
        h = self.dropout(h)
        for conv in self.convs:
            h = F.leaky_relu(conv(graph, h))
            h = self.dropout(h)
        # Compute mean and log variance
        # 全图池化，把每个图内的节点向量求平均
        hg = self.pool(graph, h)

        logits = self.classifiler(hg)
        fault_type_label = batch[4]  # (batch_size,)

        predict = F.log_softmax(logits, dim=1).to(self.device)  # SAGEConv:(128,5)
        labels = torch.tensor(fault_type_label).to(self.device)  # (128,)
        loss = self.criterion(predict, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def collate_cls_gcn(samples):
    timestamps, graphs, feats, fault_ms_name, failure_type = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return (timestamps, batched_graph, torch.cat(feats, dim=0), fault_ms_name, failure_type)


def create_dataloader(samples, batch_size, shuffle=True, with_sampler=False):
    """
    train_samples, test_samples = GraphDataUtil.load_samples()
    print("train, test samples: ",len(train_samples), len(test_samples))
    data_loader=GraphDataUtil.create_dataloader(train_samples, 128)
    for batch in data_loader:
        print(batch)

    -------

    """
    label_list = [item[-1] for item in samples]
    if with_sampler:
        class_counts = Counter(label_list)  # e.g. {0:80, 1:20}
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        # 给每个样本分配 weight
        sample_weights = [class_weights[lbl] for lbl in label_list]
        sample_weights = torch.tensor(sample_weights, dtype=torch.double)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(label_list),
            replacement=True
        )
        dataloader = torch.utils.data.DataLoader(samples, batch_size=batch_size, shuffle=False,
                                                 collate_fn=GraphDataUtil.collate_cls, sampler=sampler)
    else:
        dataloader = torch.utils.data.DataLoader(samples, batch_size=batch_size, shuffle=shuffle,
                                                 collate_fn=GraphDataUtil.collate_cls)
    return dataloader
