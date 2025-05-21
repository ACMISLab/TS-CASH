import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from pylibs.ucr_models.Augment.common import augament
from ucr.ucr_dataset_loader import calc_acc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Devicd: {device}")


class TestDataset(Dataset):
    def __init__(self, ts) -> None:
        super().__init__()
        self.ts = ts

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, index):
        item = self.ts[index]
        return torch.tensor(item, dtype=torch.float32)


class DataWithLable:
    def __init__(self, slice, label) -> None:
        self.slice = slice
        self.label = label


class TrainDataset(Dataset):
    def __init__(self, pos_ts, neg_ts) -> None:
        super().__init__()
        self.ts = []
        for pos in pos_ts:
            self.ts.append(DataWithLable(pos, 1))
        for neg in neg_ts:
            self.ts.append(DataWithLable(neg, 0))

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, index):
        item = self.ts[index]
        return torch.tensor(item.slice, dtype=torch.float32), torch.tensor(item.label, dtype=torch.float32)


class Cnn3way(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.conv1 = nn.Conv1d(1, 8, 15, padding='same')
        self.bn1 = nn.BatchNorm1d(8)
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(8, 16, 15, padding='same')
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(16, 16, 15, padding='same')
        self.bn3 = nn.BatchNorm1d(16)
        self.dropout3 = nn.Dropout(0.3)
        # ==================================================

        self.conv1_2 = nn.Conv1d(1, 8, 7, padding='same')
        self.bn1_2 = nn.BatchNorm1d(8)
        self.dropout1_2 = nn.Dropout(0.3)
        self.conv2_2 = nn.Conv1d(8, 16, 7, padding='same')
        self.bn2_2 = nn.BatchNorm1d(16)
        self.dropout2_2 = nn.Dropout(0.3)

        self.conv3_2 = nn.Conv1d(16, 16, 7, padding='same')
        self.bn3_2 = nn.BatchNorm1d(16)
        self.dropout3_2 = nn.Dropout(0.3)
        # ==================================================

        self.conv1_3 = nn.Conv1d(1, 8, 3, padding='same')
        self.bn1_3 = nn.BatchNorm1d(8)
        self.dropout1_3 = nn.Dropout(0.3)
        self.conv2_3 = nn.Conv1d(8, 16, 3, padding='same')
        self.bn2_3 = nn.BatchNorm1d(16)
        self.dropout2_3 = nn.Dropout(0.3)

        self.conv3_3 = nn.Conv1d(16, 16, 3, padding='same')
        self.bn3_3 = nn.BatchNorm1d(16)
        self.dropout3_3 = nn.Dropout(0.3)

        # =================================================
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 1)
        self.fc_dropout = nn.Dropout(0.5)
        # =================================================

    def forward(self, x):
        x0 = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        # =================================
        y = self.conv1_2(x0)
        y = self.bn1_2(y)
        y = F.relu(y)
        y = self.pool(y)

        y = self.conv2_2(y)
        y = self.bn2_2(y)
        y = F.relu(y)
        y = self.pool(y)

        y = self.conv3_2(y)
        y = self.bn3_2(y)
        y = F.relu(y)
        y = self.pool(y)

        # =================================
        z = self.conv1_3(x0)
        z = self.bn1_3(z)
        z = F.relu(z)
        z = self.pool(z)

        z = self.conv2_3(z)
        z = self.bn2_3(z)
        z = F.relu(z)
        z = self.pool(z)

        z = self.conv3_3(z)
        z = self.bn3_3(z)
        z = F.relu(z)
        z = self.pool(z)

        # xy = torch.add(x, y)
        xyz = (x + y + z) / 3

        out = torch.cat((xyz, z, y, x,), dim=1)

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc_dropout(out)
        out = F.relu(out)
        out = self.fc2(out)

        return torch.sigmoid(out)


class Channel:
    def __init__(self, pos_samples, neg_samples, epoch=1, batch_size=128) -> None:
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples
        self.model = Cnn3way()

        self.optimizer = torch.optim.Adam(self.model.parameters(), 5e-5)
        self.loss_fn = torch.nn.BCELoss()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=(), gamma=0.1)
        self.model.to(device)

        # hyperparameters
        self.epoch = epoch
        self.batch_size = batch_size

    def train(self):
        print(f"==========>pos_samples.len: {len(self.pos_samples)}, neg_samples.len: {len(self.neg_samples)}")
        dataset = TrainDataset(self.pos_samples, self.neg_samples)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()

        for epoch in range(self.epoch):
            loss_epoch = 0.
            batch_epoch = 0
            for i, (slice, label) in enumerate(train_loader):
                self.optimizer.zero_grad()

                input_data = slice.to(device)
                input_data = input_data.unsqueeze(1)  # 增加通道
                out = self.model(input_data)
                out = out.squeeze(-1).float()

                label = label.to(device)
                if (len(out.shape) > 0):
                    loss = self.loss_fn(out, label)

                    loss.backward()
                    self.optimizer.step()

                batch_epoch += 1
                loss_epoch += loss.item()
            print(f"epoch:{epoch}, loss:{loss_epoch / batch_epoch:.5f}")

    def predict(self, test_list):
        self.model.eval()

        with torch.no_grad():
            dataset = TestDataset(test_list)
            test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            print("==============>test_list.len:", len(test_list))
            print("==============>dataset:", len(dataset))
            scores = []
            for i, (slice) in enumerate(test_loader):
                input_data = slice.to(device)
                input_data = input_data.unsqueeze(1)  # 增加通道
                out = self.model(input_data)
                out = out.squeeze(-1).float()
                if (len(out.shape) > 0):
                    # TypeError: iteration over a 0-d tensor
                    for item in out:
                        scores.append(item.cpu().item())
            print("==============>scores.len:", len(scores))
        return np.asarray(scores)


# /Users/sunwu/SW-OpenSourceCode/AutoML-Benchmark/deps/pylibs/pylibs/ucr_models

class Augment:
    def __init__(self, epoch=1):
        self.epoch = epoch

    def fit(self, X_train):

        start_time = time.time()
        train_neg_data = []  # 增强后的数据
        for train_pos in X_train:
            augamented_neg_list = augament(train_pos)
            for augamented_neg in augamented_neg_list:
                train_neg_data.append(augamented_neg)
        print("aug_time:", (time.time() - start_time))

        channel = Channel(X_train, train_neg_data, epoch=self.epoch)
        channel.train()
        self.channel_model_ = channel

    def accuracy_score(self, X_test, baseline_range):
        scores = self.channel_model_.predict(test_list=X_test)
        self.anomaly_scores_ = scores
        self.anomaly_pos_ = np.argmax(scores)
        return calc_acc_score(baseline_range, self.anomaly_pos_, len(X_test))


if __name__ == '__main__':
    from libs import eval_model

    clf = Augment(epoch=10)
    eval_model(clf, debug=False)
