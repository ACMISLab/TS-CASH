"""
AIOps dataset for pytorch

usage:
da = DatasetAIOps2018(kpi_id=DatasetAIOps2018.KPI_IDS.D1,
                      windows_size=20,
                      is_include_anomaly_windows=True,
                      valid_rate=0.2)
x, train_y, valid_x, valid_y, test_x, test_y = da.get_sliding_windows_three_splits()

train_dataset = AIOpsDataset(x, train_y)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
for batch_x, batch_y in train_dataloader:
    ...


"""
from torch.utils.data import Dataset, DataLoader

from pylibs._del_dir.dataset.DatasetAIOPS2018 import DatasetAIOps2018


class AIOpsDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


if __name__ == '__main__':
    da = DatasetAIOps2018(kpi_id=DatasetAIOps2018.KPI_IDS.D1,
                          windows_size=20,
                          is_include_anomaly_windows=True,
                          valid_rate=0.2)
    train_x, train_y, valid_x, valid_y, test_x, test_y = da.get_sliding_windows_three_splits()
    train_dataset = AIOpsDataset(train_x, train_y)
    print(len(train_dataset))
    print(train_dataset[1])

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    for x, y in train_dataloader:
        print(x, y)
