import pandas as pd

from pylibs.uts_dataset.dataset_loader import DatasetLoader

# todo:
names=["dataset_name","dataid"]
source_data=pd.read_csv("./has_anomaly_in_test_set.csv",names=names,header=1)

selected_datasets=DatasetLoader.select_top_data_ids()
data=pd.DataFrame(selected_datasets,columns=names)
print(data.values.tolist())
print("Selected datasets")
print(selected_datasets)
print("The number of datasets")
print(f"#dataset: {len(data['dataset_name'].unique().tolist())}")
print(data['dataset_name'].unique().tolist())
print(" ".join(data['dataset_name'].unique().tolist()))

# Daphnet ECG GHL IOPS KDD21 MGAB MITDB NAB NASA-MSL NASA-SMAP OPPORTUNITY Occupancy SMD SVDB SensorScope YAHOO



