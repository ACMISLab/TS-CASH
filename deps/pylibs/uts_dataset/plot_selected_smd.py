from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView
from pylibs.uts_dataset.dataset_loader import DatasetLoader

# datasets = [
#     ["MITDB", "201.test.csv@2.out"],
#     ["IOPS", "KPI-ba5f3328-9f3f-3ff5-a683-84437d16d554.test.out"],
#     ["IOPS", "KPI-6d1114ae-be04-3c46-b5aa-be1a003a57cd.train.out"],
#     ["Daphnet", "S09R01E0.test.csv@6.out"],
#     ["IOPS", "KPI-ba5f3328-9f3f-3ff5-a683-84437d16d554.test.out"],
#     ["IOPS", "KPI-55f8b8b8-b659-38df-b3df-e4a5a8a54bc9.test.out"],
#     ["Daphnet", "S09R01E0.test.csv@9.out"],
#     ["MGAB", "3.test.out"],
#     ["Daphnet", "S09R01E4.test.csv@3.out"],
#     ["Daphnet", "S09R01E4.test.csv@2.out"],
#     ["SMD", "machine-3-11.test.csv@6.out"],
#     ["SMD", "machine-3-11.test.csv@6.out"],
#
#     ["IOPS", "KPI-6d1114ae-be04-3c46-b5aa-be1a003a57cd.test.out"]]

# datasets = [["SMD", "machine-3-10.test.csv@21.out"],
#             ["SMD", "machine-3-1.test.csv@20.out"],
#             ["IOPS", "KPI-55f8b8b8-b659-38df-b3df-e4a5a8a54bc9.test.out"],
#             ["IOPS", "KPI-6efa3a07-4544-34a0-b921-a155bd1a05e8.test.out"]]

datasets = [["OPPORTUNITY", "S3-ADL4.test.csv@52.out"],
            ["OPPORTUNITY", "S1-ADL3.test.csv@91.out"],
            ["OPPORTUNITY", "S3-ADL4.test.csv@52.out"],
            ["Daphnet", "S09R01E4.test.csv@3.out"],
            ["OPPORTUNITY", "S2-ADL1.test.csv@110.out"],
            ["OPPORTUNITY", "S1-ADL5.test.csv@125.out"],
            ["Daphnet", "S03R02E0.test.csv@4.out"]]

for dataset_name, dataid in datasets:
    dl = DatasetLoader(data_set=dataset_name)
    data = dl.read_data(dataset_name, dataid)
    uv = UnivariateTimeSeriesView(dl=dl, is_save_fig=True)
    uv.plot_x_label(data.iloc[:, 0].values, data.iloc[:, 1].values)
