import pandas as pd

from pylibs.uts_dataset.dataset_loader import DatasetLoader
from pylibs.utils.util_file import FileUtil

fu = FileUtil()
files = fu.get_all_files("benchmark", ext=".out")
_output = []
for file in files:
    file = file.split("benchmark/")[-1]
    dataset, data_id = file.split("/")
    dl = DatasetLoader(dataset, data_id)
    _output.append([dataset, data_id, dl.get_anomaly_rate()])

pd.DataFrame(_output, columns=["dataset", "data_id", "anomaly_rate"]).to_csv("benchmark_dataset_anomaly_rate.csv",
                                                                             index=False,
                                                                             )
