import numpy as np
import pandas as pd

from pylibs.utils.util_file import FileUtils
from pylibs.uts_dataset.dataset_loader import DatasetLoader


class BenchmarkStatistics:

    def __init__(self, kpis:list):
        self._kpis=kpis

    def anomaly_rate(self):
        out=[]
        for data_set,data_id in self._kpis:
            dl=DatasetLoader(data_set=data_set,data_id=data_id)
            ar=dl.get_anomaly_rate()
            lenth=dl.get_length()
            out.append([data_set,ar,lenth])
        df=pd.DataFrame(out,columns=["dataset","anomaly_rate","length"])

        group_datasets=df.groupby(by="dataset",as_index=False)

        _static_out=[]
        for dataset_name,gd in group_datasets:
            avg_ar_percent= np.round(gd['anomaly_rate'].mean()*100,2)
            avg_lenth= int(gd['length'].mean())

            _static_out.append([dataset_name,avg_ar_percent,avg_lenth])

        out_df=pd.DataFrame(_static_out,columns=['dataset','avg_anomaly_rate','avg_length'])
        out_df=out_df.sort_values(by=['avg_anomaly_rate'])

        FileUtils.save_df_to_excel(out_df,"benchmark_statistic")



if __name__ == '__main__':
    kpis=DatasetLoader.select_top_data_ids(
        ['Daphnet', 'ECG', 'IOPS', 'MGAB', 'MITDB', 'NAB', 'NASA-MSL', 'NASA-SMAP', 'OPPORTUNITY', 'Occupancy', 'SMD',
         'SVDB', 'SensorScope', 'YAHOO'], 10)
    bs=BenchmarkStatistics(kpis)
    bs.anomaly_rate()