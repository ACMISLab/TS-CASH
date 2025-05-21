################################################################
#  Configuration
from pathlib import Path

import pandas as pd

from exps.e_cash_libs import ExpRun, OMT
from pylibs.utils.util_servers import Servers

DEBUG = False
REPEATED_RUN = 5
from pylibs.utils.util_pandas import PDUtil
from exps.e_cash_libs import  SELECTED_DATASETS
server = Servers.S100_9
server.upload_pylibs()
client = server.get_dask_client()
client.restart(wait_for_workers=False)
def get_tasks():
    L = []
    for _opt_metric in [OMT.BEST_F1_SCORE, OMT.VUS_ROC, OMT.VUS_PR]:
        for dataset_name, data_id in SELECTED_DATASETS:
            for _seed in range(REPEATED_RUN):
                future_best_sample = client.submit(find_best_sample_rate,
                                                   dataset_name=dataset_name,
                                                   data_id=data_id,
                                                   repeat_run=_seed,
                                                   window_size=ExpRun.window_size,
                                                   metric=_opt_metric,
                                                   stop_threshold=0.001)
                L.append(dict(
                    dataset_name=dataset_name,
                    data_id=data_id,
                    repeat_run=_seed,
                    window_size=ExpRun.window_size,
                    metric=_opt_metric,
                    stop_threshold=0.001,
                    best_sample_rate=future_best_sample
                ))
            if DEBUG:
                return L
    return L


if __name__ == '__main__':
    futures = get_tasks()
    result = client.gather(futures)
    PDUtil.save_list_to_csv(result, f"top_10_dataset_origin.csv", home=Path("./"))
    df = pd.DataFrame(result)
    df = df[df['best_sample_rate'] >= 0]
    res = df.groupby(by=['dataset_name', 'data_id']).mean()
    res = res.reset_index()
    PDUtil.save_to_csv(res, "best_sample_rate.csv", home="./")
