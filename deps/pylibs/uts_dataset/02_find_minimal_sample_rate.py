################################################################
#  Configuration
from pathlib import Path
import pandas as pd
from exps.search_pace import SELECTED_DATASETS
from pylibs.utils.util_servers import Servers
DEBUG = False
REPEATED_RUN = 3
from pylibs.utils.util_pandas import PDUtil
from pylibs.uts_dataset.dataset_loader import UTSDataset, find_best_sample_rate

SELECT_DATASET = UTSDataset.SELECTED_DATASETS
server=Servers.S164
server.upload_pylibs()
client = server.get_dask_client()
client.restart(wait_for_workers=False)
def get_tasks():
    L = []
    _datasets=SELECTED_DATASETS
    for dataset_name, data_id in _datasets:
        for _repeat_run in range(REPEATED_RUN):
            future_best_sample = client.submit(find_best_sample_rate, dataset_name, data_id, _repeat_run)
            L.append(dict(
                dataset_name=dataset_name,
                data_id=data_id,
                repeat_run=_repeat_run,
                best_sample_rate=future_best_sample)
            )
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
    PDUtil.save_to_csv(res, "best_sample_rate.csv", home="./", index=False)
