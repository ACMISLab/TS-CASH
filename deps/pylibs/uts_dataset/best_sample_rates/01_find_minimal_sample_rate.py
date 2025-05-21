################################################################
#  Configuration
import pandas as pd

from pylibs.utils.util_servers import Server, DEVHelper

DEBUG = False
REPEATED_RUN = 5
################################################################


from pylibs.utils.util_dask import get_cluster_a100_220, get_local_cluster
from pylibs.utils.util_pandas import PDUtil
from pylibs.uts_dataset.dataset_loader import UTSDataset, find_best_sample_rate

SELECT_DATASET = [UTSDataset.DATASET_YAHOO, "NASA-MSL", "NASA-SMAP", "SMD"]


def get_tasks():
    L = []
    for dataset_name, data_id in datasets:
        for _repeat_run in range(REPEATED_RUN):
            future = client.submit(find_best_sample_rate, dataset_name, data_id, _repeat_run)
            L.append(dict(
                dataset_name=dataset_name,
                data_id=data_id,
                repeat_run=_repeat_run,
                best_sample_rate=future)
            )
        if DEBUG:
            return L
    return L


if __name__ == '__main__':
    server = Server(ip="your_server_ip")
    DEVHelper.prepare_env(server)
    client = get_cluster_a100_220()
    # client.restart()

    datasets = UTSDataset.select_considered_good_datasets(SELECT_DATASET, top_n=9999)
    print("Test datasets length:", len(datasets))
    futures = get_tasks()
    result = client.gather(futures)
    PDUtil.save_list_to_csv(result, f"best_sample_rate_ori.csv", home="./")

    res = pd.pivot_table(pd.DataFrame(result),
                         index=["dataset_name", "data_id"],
                         values=['best_sample_rate'],
                         aggfunc=['mean'])
    res = res.reset_index()
    PDUtil.save_to_csv(res, "best_sample_rate.csv", home="./", index=False)
