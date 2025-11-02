import pandas as pd

from pylibs.experiments.exp_config import ExpConf
from pylibs.experiments.exp_helper import start_fastuts_jobs, run_job_fastuts, main_fast_uts
from pylibs.utils.util_hash import get_str_hash
from pylibs.utils.util_pandas import PDUtil
from pylibs.utils.util_system import US


class UtilTasks:
    def __init__(self):
        self.tasks = []

    def append(self, task: ExpConf):
        self.tasks.append(task)

    def get_task_df(self):
        outs = []
        for task in self.tasks:
            metrics = task.get_dict()
            metrics.update({
                "key_params_base_code_64": task.encode_key_params_bs64(),
                "key_params_id": get_str_hash(task.encode_key_params_bs64())
            })
            outs.append(task.get_dict())
        df = pd.DataFrame(outs)
        return df

    def save_to_excel(self):
        PDUtil.save_to_excel(self.get_task_df())

    def launch(self):
        if US.is_macos():
            for task in self.tasks[:3]:
                main_fast_uts(task)
        else:
            start_fastuts_jobs(self.tasks)
