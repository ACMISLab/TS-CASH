import os

from pyutils.pickle.util_pickle import load_pkl
from pyutils.training.parallel_training_util import ParallelGPUTask

jobs = load_pkl("tasks.pkl")
def fun(job_params, gpu_index):
    # job_params= --dataset D1 --cla_hidden_dims 1024,512
    params = str(job_params).strip()
    gpu_index = int(gpu_index)

    run_cmd = f"CUDA_VISIBLE_DEVICES={gpu_index} {params}"
    print(run_cmd)
    os.system(run_cmd)


ParallelGPUTask(tasks=jobs, fun=fun, gpu_count=2, max_tasks_per_gpu=1).start()