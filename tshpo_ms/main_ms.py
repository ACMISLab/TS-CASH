"""
在每个算法上随机抽取N个配置， 是👉每个算法上，如果每个算法抽30个，14个算法就总共14*30=420个配置

调试模式: python main.py --file debug.yaml --debug
"""
import os
import sys

from pytorch_lightning import seed_everything

# 当前绝对路径
CURRENT_HOME = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_HOME, "../"))
sys.path.append(os.path.join(CURRENT_HOME, "deps/"))
sys.path.append(os.path.join(CURRENT_HOME, "deps/microservices/fc_classifier"))
from pyutils.training.parallel_training_util import ParallelGPUTask
from pyutils.pickle.util_pickle import save_pkl
from pyutils.util_sys import is_macos, get_entry_name

if is_macos():
    PROJECT_HOME = os.environ["TSHPO_MS_HOME"]
else:
    PROJECT_HOME = "/remote-home4/cs_acmis_xxx/tshpo_ms/"

print(f"Project home: {PROJECT_HOME}")
sys.path.append(PROJECT_HOME)  # 将当前目录添加到 sys.path
sys.path.append(os.path.join(PROJECT_HOME, "deps"))
# fix: ModuleNotFoundError: No module named 'autosklearn'
sys.path.append(os.path.join(PROJECT_HOME, "deps/autosklearn_0.15.0"))
os.environ["OMP_NUM_THREADS"] = "1"  # 关闭OpenMP多线程
from tshpo.lib_func import *

hpy_config_home = "hpys"

os.makedirs(hpy_config_home, exist_ok=True)
log = get_log()

import click
from tshpo.automl_libs import *
from tshpo.hpo import load_hpo_method

sys.path.append(os.path.dirname(__file__))  # 将当前目录添加到 sys.path


def _run(econf: ExpConf):
    """
    运行baseline

    Parameters
    ----------
    econf :

    Returns
    -------

    """

    enable_numpy_reproduce(econf.random_state)
    X_train, y_train, X_test, y_test, cs, history, watch = TSHPOFrameworkMS.prepare_resources(econf)
    if history.is_metric_cache_available() and econf.debug != True:
        log.info(f"Metric cache is available at {history.get_metrics_file_name()}")
        return history.get_metrics_file_name()

    # AutoCASH.py
    # BO.py
    # HB.py
    # RS.py
    # smac = TSHPOFramework.get_optimizer(econf, cs, history)
    optimizer = load_hpo_method(econf.hpo_opt_method, history=history, cs=cs, econf=econf)
    watch.start(Steps.OPTIMIZATION)
    # Step2: Optimization
    for _ in tqdm(range(econf.max_iteration), leave=False, position=1,
                  desc=f"{econf.dataset}_{econf.metric}", ncols=88):
        info = optimizer.ask()
        run_value = train_model_smac_ms(RunJobMS(
            X_train=X_train,
            y_train=y_train,

            X_test=X_test,
            y_test=y_test,
            metric=econf.metric,
            config=info.config,
            debug=econf.debug,
            cs=cs,
            seed=econf.random_state,
            exp_conf=econf
        ))
        TSHPOFrameworkMS.update_history(info, run_value, optimizer, econf, history)

    watch.stop(Steps.OPTIMIZATION)
    history.save(econf)
    if not econf.debug:

        return history.get_metrics_file_name()
    else:
        return history.get_training_time()


def run_job(job_params, gpu_index):
    # job_params= --dataset D1 --cla_hidden_dims 1024,512
    params = str(job_params).strip()
    gpu_index = int(gpu_index)

    run_cmd = f"CUDA_VISIBLE_DEVICES={gpu_index} python entroy.py --conf {params}"
    print(run_cmd)
    os.system(run_cmd)


@click.command()
@click.option('--debug', is_flag=True, default=False, help="是否是调试模式")
@click.option('--file', is_flag=False, default="c09_select_optimal_alg_v1.yaml", help="配置文件")
def main(file, debug):
    if is_macos():
        clear_cache()
    random_state = 42
    seed_everything(42)
    debug = debug
    configs = []
    datasets = ["D1", "D2"]
    counter = 0

    hpy_jobs = []
    for _dataset in datasets:
        for _hpo_method in ["BO"]:
            for _acc in ["prec", "recall", "f1"]:
                configs.append(ExpConf(
                    config_file_name=file,
                    random_state=random_state,
                    dataset=_dataset,
                    feature_selec_method=None,
                    feature_selec_rate=None,
                    metric=_acc,
                    folds=-1,
                    fold_index=-1,
                    # max_iteration=50,  # 资源限制50次
                    max_iteration=100,  # 资源限制50次
                    debug=debug,
                    hpo_opt_method=_hpo_method,
                    n_exploration=20,
                    data_sample_method=None,
                    data_sample_rate=None,
                    n_high_performing_model=1  # 控制剪枝的数量, 1表示保留1个,2表示保留2个,
                ))
                # 保留第一个配置,用来做调试

                counter += 1
                cfn = os.path.abspath(f"{hpy_config_home}/{get_entry_name()}_{_hpo_method}_{_acc}_{counter}.pkl")
                hpy_jobs.append(cfn)
                save_pkl(configs[-1], cfn)

    ParallelGPUTask(hpy_jobs, fun=run_job, max_tasks_per_gpu=1).start()


if __name__ == '__main__':
    main()
