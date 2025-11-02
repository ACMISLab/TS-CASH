"""
在每个算法上随机抽取N个配置， 是👉每个算法上，如果每个算法抽30个，14个算法就总共14*30=420个配置

调试模式: python main.py --file debug.yaml --debug
"""
import os
import sys

# 当前绝对路径
CURRENT_HOME = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_HOME, "../"))
sys.path.append(os.path.join(CURRENT_HOME, "deps/"))
sys.path.append(os.path.join(CURRENT_HOME, "deps/microservices/fc_classifier"))
from pyutils.util_sys import is_macos

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
            exp_conf=econf,
            mode="min"
        ))
        TSHPOFrameworkMS.update_history(info, run_value, optimizer, econf, history)
        if econf.debug:
            break

    watch.stop(Steps.OPTIMIZATION)
    history.save(econf)
    if not econf.debug:

        return history.get_metrics_file_name()
    else:
        return history.get_training_time()


@click.command()
@click.option('--conf', default='hpys/1.pkl', help='参数')
@click.option('--debug', default=False, is_flag=True, help='参数')
def main(conf, debug):
    conf = load_pkl(conf)
    conf.debug = debug
    if debug:
        conf.hpo_opt_method = "RS"
    _run(conf)


if __name__ == '__main__':
    main()
