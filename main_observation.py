"""
在每个算法上随机抽取N个配置， 是👉每个算法上，如果每个算法抽30个，14个算法就总共14*30=420个配置

调试模式: python main.py --file debug.yaml --debug
"""
import click
import yaml
import sys
import os

PROJECT_HOME = os.environ["TSHPO_HOME"]

print(f"Project home: {PROJECT_HOME}")
sys.path.append(PROJECT_HOME)  # 将当前目录添加到 sys.path
sys.path.append(os.path.join(PROJECT_HOME, "deps"))
# fix: ModuleNotFoundError: No module named 'autosklearn'
sys.path.append(os.path.join(PROJECT_HOME, "deps/autosklearn_0.15.0"))
os.environ["OMP_NUM_THREADS"] = "1"  # 关闭OpenMP多线程
from tshpo.automl_libs import *
from tshpo.lib_func import *
from tshpo.lib_class import *
from tshpo.hpo import load_hpo_method

log = get_log()


# @memory.cache
def _run(econf: ExpConf):
    """
    运行baseline

    Parameters
    ----------
    econf :

    Returns
    -------

    """

    X_train, y_train, X_test, y_test, cs, history, watch = TSHPOFramework.prepare_resources(econf)
    if history.is_metric_cache_available() and econf.debug != True:
        log.info(f"Metric cache is available at {history.get_metrics_file_name()}")
        return history.get_metrics_file_name()
    assert len(cs) == 84
    smac = load_hpo_method(econf.hpo_opt_method, history=history, cs=cs, econf=econf)
    configs = ExpHelper.load_search_space_of_each_model(econf.n_exploration)
    log.info({
        "X_train.shape": X_train.shape,
        "y_train.shape": y_train.shape,
        "X_test.shape": X_test.shape,
        "y_test.shape": y_test.shape,
    })
    watch.start(Steps.OPTIMIZATION)
    for _conf in tqdm(configs, leave=False, position=1, desc="Optimization", ncols=88):
        info = TrialInfo(config=_conf, seed=econf.random_state, budget=0)
        run_value = train_model_smac(RunJob(
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
        TSHPOFramework.update_history(info, run_value, smac, econf, history)

    watch.stop(Steps.OPTIMIZATION)
    history.save(econf)
    if not econf.debug:

        return history.get_metrics_file_name()
    else:
        return history.get_training_time()


@click.command()
@click.option('--debug', is_flag=True, default=False, help="Is debug mode")
@click.option('--file', is_flag=False, default="c09_select_optimal_alg_v2.yaml", help="Configuration file")
def main(file, debug):
    if is_macos():
        clear_cache()

    with open(f"resources/{file}", 'r') as f:
        cf = yaml.safe_load(f)
    random_state = cf['random_state']
    enable_numpy_reproduce(random_state)
    configs = []
    datasets = cf['datasets']
    for _dataset in datasets:
        for _feature_selec_method in cf['feature_selec_method']:
            for _feature_selec_rate in cf['feature_selec_rate']:
                for _metric in cf['metrics']:
                    for _fold_index in range(cf['n_fold']):
                        for _hpo_opt_method in cf['hpo_opt_methods']:
                            for _n_explorations in cf['n_exploration']:
                                for _data_sample_method in cf['data_sample_method']:
                                    for _data_sample_rate in cf['data_sample_rate']:
                                        configs.append(ExpConf(
                                            config_file_name=file,
                                            random_state=random_state,
                                            dataset=_dataset,
                                            feature_selec_method=_feature_selec_method,
                                            feature_selec_rate=_feature_selec_rate,
                                            metric=_metric,
                                            folds=int(cf['n_fold']),
                                            fold_index=_fold_index,
                                            debug=debug,
                                            hpo_opt_method=_hpo_opt_method,
                                            n_exploration=_n_explorations,
                                            data_sample_method=_data_sample_method,
                                            data_sample_rate=_data_sample_rate,
                                        ))

    TSHPOFramework.start(_run, configs, debug)


if __name__ == '__main__':
    main()
