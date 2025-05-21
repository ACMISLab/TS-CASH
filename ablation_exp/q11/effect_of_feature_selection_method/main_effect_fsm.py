"""
Q11： 只变动随机抽样算法的抽样率，查看tshpo的效果
调试模式: python main.py --file debug.yaml --debug
"""
import os
import sys
import click
import yaml

PROJECT_HOME = os.environ["TSHPO_HOME"]
print(f"Project home: {PROJECT_HOME}")
sys.path.append(PROJECT_HOME)  # 将当前目录添加到 sys.path
sys.path.append(os.path.join(PROJECT_HOME, "deps"))
# fix: ModuleNotFoundError: No module named 'autosklearn'
sys.path.append(os.path.join(PROJECT_HOME, "deps/autosklearn_0.15.0"))
from tshpo.automl_libs import *
from tshpo.lib_func import *
from tshpo.lib_class import *
from tshpo.hpo import load_hpo_method

log = get_log()


def _run_v2(econf: ExpConf):
    """
    运行baseline

    Parameters
    ----------
    econf :

    Returns
    -------

    """
    enable_numpy_reproduce(econf.random_state)
    try:
        query_key = {
            "dataset": econf.dataset,
            'metric': econf.metric,
            'fsm': econf.feature_selec_method,
            'fsr': econf.feature_selec_rate,
        }
        db = KVDBMySQL(table_name="q11_tshpo_top2_on_dataset_fsm")
        qr = db.query(query_key)
        top2_models = qr['models']
        assert len(top2_models) == 2

        # target_cs
        pruned_cs = get_auto_sklearn_classification_search_space(y_train=[0, 1],
                                                                 random_state=econf.random_state,
                                                                 include=top2_models)
        assert econf.data_sample_rate == 1

        # 加载数据集的时候，不能对数据集进行抽样了，应为是评估TS-HPO了
        X_train, y_train, X_test, y_test, history, watch = \
            TSHPOFramework.prepare_resources_with_pruned_cs_without_data_reduced(econf, pruned_cs)

        if history.is_metric_cache_available() and econf.debug != True:
            log.info(f"Metric cache is available at {history.get_metrics_file_name()}")
            return history.get_metrics_file_name()

        # smac = TSHPOFramework.get_optimizer(econf, cs, history)
        optimizer = load_hpo_method(econf.hpo_opt_method, history=history, cs=pruned_cs, econf=econf)
        watch.start(Steps.OPTIMIZATION)

        # Step2: Optimization
        log.info({
            "X_train.shape": X_train.shape,
            "y_train.shape": y_train.shape,
            "X_test.shape": X_test.shape,
            "y_test.shape": y_test.shape,
        })
        for _ in tqdm(range(econf.max_iteration), leave=False, position=1,
                      desc=f"{econf.dataset}_{X_train.shape[0]}_{econf.metric}", ncols=88):
            info = optimizer.ask()
            run_value = train_model_smac(RunJob(
                X_train=X_train,
                y_train=y_train,

                X_test=X_test,
                y_test=y_test,
                metric=econf.metric,
                config=info.config,
                debug=econf.debug,
                cs=pruned_cs,
                seed=econf.random_state,
                exp_conf=econf
            ))
            TSHPOFramework.update_history(info, run_value, optimizer, econf, history)

        watch.stop(Steps.OPTIMIZATION)
        history.save(econf)
        if not econf.debug:
            return history.get_metrics_file_name()
        else:
            return history.get_training_time()
    except:
        print("❌❌❌❌❌❌❌❌❌")
        print(traceback.format_exc())


@click.command()
@click.option('--debug', is_flag=True, default=False, help="是否是调试模式")
@click.option('--file', is_flag=False, default="Q11_effect_of_sample_method.yaml", help="配置文件")
def main(file, debug):
    if is_macos():
        clear_cache()

    with open(f"{file}", 'r') as f:
        cf = yaml.safe_load(f)
    print("config：", cf)
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
                            for _max_iteration in cf['max_iteration']:
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
                                            # n_exploration=_n_explorations,
                                            data_sample_method=_data_sample_method,
                                            data_sample_rate=_data_sample_rate,
                                            max_iteration=_max_iteration
                                        ))

    TSHPOFramework.start(_run_v2, configs, debug)


if __name__ == '__main__':
    main()
