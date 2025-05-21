import yaml
from tshpo.automl_libs import *
from tshpo.hpo import load_hpo_method

PROJECT_HOME = os.environ["TSHPO_HOME"]
sys.path.append(os.path.dirname(__file__))

# Preprocessing the model selection information to sort out the algorithm.
if not os.path.exists(os.path.join(PROJECT_HOME, "tshpo/model_select_info.pkl")):
    HPTrimHelper.generate_outputs("c09_select_optimal_alg_v2_original_20241031_1352.csv.gz")


def _run(econf: ExpConf):
    """
    Run baseline

    Parameters
    ----------
    econf :

    Returns
    -------

    """

    try:

        enable_numpy_reproduce(econf.random_state)
        X_train, y_train, X_test, y_test, cs, history, watch = TSHPOFramework.prepare_resources(econf)
        if history.is_metric_cache_available() and econf.debug != True:
            log.info(f"Metric cache is available at {history.get_metrics_file_name()}")
            return history.get_metrics_file_name()

        # smac = TSHPOFramework.get_optimizer(econf, cs, history)
        optimizer = load_hpo_method(econf.hpo_opt_method, history=history, cs=cs, econf=econf)
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
                cs=cs,
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


import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default="c04_tshpo_acc_compare_v12_middle.yaml", help="Configuration file")
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
    with open(f"resources/{args.file}", 'r') as f:
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
                            # for _n_explorations in cf['n_exploration']:
                            for _max_iteration in cf['max_iteration']:
                                for _data_sample_method in cf['data_sample_method']:
                                    for _data_sample_rate in cf['data_sample_rate']:
                                        for _n_high_performing_model in cf['n_high_performing_model']:
                                            configs.append(ExpConf(
                                                config_file_name=args.file,
                                                random_state=random_state,
                                                dataset=_dataset,
                                                feature_selec_method=_feature_selec_method,
                                                feature_selec_rate=_feature_selec_rate,
                                                metric=_metric,
                                                folds=int(cf['n_fold']),
                                                fold_index=_fold_index,
                                                debug=args.debug,
                                                hpo_opt_method=_hpo_opt_method,
                                                # n_exploration=_n_explorations,
                                                data_sample_method=_data_sample_method,
                                                data_sample_rate=_data_sample_rate,
                                                max_iteration=_max_iteration,
                                                n_high_performing_model=_n_high_performing_model
                                            ))

    TSHPOFramework.start(_run, configs, args.debug)


if __name__ == '__main__':
    main()

"""
python main_hpo.py  --file b02_result_compare.yaml
"""
