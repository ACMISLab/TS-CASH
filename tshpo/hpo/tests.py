#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/10/25 08:31
# @Author  : gsunwu@163.com
# @File    : tests.py
# @Description:
from __init__ import load_hpo_method

if __name__ == '__main__':
    from tshpo.lib_class import ExpConf, RunJob
    from tshpo.automl_libs import TSHPOFramework, train_model_smac

    econf = ExpConf(feature_selec_rate=1)
    X_train, y_train, X_test, y_test, cs, history, watch = TSHPOFramework.prepare_resources(econf)
    for _fpm in ["RS", "BO", "HB"]:
        print(f"============\n{_fpm}")
        model = load_hpo_method(_fpm, history=history, cs=cs, econf=econf)
        run_job = model.ask()
        run_value = train_model_smac(RunJob(
            X_train=X_train,
            y_train=y_train,

            X_test=X_test,
            y_test=y_test,
            metric=econf.metric,
            config=run_job.config,
            debug=econf.debug,
            cs=cs,
            seed=econf.random_state
        ))
        print("\tjob:", run_job, "\n\tvalue:", run_value)
