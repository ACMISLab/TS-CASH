# ID2024102308311232108888
import click
import pandas as pd
import yaml

from tshpo.automl_libs import *

from pylibs.utils.util_numpy import enable_numpy_reproduce
from tshpo.automl_libs import ExpHelper
from tshpo.lib_class import ExpConf


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

    enable_numpy_reproduce(econf.random_state)
    eh = ExpHelper(econf)
    cs = eh.load_search_space()
    _t = econf.__dict__
    _t['n_cs'] = len(cs)
    return _t


@click.command()
@click.option('--debug', is_flag=True, default=False, help="是否是调试模式")
@click.option('--file', is_flag=False, default="c06_hp_pruning_ratio.yaml", help="配置文件")
def main(file, debug):
    with open(f"resources/{file}", 'r') as f:
        cf = yaml.safe_load(f)
    random_state = cf['random_state']
    enable_numpy_reproduce(random_state)
    datasets = cf['datasets']
    outputs = []
    for _dataset in datasets:
        for _feature_selec_method in cf['feature_selec_method']:
            for _feature_selec_rate in cf['feature_selec_rate']:
                for _metric in cf['metrics']:
                    for _fold_index in range(cf['n_fold']):
                        for _hpo_opt_method in cf['hpo_opt_methods']:
                            for _max_iteration in cf['max_iteration']:
                                for _data_sample_method in cf['data_sample_method']:
                                    for _data_sample_rate in cf['data_sample_rate']:
                                        for _n_high_performing_model in cf['n_high_performing_model']:
                                            _c = ExpConf(
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
                                                data_sample_method=_data_sample_method,
                                                data_sample_rate=_data_sample_rate,
                                                max_iteration=_max_iteration,
                                                n_high_performing_model=_n_high_performing_model
                                            )
                                            outputs.append(_run(_c))

    pd.DataFrame(outputs).to_csv("cs_number.csv")


if __name__ == '__main__':
    main()

"""
python main_hpo.py  --file b02_result_compare.yaml
"""
