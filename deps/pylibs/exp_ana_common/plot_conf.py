import os
from typing import List









class PlotConf:
    baseline_dir_ = "/Users/sunwu/Documents/exp_data/343/V343-baseline_classical_model_sklearn_debug_0_ECG_SVDB_IOPS"
    # found_dir_ = "/Users/sunwu/Downloads/download_metrics_153/100-main_fastuts_tf_e_dist1"
    # found_dir_ = "/Users/sunwu/Documents/exp_data/343/V343-main_fastuts_sklearn_dist1"
    # found_dir_ = "/Users/sunwu/Documents/exp_data/343/V343-main_fastuts_sklearn_lhs"
    found_dir_ = "/Users/sunwu/Documents/exp_data/343/V343-main_fastuts_sklearn_random"

    @staticmethod
    def get_exp_name():
        return os.path.basename(PlotConf.baseline_dir_) + "_" + os.path.basename(PlotConf.found_dir_)

#
# def _generate_configs(target_metrics, sample_methods, stop_alhpas, m_type="dl",
#                       home=f"/Users/sunwu/Documents/exp_data/{JobConfV1.get_exp_version()}"):
#     configs = []
#     for _target_metric in target_metrics:
#         for _sample_method in sample_methods:
#             for _stop_alpha in stop_alhpas:
#                 configs.append(ResConf(
#                     base_file=os.path.join(home,
#                                           f"{JobConfV1.get_exp_version()}_baseline_{m_type}_VUS_ROC_0.001_random"),
#                     fastuts_dir=os.path.join(home,
#                                              f"{JobConfV1.get_exp_version()}_fastuts_{m_type}_{_target_metric}_{_stop_alpha}_{_sample_method}"),
#                     desc=f"{m_type}_{_target_metric}_{_sample_method}",
#                     group=f"{m_type}_{_target_metric}"
#                 ))
#     return configs
