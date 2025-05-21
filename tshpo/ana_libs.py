import itertools
import numpy as np
import pandas as pd
from tshpo.lib_class import AnaHelper, BaselineHelper
from tshpo.lib_func import get_memory

memory = get_memory()


class AnaDataLoader:
    @staticmethod
    @memory.cache
    def load_acc_select_data(csv_file, n_explorations=None, top_select_n=None):
        df = pd.read_csv(csv_file)
        metrics = df['metric'].drop_duplicates().tolist()
        datasets = df['dataset'].drop_duplicates().tolist()
        feature_selec_method = df['feature_selec_method'].drop_duplicates().tolist()
        feature_selec_rate = df['feature_selec_rate'].drop_duplicates().tolist()
        date_sample_rate = df['data_sample_rate'].drop_duplicates().tolist()
        date_sample_method = df['data_sample_method'].drop_duplicates().tolist()
        if n_explorations is None:
            n_explorations = df['n_exploration'].drop_duplicates().tolist()
            n_explorations = np.append([1], np.arange(10, np.max(n_explorations) + 1, 10))
        top_select_n = np.arange(1, 9, 1)
        # baseline_explorations = np.max(n_explorations)
        baseline_explorations = 50
        outputs = []
        for _feature_selec_method, _feature_selec_rate, _data_sample_method, _data_sample_rate, _dataset, _metric in itertools.product(
                feature_selec_method,
                feature_selec_rate,
                date_sample_method,
                date_sample_rate,
                datasets,
                metrics,
        ):

            _baseline_df = df[(df['feature_selec_method'] == _feature_selec_method) &
                              (df['feature_selec_rate'] == 1) &
                              (df['dataset'] == _dataset) &
                              (df['data_sample_rate'] == 1) &
                              (df['data_sample_method'] == _data_sample_method) &
                              (df['metric'] == _metric)
                              ]

            baseline_models, baseline_accs = AnaHelper.get_rank_model_and_acc(_baseline_df,
                                                                              baseline_explorations,
                                                                              metric=_metric)

            _test_df = df[
                (df['feature_selec_method'] == _feature_selec_method)
                & (df['feature_selec_rate'] == _feature_selec_rate)
                & (df['dataset'] == _dataset)
                & (df['data_sample_method'] == _data_sample_method)
                & (df['data_sample_rate'] == _data_sample_rate)
                & (df['metric'] == _metric)
                ]

            for _n_exploration in n_explorations:
                assert _n_exploration > 0
                _found_models, _found_accs = AnaHelper.get_rank_model_and_acc(_test_df, _n_exploration, metric=_metric)

                for _top_n in top_select_n:
                    _acc1 = int(len(np.intersect1d(baseline_models[:1], _found_models[:_top_n])) > 0)
                    _acc2 = int(len(np.intersect1d(baseline_models[:2], _found_models[:_top_n])) > 0)
                    _t = {
                        "feature_selec_method": _feature_selec_method,
                        "data_sample_method": _data_sample_method,
                        "data_sample_rate": _data_sample_rate,
                        "n_exploration": _n_exploration,
                        "fsr": _feature_selec_rate,
                        "dsr": _data_sample_rate,
                        "dataset": _dataset,
                        "metric": _metric,
                        "acc1": _acc1,
                        "acc2": _acc2,
                        "top_n": _top_n,
                    }
                    outputs.append(_t)
                    print(outputs[-1])
        adf = pd.DataFrame(outputs)
        # res = adf.groupby(by=['feature_selec_method', 'n_exploration', 'fsr', 'dsr', 'top_n', 'metric'])['acc'].agg(
        #     ["sum", 'count']).reset_index()
        # res['acc'] = res['sum'] / res['count']
        return adf

    @staticmethod
    @memory.cache
    def load_acc_select_data_v2(csv_file, n_explorations=None, top_select_n=None):
        df = pd.read_csv(csv_file)
        metrics = df['metric'].drop_duplicates().tolist()
        datasets = df['dataset'].drop_duplicates().tolist()
        feature_selec_method = df['feature_selec_method'].drop_duplicates().tolist()
        feature_selec_rate = df['feature_selec_rate'].drop_duplicates().tolist()
        date_sample_rate = df['data_sample_rate'].drop_duplicates().tolist()
        date_sample_method = df['data_sample_method'].drop_duplicates().tolist()
        if n_explorations is None:
            n_explorations = df['n_exploration'].drop_duplicates().tolist()
            n_explorations = np.append([1], np.arange(10, np.max(n_explorations) + 1, 10))
        top_select_n = np.arange(1, 7, 1)
        # baseline_explorations = np.max(n_explorations)
        baseline_explorations = 50
        outputs = []
        for _feature_selec_method, _feature_selec_rate, _data_sample_method, _data_sample_rate, _dataset, _metric in itertools.product(
                feature_selec_method,
                feature_selec_rate,
                date_sample_method,
                date_sample_rate,
                datasets,
                metrics,
        ):
            baseline_models, baseline_accs = BaselineHelper.get_models_and_acc_with_alpha(dataset=_dataset,
                                                                                          metric=_metric)

            _test_df = df[
                (df['feature_selec_method'] == _feature_selec_method)
                & (df['feature_selec_rate'] == _feature_selec_rate)
                & (df['dataset'] == _dataset)
                & (df['data_sample_method'] == _data_sample_method)
                & (df['data_sample_rate'] == _data_sample_rate)
                & (df['metric'] == _metric)
                ]

            for _n_exploration in n_explorations:
                assert _n_exploration > 0
                _found_models, _found_accs = AnaHelper.get_rank_model_and_acc(_test_df, _n_exploration, metric=_metric)

                for _top_n in top_select_n:
                    # _acc1 = int(len(np.intersect1d(baseline_models[:1], _found_models[:_top_n])) > 0)
                    # _acc2 = int(len(np.intersect1d(baseline_models[:2], _found_models[:_top_n])) > 0)
                    _acc = int(len(np.intersect1d(baseline_models, _found_models[:_top_n])) > 0)
                    _t = {
                        "feature_selec_method": _feature_selec_method,
                        "data_sample_method": _data_sample_method,
                        "data_sample_rate": _data_sample_rate,
                        "n_exploration": _n_exploration,
                        "fsr": _feature_selec_rate,
                        "dsr": _data_sample_rate,
                        "dataset": _dataset,
                        "metric": _metric,
                        # "acc1": _acc1,
                        # "acc2": _acc2,
                        "acc": _acc,
                        "top_n": _top_n,
                    }
                    outputs.append(_t)
                    print(outputs[-1])
        adf = pd.DataFrame(outputs)
        # res = adf.groupby(by=['feature_selec_method', 'n_exploration', 'fsr', 'dsr', 'top_n', 'metric'])['acc'].agg(
        #     ["sum", 'count']).reset_index()
        # res['acc'] = res['sum'] / res['count']
        return adf

    @staticmethod
    # @memory.cache
    def load_acc_select_data_v1(csv_file, n_explorations=None):
        df = pd.read_csv(csv_file)
        df = AnaHelper.pre_process(df)
        metrics = df['metric'].drop_duplicates().tolist()
        datasets = df['dataset'].drop_duplicates().tolist()
        feature_selec_method = df['feature_selec_method'].drop_duplicates().tolist()
        feature_selec_rate = [0.3]
        date_sample_rate = [0.5]
        if n_explorations is None:
            n_explorations = df['n_exploration'].drop_duplicates().tolist()
            n_explorations = np.append([1], np.arange(10, np.max(n_explorations) + 1, 10))
        top_select_n = np.arange(1, 9, 1)
        baseline_explorations = np.max(n_explorations)
        outputs = []
        for _feature_selec_method, _feature_selec_rate, _data_sample_rate, _dataset, _metric in itertools.product(
                feature_selec_method,
                feature_selec_rate,
                date_sample_rate,
                datasets,
                metrics,
        ):

            _baseline_df = df[(df['feature_selec_method'] == _feature_selec_method) &
                              (df['feature_selec_rate'] == 1) &
                              (df['dataset'] == _dataset) &
                              (df['data_sample_rate'] == 1) &
                              (df['metric'] == _metric)
                              ]

            baseline_models, baseline_accs = AnaHelper.get_rank_model_and_acc(_baseline_df,
                                                                              baseline_explorations,
                                                                              metric=_metric)

            _test_df = df[
                (df['feature_selec_method'] == _feature_selec_method)
                & (df['feature_selec_rate'] == _feature_selec_rate)
                & (df['dataset'] == _dataset)
                & (df['data_sample_rate'] == _data_sample_rate)
                & (df['metric'] == _metric)
                ]

            for _n_exploration in n_explorations:
                assert _n_exploration > 0
                _found_models, _found_accs, _time = AnaHelper.get_rank_model_and_acc(_test_df, _n_exploration,
                                                                                     metric=_metric,
                                                                                     return_time=True)

                for _top_n in top_select_n:
                    _acc1 = int(len(np.intersect1d(baseline_models[:1], _found_models[:_top_n])) > 0)
                    _acc2 = int(len(np.intersect1d(baseline_models[:2], _found_models[:_top_n])) > 0)
                    _t = {
                        "feature_selec_method": _feature_selec_method,
                        "n_exploration": _n_exploration,
                        "fsr": _feature_selec_rate,
                        "dsr": _data_sample_rate,
                        "dataset": _dataset,
                        "metric": _metric,
                        "acc1": _acc1,
                        "acc2": _acc2,
                        "top_n": _top_n,
                        "wall_time_seconds": _time
                    }
                    outputs.append(_t)
                    print(outputs[-1])
        adf = pd.DataFrame(outputs)
        # res = adf.groupby(by=['feature_selec_method', 'n_exploration', 'fsr', 'dsr', 'top_n', 'metric'])['acc'].agg(
        #     ["sum", 'count']).reset_index()
        # res['acc'] = res['sum'] / res['count']
        return adf

    @classmethod
    @memory.cache
    def load_baseline(cls, csv_file=None):
        # 每个算法探索100次，数据不抽样、不降维

        if csv_file is None:
            csv_file = "/Users/sunwu/Documents/baseline_original_20241019_0148.csv.gz"
        df = pd.read_csv(csv_file)
        metrics = AnaHelper.get_all_metrics()
        datasets = df['dataset'].drop_duplicates().tolist()
        feature_selec_method = df['feature_selec_method'].drop_duplicates().tolist()
        date_sample_method = df['data_sample_method'].drop_duplicates().tolist()
        feature_selec_rate = [1]
        date_sample_rate = [1]
        n_explorations = df['n_exploration'].drop_duplicates().tolist()
        _baseline_exploration = np.max(n_explorations)

        outputs = []
        for _feature_selec_method, _feature_selec_rate, _data_sample_method, _data_sample_rate, _dataset, _metric in itertools.product(
                feature_selec_method,
                feature_selec_rate,
                date_sample_method,
                date_sample_rate,
                datasets,
                metrics,
        ):
            _baseline_df = df[(df['feature_selec_method'] == _feature_selec_method) &
                              (df['feature_selec_rate'] == 1) &
                              (df['dataset'] == _dataset) &
                              (df['data_sample_rate'] == 1) &
                              (df['data_sample_method'] == _data_sample_method)
                              ]

            baseline_models, baseline_accs = AnaHelper.get_rank_model_and_acc(_baseline_df,
                                                                              _baseline_exploration,
                                                                              metric=_metric)

            _t = {
                "feature_selec_method": _feature_selec_method,
                "feature_selec_rate": _feature_selec_rate,
                "baseline_explorations": _baseline_exploration,
                "data_sample_method": _data_sample_method,
                "data_sample_rate": _data_sample_rate,
                "dataset": _dataset,
                "metric": _metric,
                "models": baseline_models,
                "accs": baseline_accs,
                "model_training_time_sum": _baseline_df['model_training_time'].sum(),
                "model_training_time_mean": _baseline_df['model_training_time'].mean(),
                "data_processing_time_sum": _baseline_df['data_processing_time'].mean(),
                "data_processing_time_mean": _baseline_df['data_processing_time'].mean(),

            }

            outputs.append(_t)
        return pd.DataFrame(outputs)


class PaperNormalizer:

    @staticmethod
    def normalize_title(title):
        title_rep_arr = [
            ['data_sample_method', 'dsm'],
            ['random', 'random'],
            ['dsm = random | dsr = 1.0', 'null'],
            ['dsm = stratified | dsr = 1.0', 'null'],
        ]
        for _old, _new in title_rep_arr:
            title = title.replace(_old, _new)
        return title


class PaperLabel:
    TOP_N_SELECTED_ALGORITHMS = "$A_{top-N}$"
    ACC_SELECT = "$Acc_{sel}$"


if __name__ == '__main__':
    # file = "/Users/sunwu/Downloads/b00_observation_v3_original_20241018_0321.csv.gz"
    # # df = AnaDataLoader.load_acc_select_data(File, n_explorations=[10])
    # df = AnaDataLoader.load_acc_select_data(file)
    # df = AnaDataLoader.load_acc_select_data_v1(file)
    # df
    # df=AnaDataLoader.load_baseline()
    # df
    # df = AnaDataLoader.load_acc_select_data(
    #     "/Users/sunwu/Downloads/b04_ablation_sample_method_original_20241018_0938.csv.gz",
    #     n_explorations=[1, 10, 20, 30])
    # from ana_libs import AnaDataLoader
    #
    # df = AnaDataLoader.load_acc_select_data_v2(
    #     "/Users/sunwu/Downloads/b04_ablation_sample_method_original_20241018_1241.csv.gz",
    #     n_explorations=[10, 20, 30])
    bdf = BaselineHelper.get_models_and_acc("pc1", 'f1')
    bdf
