from pylibs.uts_dataset.dataset_loader import DataProcessingType, KFoldDatasetLoader


def load_dataset_kfold_fast_uts(conf):
    """
    每次训练的训练集和测试集

    Parameters
    ----------
    conf :

    Returns
    -------

    """
    is_include_anomaly_window = True
    if conf.is_semi_supervised_model():
        is_include_anomaly_window = False

    dl = KFoldDatasetLoader(
        conf.dataset_name,
        conf.data_id,
        sample_rate=conf.data_sample_rate,
        window_size=conf.window_size,
        is_include_anomaly_window=is_include_anomaly_window,
        processing=DataProcessingType.NORMAL,
        anomaly_window_type=conf.anomaly_window_type,
        test_rate=conf.test_rate,
        fill_nan=True
    )

    for _fold_index in range(conf.kfold):
        yield dl.get_kfold_sliding_windows_train_and_test_by_fold_index(_fold_index)
