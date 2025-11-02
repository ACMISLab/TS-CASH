#     if not os.path.exists("test_files.csv"):
#         dl = DatasetLoader(DatasetType.IOPS, 1)
#         train_x, train_y = dl.get_sliding_windows()
#
#         modelName = 'IForest'
#         clf = IForest(n_jobs=1)
#         clf.fit(train_x)
#         score = clf.decision_scores_
#
#         # Post-processing
#         score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
#         from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView
#
#         uv = UnivariateTimeSeriesView()
#         uv.plot_kpi_with_anomaly_score_row1(train_x[:, -1], train_y, score)
#
#         pd.DataFrame({
#             "score": score,
#             "train_y": train_y
#         }).to_csv("test_files.csv")
#
# def test_RangeAUC_parallel(self):
#     # pd.DataFrame({
#     #     "score": score,
#     #     "train_y": train_y
#     # }).to_csv("test_files.csv")
#     df = pd.read_csv("test_files.csv")
#     score = df.iloc[:, 0]
#     train_y = df.iloc[:, 1]
#
#     slidingWindow = 99
#     grader = metricor()
#     R_AUC_ROC1, R_AUC_PR1, _, _, _ = grader.RangeAUC(train_y,
#                                                      score=score, window=slidingWindow,
#                                                      plot_ROC=True)
#     R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC_parallel(train_y,
#                                                             score=score, window=slidingWindow,
#                                                             plot_ROC=True)
#
#     UtilSys.is_debug_mode()  and log.info(f"R_AUC_ROC:{R_AUC_ROC},R_AUC_PR:{R_AUC_PR} ")
#     assert_almost_equal(R_AUC_ROC, R_AUC_ROC1, decimal=7)
#     assert_almost_equal(R_AUC_PR, R_AUC_PR1, decimal=7)
#
#     if slidingWindow is None:
#         metrics = get_metrics_parallel(score, train_y, "all", 99)
