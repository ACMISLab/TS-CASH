"""
@summary 找出每个数据集上最好的算法名称及其fscore(acc+f1, 来源于 auto-cash)

"""
import sys

sys.path.append("/Users/sunwu/SW-Research/AutoML-Benchmark")
sys.path.append("/Users/sunwu/SW-Research/AutoML-Benchmark/deps")
from tshpo.automl_libs import *

metircs = AnaHelper.get_all_metrics()
df_baseline = AnaHelper.load_csv_file("c00_baseline_n500_madelon_original_20241029_1434.csv.gz")
# 在33个数据集上


metircs = AnaHelper.get_all_metrics()
datasets = sorted(df_baseline['dataset'].drop_duplicates().tolist())
# n_exploration = df['n_exploration'].drop_duplicates().tolist()

outputs = []
_arr = list(itertools.product(
    datasets,
    metircs,
    # n_exploration
))

scores = []
for _dataset in tqdm(datasets):
    _baseline_df = df_baseline[df_baseline['dataset'] == _dataset]
    top_models_baseline_acc, acc_baseline_acc = AnaHelper.get_models_and_acc_baseline(_baseline_df, metric="accuracy")
    top_models_baseline_roc_auc, acc_baseline_roc_auc = AnaHelper.get_models_and_acc_baseline(_baseline_df,
                                                                                              metric="roc_auc")

    # 算法顺序必须一致
    assert top_models_baseline_acc == top_models_baseline_roc_auc
    # acc_baseline_roc_auc 和 acc_baseline_acc 向量元素相乘
    auto_cash_fscore = np.asarray(acc_baseline_acc) * np.asarray(acc_baseline_roc_auc)

    max_index = np.argmax(auto_cash_fscore)
    tmp = {
        "dataset": _dataset,
        # Fscore = accuracy AUC
        "best_model": top_models_baseline_roc_auc[max_index],
        "best_autocash_fscore": auto_cash_fscore[max_index],
    }
    print(tmp)
    scores.append(tmp)
pd.DataFrame(scores).to_csv("autocash_每个数据集上最好的算法及其fscore.csv")
