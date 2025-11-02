from pyutils.grid.util_grid import GridSearch
from pyutils.pickle.util_pickle import save_pkl

datasets = ["d1",
            "d2",
            ]

cfgs = {
    # "fold_index": [1, 2, 3, 4, 5],
    "fold_index": [0],
    # "roc_auc", "f1",
    "eval_metric": ['precision_weighted','f1_weighted','recall_weighted'],
    "dataset_name": datasets,
    "seed": [42]

}
all_cfgs = GridSearch.get_search_item(cfgs)

print(all_cfgs)

cmds = []
for par in all_cfgs:
    cmds.append(
        f"python main_autogluon_ms.py --dataset_name {par['dataset_name']} --fold_index {par['fold_index']} --seed {par['seed']} --eval_metric {par['eval_metric']} \n")

save_pkl(cmds, "tasks.pkl")
with open("tasks.sh", "w") as f:
    f.writelines(cmds)
print(cmds)
