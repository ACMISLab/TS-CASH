The source code and data for "A Two-Stage Framework with Search Space Pruning for Combined Algorithm Selection and
Hyperparameter Optimization"

## How to access the experimental data?

The experimental data is avaliable at: http://restic.gwusun.top/tscash/tscash_expermental_data.zip

## How to run

0. Prepare the environment:

``` 
Set the experiment: export TSHPO_HOME=/Users/xx/tshpo
```

``` 
Configure your MySQL user and password settings in line 42 of the file deps/pylibs/kvdb_mysql.py, 
```

``` 
conda create -n tshpo310 python=3.10
pip install -r requirement_ubuntu20.04.txt # for ubuntu
or 
pip install -r requirements_mbp_m1.txt # for mac m1 
```

1. Prepare the dataset:

```
cd deps/datasets/openml
prepared_dataset_cc18_binary_classification.py
```

2. Run the experiment:

```
python main_observation.py -f c00_baseline_n500_madelon.yaml
python main_observation.py -f c09_select_optimal_alg_v2.yaml
python main_tshpo.py -f c04_tshpo_acc_compare_v12_middle.yaml
```

 