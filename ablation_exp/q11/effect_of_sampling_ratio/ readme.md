## reproduce

1 运行抽样预处理程序

python main.py --file c05_effect_of_sample_method_v2.yaml

2 运行
alg_find_of_rs.ipynb

3 运行
cd ablation_exp/q11/effect_of_sampling_ratio
CUDA_VISIBLE_DEVICES=-1 python main_effect_sample_ratio.py

4.

ana_effect_of_sr_for_tshpo.ipynb 得到结果