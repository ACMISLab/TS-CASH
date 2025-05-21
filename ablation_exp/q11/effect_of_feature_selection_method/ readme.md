## reproduce

1 运行抽样预处理程序
CUDA_VISIBLE_DEVICES=-1 python main.py --file c05_effect_of_feature_method_v2.yaml

2 运行, 选出top2
alg_find_of_fsm_fsr.ipynb

3 运行
cd ablation_exp/q11/effect_of_feature_selection_method
CUDA_VISIBLE_DEVICES=-1 python main_effect_fsm.py --file Q11_effect_of_fsm.yaml