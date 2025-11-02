先执行预选算法脚本, 为后面的剪枝做准备: /Volumes/sw_data/phd_paper_data/experiments/ts_cash_ms/deps/microservices/fc_classifier/alg_evaluation_ms.py,



main_ms.py 只处理剪枝的情况,n_high_performing_model 设置为保留的算法的数量
main_ms_baseline.py 只处理不剪枝的情况, n_high_performing_model 设置为None



调试:
python entroy.py --conf /Volumes/sw_data/phd_paper_data/experiments/ts_cash_ms/ts_hpo_ms/hpys/main_ms_baseline_16_30e5f53cea376d02798764906febd41b.pkl
Project home: /Volumes/sw_data/phd_paper_data/experiments/ts_cash_ms