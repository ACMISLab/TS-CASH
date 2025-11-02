```
从gpt获取推荐算法及其超参数,包括3个GPT["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
python 01_get_suggestion_from_gpt.py
python 02_parse_model_and_args.py
python compare_auto_cash_and_tshpo.py
```

statics.csv: GPT 选用了哪些算法
GPT 更倾向于选择下面的算法：

``` 算法名称    选择次数
bernoulli_nb	1 
decision_tree	1 
extra_trees	2 
gaussian_nb	1 
libsvm_svc	2 
random_forest	77
```