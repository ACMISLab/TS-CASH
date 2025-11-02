import pickle

import pandas as pd

pickle_file = "../meta_datas/all_data.pkl"
conf = pickle.load(open(pickle_file, "rb"))
print(conf)
pd.DataFrame(conf['dresses-sales:::adaboost'])
best_conf = pd.DataFrame(conf['dresses-sales:::adaboost']).sort_values(by=['label'], ascending=False).iloc[0]
init_hpy = best_conf.to_dict()
del init_hpy['label']
print(best_conf)
