import pickle

pickle_file = "../meta_datas/select_features.pkl"
conf = pickle.load(open(pickle_file, "rb"))
print(conf)
