
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#from slide_window import create_window_list


def analyze_filename(fn):
    x = fn.split('.')[0].split('_')
    split_pos = int(x[-3])
    anomaly_range = (int(x[-2]), int(x[-1]))

    return split_pos, anomaly_range

def get_series(num):
    l = []
    num_str = '%03d' % num
    file_name = ""
    for filename in os.listdir("/Users/sunwu/SW-OpenSourceCode/AutoML-Benchmark/deps/datasets/ucr/data/"):
        x = filename.split('_')
        if x[0] == num_str:
            file_name = filename
            for line in open(os.path.join("/Users/sunwu/SW-OpenSourceCode/AutoML-Benchmark/deps/datasets/ucr/data/", filename)):
                l.append(float(line.strip()))

    split_pos, anomaly_range = analyze_filename(file_name)
    
    return l, split_pos, anomaly_range
