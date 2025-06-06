import os
import numpy as np

def analyze_filename(fn):
    x = fn.split('.')[0].split('_')
    train_test_split_pos = int(x[-3])
    abnormal_range = (int(x[-2]), int(x[-1]))

    return train_test_split_pos, abnormal_range

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

    train_test_split_pos, abnormal_range = analyze_filename(file_name)
    
    return l, train_test_split_pos, abnormal_range

