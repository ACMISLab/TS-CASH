import os
import time
from multiprocessing import Pool


def funx(x):
    time.sleep(3)
    print(x)


if __name__ == '__main__':
    with Pool(1024) as p:
        p.map(funx, [i for i in range(10000)])
