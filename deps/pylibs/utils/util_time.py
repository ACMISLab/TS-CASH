import time


class RunningTime(object):
    def __init__(self, func, *args, **kwargs):
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def __call__(self, *args, **kwargs):
        start = time.time()
        self._func(*args, **kwargs)
        end = time.time()
        print("The running time(s) of function [%s] is %.6fs" % (self._func.__name__, (end - start)))


if __name__ == '__main__':
    @RunningTime
    def bar():
        time.sleep(1)
        print('bar')


    bar()
