#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2024/3/22 09:43
# @Author  : gsunwu@163.com
# @File    : util_object.py
# @Description:
import base64
import pickle
from pathlib import Path


def cached_object(cache_name: Path, obj: object):
    if not cache_name.parent.exists():
        cache_name.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_name.as_posix(), "wb") as f:
        pickle.dump(obj, f)
    return cache_name


def load_cached_object(cache_name: Path):
    # #
    # with open(cache_name.as_posix(), "r") as f:
    #     data = f.read()
    # return load_str_as_object(data)
    with open(cache_name.as_posix(), 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data


def load_str_as_object(obj: object):
    serialized_data = base64.b64decode(obj)
    return pickle.loads(serialized_data)


def dump_object_as_str(obj: object):
    """
    tunnel_data=TunnelData(server_name=dask_worker.name,port_mappings=port_mappings)
    series_data=pickle.dumps(tunnel_data)
    b64_encoded_data = base64.b64encode(series_data).decode('utf-8')
    Parameters
    ----------
    obj :

    Returns
    -------

    """

    series_data = pickle.dumps(obj)
    return base64.b64encode(series_data).decode('utf-8')


if __name__ == '__main__':
    print(dump_object_as_str({
        "dkdjs": 1
    }))
