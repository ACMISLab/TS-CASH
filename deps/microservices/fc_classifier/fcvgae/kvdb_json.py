import os
import json
import ast
import fcntl
import logging
import sys
import time
from typing import Optional, List

import pandas as pd


def _sort_dict_key(d: dict) -> dict:
    """对 dict 的 key 进行排序"""
    return {k: d[k] for k in sorted(d.keys())}


def _round_dict_value(d: dict, decimal: int = 6) -> dict:
    """对 dict 的 value 四舍五入"""
    ret = {}
    for k, v in d.items():
        if isinstance(v, (float,)):
            v = float(round(v, decimal))
        ret[k] = v
    return ret


class KVDBJson:
    """
    基于 JSON 文件的 KVDB，支持多进程安全写入。
    key 和 value 都是 dict，内部会对其 key 排序并对数值四舍五入，再 str 化或存储。
    """

    def __init__(self, json_file: Optional[str] = f"kvdb_{os.uname().nodename}_{os.path.basename(sys.argv[0])}.json"):
        # 默认文件名
        if json_file is None:
            fname = f"kvdb_{os.uname().nodename}.json"
            self.json_file = os.path.abspath(fname)
        else:
            self.json_file = os.path.abspath(json_file)

        # 如果文件不存在，则创建空的 {}
        if not os.path.exists(self.json_file):
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2, ensure_ascii=False)

    def _read_all(self) -> dict:
        """
        读出整个 JSON 对象，使用共享锁（LOCK_SH）
        """
        with open(self.json_file, 'r', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return data

    def _write_all(self, data: dict):
        """
        将整个 JSON 对象写回文件，使用独占锁（LOCK_EX）
        写前会 truncate，以避免残留旧数据
        """
        with open(self.json_file, 'r+', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def prepare_query_key(self, k: dict) -> str:
        """对 key(dict) 排序、四舍五入后 str 化"""
        assert isinstance(k, dict), "key 必须是 dict"
        sorted_k = _sort_dict_key(k)
        rounded_k = _round_dict_value(sorted_k)
        return str(rounded_k)

    def add(self, k: dict, v: dict, update: bool = False):
        """
        插入或更新一条记录。
        k: dict, 作为 key
        v: dict, 作为 value
        update: 如果key已存在，是否 overwrite
        """
        key_str = self.prepare_query_key(k)
        # 对 value 做同样的排序+四舍五入，但存为 dict
        sorted_v = _sort_dict_key(v)
        rounded_v = _round_dict_value(sorted_v)

        # 先读出全量数据
        data = self._read_all()

        if key_str in data:
            if update:
                logging.debug(f"update existing key {k} with value {v}")
                data[key_str] = rounded_v
                self._write_all(data)
            else:
                logging.info("skip since key already exists, "
                             "you can set update=True to overwrite")
        else:
            logging.debug(f"add new data {k}: {v}")
            data[key_str] = rounded_v
            self._write_all(data)

    def query(self, k: dict) -> Optional[dict]:
        """
        根据 key(dict) 查询 value(dict)，不存在返回 None
        """
        key_str = self.prepare_query_key(k)
        data = self._read_all()
        return data.get(key_str)

    def query_all(self) -> List[List[dict]]:
        """
        返回所有记录，格式 [[key_dict, value_dict], ...]
        """
        data = self._read_all()
        result = []
        for ks, vs in data.items():
            key_dict = ast.literal_eval(ks)
            result.append([key_dict, vs])
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """
        将所有记录合并成 DataFrame，每条记录 key 和 value 合并为一行
        """
        rows = []
        for key_dict, val_dict in self.query_all():
            merged = {}
            merged.update(key_dict)
            merged.update(val_dict)
            rows.append(merged)
        return pd.DataFrame(rows)

    def to_csv(self, csv_file_name: str) -> str:
        """
        导出到 CSV 文件
        """
        df = self.to_dataframe()
        df.to_csv(csv_file_name, index=False)
        return csv_file_name


def worker(name, db_path):
    db = KVDBJson(db_path)
    db.add({'proc': name, 'i': 1, "time": time.time()}, {'time': 3.1415})
    print(name, db.query({'proc': name, 'i': 1}))


if __name__ == '__main__':

    from multiprocessing import Process

    path = 'testdb.json'
    procs = []
    for n in range(5):
        p = Process(target=worker, args=(f'p{n}', path))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # 最终所有进程写入的数据都能保留
    db = KVDBJson(path)
    print(db.query_all())
    print(db.to_dataframe())
