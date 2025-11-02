"""
pip install protobuf==3.20
"""
from dataclasses import dataclass
import json
import etcd3
import typing


@dataclass
class Etcd:
    host: str = 'your_server_ip'
    port: int = 2379
    prefix: str = "tmp"
    client = None

    def __post_init__(self):
        self.client = etcd3.client(host=self.host, port=self.port)
        assert not self.prefix.endswith("/")
        assert not self.prefix.startswith("/")
        return self.client

    def _init_key(self, key):
        return f"{self.prefix}/{key}"

    def set(self, key, val) -> None:
        return self.save_data(key, val)

    def save(self, key, val) -> None:
        return self.set(key, val)

    def save_data(self, key, val) -> None:
        """保存数据"""
        if isinstance(val, dict):
            val = json.dumps(val)
            # val=val.encode("utf-8")
        assert isinstance(val, str)

        key = self._init_key(key)
        self.client.put(key, val)
        return True

    def get(self, key) -> typing.Union[dict, None]:
        return self.get_data(key)

    def get_data(self, key) -> typing.Union[dict, None]:
        """获取数据。同时也能检查数据是否存在
        如果key存在，就返回具体的值
        如果key不存在，就返回None
        """
        key = self._init_key(key)
        value, metadata = self.client.get(key)
        if value is not None:
            return json.loads(value.decode("utf-8"))
        else:
            return None

    def get_keys(self) -> list[str]:
        """获取指定  self.prefix 下所有的key
        例如：
        28b10c14278ed3fabceb58e0156a5de99af35389
        3262365309483f0803980ae817fbd036011a569c
        368f9e41679d9285f158c85f99721b5cae32c837
        """
        data = self.client.get_prefix(key_prefix=self.prefix)
        outputs = []
        for _, meta in data:
            outputs.append(meta.key.decode("utf-8").replace(self.prefix + "/", ""))
        return outputs


if __name__ == "__main__":
    etcd = Etcd(prefix="k8s_exp_results")
    print(etcd.get_keys())
    # etcd = Etcd()
    # etcd.save_data("test1","{'key':1}")
    # print(etcd.get_data("test1"))
