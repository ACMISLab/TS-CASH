import redis
import time


# 连接到Redis
# 7.2: https://github.com/zkteco-home/redis-windows
class RedisConf:
    def __init__(self, host="localhost", port=6379, db=0):
        self._host = host
        self._port = port
        self._db = db

    def get_connection(self):
        return redis.Redis(host=self._host, port=self._port, db=self._db, decode_responses=True)


class RedisQueue:
    def __init__(self, queue_name="task_queue", redis_conf=RedisConf()):
        """
        先进先出的数据结构
        Parameters
        ----------
        queue_name :
        redis_conf :
        """
        self._queue_name = queue_name
        self._redis_conf = redis_conf
        self._r = self._redis_conf.get_connection()

    def produce(self, task):
        """向队列添加任务"""
        return self._r.lpush(self._queue_name, task)

    def count(self):
        return self._r.llen(self._queue_name)

    def add(self, task):
        return self.produce(task)

    def get_one_task(self):
        task = self._r.rpop(self._queue_name)
        return task

    def get_all(self) -> list:
        return self._r.lrange(self._queue_name, 0, -1)

    def is_exist(self, element) -> bool:
        position = self._r.lpos(self._queue_name, element)
        if position is not None:
            print(f"元素 '{element}' 存在于列表中，位置为: {position}")
            return True
        else:
            print(f"元素 '{element}' 不存在于列表中")
            return False


if __name__ == "__main__":
    rq = RedisQueue("list_test")
    for i in range(10):
        rq.produce(f"task_{i}")
    print(rq.get_all())
