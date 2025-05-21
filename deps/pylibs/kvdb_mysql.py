"""
msql 安装:
mkdir -p /docker/appdata/mysql901_tshpo
mkdir -p /docker/appdata/mysql901_tshpo/data
cat > /docker/appdata/mysql901_tshpo/my.cnf <<EOF
[mysqld]
lower_case_table_names=1
sql_mode="STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION"
max_connections = 1000
EOF

docker rm -f mysql901_tshpo
docker run -d --restart=always -p 33306:3306  --name mysql901_tshpo -v /docker/appdata/mysql901_tshpo/my.cnf:/etc/mysql/my.cnf  -v /docker/appdata/mysql901_tshpo/data:/var/lib/mysql -e MYSQL_ROOT_PASSWORD=your_password  -v /etc/localtime:/etc/localtime registry.cn-hangzhou.aliyuncs.com/sunwuer/common:mysql_9_0_1
docker logs mysql901_tshpo

pip install mysql-connector-python
"""

import mysql.connector
import ast
from mysql.connector import IntegrityError
from pylibs.util_log import getlog
from pylibs.utils.util_str import get_str_hash_sha256

log = getlog()


class KVDBMySQL:
    """一个基于 MySQL 的 kvdb，插入和查询都是基于 key
    """

    def _sort_dict_key(self, sort_d):
        """对dict的key进行排序"""
        return {key: sort_d[key] for key in sorted(sort_d.keys())}

    def _round_dict_value(self, query_key, decimal=6):
        """标准化dict的key"""
        ret = {}
        for key, value in query_key.items():
            if isinstance(value, (float, int)):
                value = float(round(value, decimal))
            ret.update({key: value})
        return ret

    def __init__(self, host="your_server_ip", port=33306, user="root", password="passwd", database="tshpo_kvdb",
                 table_name="kvdb_main"):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.table_name = table_name

        # 连接到 MySQL 服务器（不指定数据库）
        self.conn = mysql.connector.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password
        )
        self.cursor = self.conn.cursor()

        # 检查数据库是否存在，如果不存在则创建
        self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{self.database}`")
        self.conn.commit()

        # 重新连接到指定的数据库
        self.conn.database = self.database

        # 创建表，如果不存在
        # key 查询的hash字符串
        # key_str: 查询的原始字符串
        self.cursor.execute(f"""
                  CREATE TABLE IF NOT EXISTS {self.table_name} (
                      `key_hash`  CHAR(64) PRIMARY KEY,
                      `key` TEXT,
                      `value` TEXT
                  )
              """)
        self.conn.commit()

    def add(self, k, v, update=False):
        """
        插入或更新键值对

        Parameters
        ----------
        k : dict
            键，必须是字典
        v : dict
            值，必须是字典
        update : bool, default False
            是否更新现有的值

        Returns
        -------
        None
        """
        assert isinstance(k, dict)
        assert isinstance(v, dict)
        _insert_key = self.prepare_query_key(k)
        _insert_key_hash = get_str_hash_sha256(_insert_key)
        _insert_value = self.prepare_query_key(v)

        try:
            log.debug(f"add new data {k}: {v}")
            self.cursor.execute(
                f"INSERT INTO {self.table_name} (`key_hash`, `key`, `value`) VALUES (%s,%s, %s)",
                (_insert_key_hash, _insert_key, _insert_value)
            )
            self.conn.commit()
        except IntegrityError:
            log.debug("skip since key already exists, you can set update=True to overwrite current value")
            if update:
                self.cursor.execute(
                    f"REPLACE INTO {self.table_name} (`key_hash`, `key`, `value`) VALUES (%s,%s, %s)",
                    (_insert_key_hash, _insert_key, _insert_value)
                )
                self.conn.commit()

        except Exception as e:
            raise e

    def query(self, k):
        """
        查询键对应的值

        Parameters
        ----------
        k : dict
            键，必须是字典

        Returns
        -------
        dict or None
            返回值，如果键不存在则返回 None
        """
        assert isinstance(k, dict)
        query_key = self.prepare_query_key(k)
        query_key_hash = get_str_hash_sha256(query_key)

        self.cursor.execute(f"SELECT `value` FROM {self.table_name} WHERE `key_hash` = %s", (query_key_hash,))
        result = self.cursor.fetchone()
        if result:
            return ast.literal_eval(result[0])
        else:
            return None

    def prepare_query_key(self, k):
        """
        预处理键，确保键是排序并规范化的字符串

        Parameters
        ----------
        k : dict
            键，必须是字典

        Returns
        -------
        str
            处理后的键
        """
        query_key = self._sort_dict_key(k)
        query_key = self._round_dict_value(query_key)
        return str(query_key)


if __name__ == '__main__':
    db = KVDBMySQL()
    db.add({"a": '1'}, {'b': 2})
    print(db.query({"a": '1'}))
