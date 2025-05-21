"""
pip install mysql-connector-python
"""
import sys
import time
from dataclasses import dataclass
from pathlib import PosixPath

from mysql.connector import ProgrammingError
from pylibs.config import Debug
import logging
from pylibs.utils.util_float import UtilFloat

log = logging.getLogger(__name__)
"""install mysql  by docker:

rm -rf /etc/docker/mysql
rm -rf /etc/docker/mysql/data
mkdir -p /etc/docker/mysql
mkdir -p /etc/docker/mysql/data
cat > /etc/docker/mysql/my.cnf <<EOF
[mysqld] 
lower_case_table_names=1 
sql_mode="STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION"

!includedir /etc/mysql/conf.d/
!includedir /etc/mysql/mysql.conf.d/ 
EOF

docker rm -f mysql

docker run -d --restart=always --name mysql -p 13306:3306 -v /etc/docker/mysql/my.cnf:/etc/mysql/my.cnf  -v /etc/docker/mysql/data:/var/lib/mysql -e MYSQL_ROOT_PASSWORD=your_password   mysql:5.7

docker logs mysql
"""
import mysql.connector


class DBField:
    """
    数据库的公用字段
    """
    FIELD_UPDATE_TIME = "update_time"
    FIELD_CREATE_TIME = "create_time"
    FIELD_ID = "id"


@dataclass
class UtilMysql:
    username: str = "root"
    password: str = 'your_password'
    ip: str = "your_server_ip"
    port: int = 13306
    database: str = "test"
    default_table_name: str = "default_table"
    retry: int = 3
    wait_seconds: int = 3

    def __post_init__(self):
        """

        Parameters
        ----------
        username :
        password :
        ip :
        port :
        database :
        default_table_name :
        retry : int
            The operation to retry
        wait_seconds: int
            The wait seconds to retry
        """
        self.conn = None
        self._wait_seconds = self.wait_seconds
        self._cursor = None
        self._default_table_name = self.default_table_name
        self._retry = self.retry
        self._database = self.database
        self._cursor = None
        try:
            self.init_conn()
        except ProgrammingError as e:
            log.info(f"Try to create new database [{self.database}]...")
            self.create_database()
            self.init_conn()

    def init_conn(self):
        log.info(f"Connection to {self.ip}:{self.port}")
        self.conn = mysql.connector.connect(
            host=self.ip,
            user=self.username,
            password=self.password,
            database=self.database,
            port=self.port
        )
        # Enable insert autocommit
        self.conn.autocommit = True

    def exec(self, query="show tables", print_results=False):
        log.info(f"Executing query: \n{query}\n")
        mycursor = self.get_cursor()
        mycursor.execute(query)
        query_results = mycursor.fetchall()
        if print_results:
            for x in query_results:
                log.info(x)
        return query_results

    def __del__(self):
        self.conn.close()
        self.conn.close()

    def get_cursor(self):
        if self._cursor is None:
            self._cursor = self.conn.cursor(dictionary=True)

        return self._cursor

    def insert(self, data: dict):
        """
        Insert data into the database.

        If the table is not existing, create a new one.
        If table structure is changed, remove old and recreate new table
        Parameters
        ----------
        data :

        Returns
        -------

        """
        assert "index" not in data.keys(), "data cant contain 'exp_index' key"
        log.info("Exec insert ...")
        # remove id
        del data['id']

        # insert_sql, values = self._insert_data(data)
        for i in range(self._retry):
            try:
                # self.get_cursor().execute(insert_sql, tuple(data.values()))
                self._insert_data(data)
                break
            except Exception as e:
                # 表结构改变
                if str(e).find("Unknown column") > -1:
                    # Table structure is changed, remove old and recreate new table
                    # self.delete_table()
                    # self.create_table(data)
                    create_table_query = self._get_create_table_sql(data)
                    log.info(
                        f"Table [{self.default_table_name}] structure is changed, remove old and recreate new table. \n{create_table_query}")
                    sys.exit(-1)

                # 表不存在
                if str(e).find("doesn't exist") > -1:
                    # ProgrammingError(1146, "1146 (42S02): Table 'test21.default_table' doesn't exist", '42S02')
                    self.create_table(data)
                Debug.errmsg(f"Exec insert failed since {e}, retry after {self._wait_seconds} seconds since")

            time.sleep(self._wait_seconds)

    def _insert_data(self, data):
        log.info("Exec _insert_data ...")

        columns = ', '.join(data.keys())
        _target_values = ""
        for _d in data.values():
            if type(_d) == PosixPath:
                _d = str(_d)
            _target_values += f"\"{_d}\","

        insert_sql = "INSERT INTO {} ({}) VALUES ({})".format(self._default_table_name, columns,
                                                              _target_values.rstrip(","))
        log.info(insert_sql)
        self.get_cursor().execute(insert_sql)

    def create_table(self, data):
        create_table_query = self._get_create_table_sql(data)
        log.info("Exec create table ...")
        self.exec(create_table_query)

    def _get_create_table_sql(self, data):
        # 生成CREATE TABLE语句
        create_table_query = f"CREATE TABLE IF NOT EXISTS {self._default_table_name} ("
        for key, value in data.items():
            column_name = key
            # if isinstance(value, int):
            #     column_type = "BIGINT"
            # elif isinstance(value, float):
            #     column_type = "FLOAT"
            # elif isinstance(value, bool):
            #     column_type = "INT"
            # else:
            #     column_type = "TEXT"
            create_table_query += f"{column_name} TEXT, "
        create_table_query = create_table_query.rstrip(", ") + ")"
        return create_table_query

    def get_all_data(self, limit=10):
        self.exec(f"select * from {self._default_table_name} limit {limit}")

    def is_table_exists(self, table_name=None):
        if table_name is None:
            table_name = self._default_table_name
        # 检查表是否存在的SQL语句
        check_table_query = f"SHOW TABLES LIKE '{table_name}'"

        self.get_cursor().execute(check_table_query)
        result = self.get_cursor().fetchone()
        if result:
            # 表存在
            return True
        else:
            # 表不存在
            return False

    def create_database(self):
        try:
            # 检查数据库是否存在的SQL语句
            conn = mysql.connector.connect(
                host=self.ip,
                user=self.username,
                password=self.password,
                port=self.port
            )
            check_database_query = f"SHOW DATABASES LIKE '{self._database}'"
            cursor = conn.cursor()
            cursor.execute(check_database_query)
            result = cursor.fetchone()

            if not result:
                # 如果数据库不存在,则创建数据库
                create_database_query = f"CREATE DATABASE {self._database}"
                cursor.execute(create_database_query)
                log.info(f"Database '{self._database}' created successfully.")
            else:
                log.info(f"Database '{self._database}' already exists.")

        except mysql.connector.Error as error:
            log.info(f"Error creating database: {error}")

    def delete_table(self):
        log.info(f"Delete table [{self.default_table_name}]...")

        self.exec(f"drop table {self.default_table_name}")

    def query_by_dict(self, conditions: dict):
        # log.info(f"Query by dict {conditions} ...")
        """
        在python中，使用{"name":"zhangsan"}(条件可能有多个）查询指定的数据，数据库是：mysql-connector-python

        Parameters
        ----------
        filter :

        Returns
        -------

        """
        # 构建WHERE子句

        where_clause = self.get_where_sql_clause_from_dict_condition(conditions)

        # 查询语句
        query = f"SELECT * FROM {self.database}.{self.default_table_name} WHERE {where_clause}"
        log.info(f"Query cmd: \n{query}\n")
        # 执行查询
        self.get_cursor().execute(query)

        # 获取查询结果
        results = self.get_cursor().fetchall()

        log.info(f"Results len: {len(results)}")
        return results

    @staticmethod
    def get_where_sql_clause_from_dict_condition(conditions):
        _conts = ""
        for _k, _v in conditions.items():
            _v = UtilMysql.normal_data(_v)
            _conts += f" and {_k}='{str(_v).strip()}' "

        return _conts.strip().lstrip("and")

    @classmethod
    def normal_data(cls, _v):
        """
        标准化数据，不然数据库不一定能查到
        Parameters
        ----------
        _v :

        Returns
        -------

        """
        if _v is None:
            return None
        elif isinstance(_v, int):
            return int(_v)
        elif isinstance(_v, str):
            return str(_v)
        elif isinstance(_v, float):
            return UtilFloat.format_float(_v)
        else:
            raise ValueError(f"Invalid type: {type(_v)} with {_v}")


# class MySQLFactor:
#     TYPE_100_9="type_100_9"
#     TYPE_RDS_MYSQL="ali_rds_mysql"
#     def __init__(self,type=MySQLFactor.TYPE_100_9):
#         if type==MySQLFactor.TYPE_RDS_MYSQL:
#             return UtilMysql(
#                 username = "sunwu",
#                 password= 'KB*bN55f3AN',
#                 ip = "rm-bp11z44447dn75e7npo.mysql.rds.aliyuncs.com",
#                 port = 3306,
#                 database = "p2_automl_exp",
#                 default_table_name = "default_table",
#                 retry = 3,
#                 wait_seconds  = 3)
#
#         elif type==MySQLFactor.TYPE_100_9:
#             #  username: str = "root"
#             #     password: str = 'your_password'
#             #     ip: str = "your_server_ip"
#             #     port: int = 13306
#             #     database: str = "test"
#             #     default_table_name: str = "default_table"
#             #     retry: int = 3
#             #     wait_seconds: int = 3
#             return UtilMysql(
#                 username="root",
#                 password='your_password',
#                 ip="your_server_ip",
#                 port=13306,
#                 database="p2_automl_exp",
#                 default_table_name="default_table",
#                 retry=3,
#                 wait_seconds=3)


if __name__ == '__main__':
    mq = UtilMysql(database="test21")

    mq.insert(data={
        "name": "zhasdfs",
        "age": 30,
        "city": "New York"
    })
    mq.query_by_dict({"name": "zhasdfs"})
    mq.insert(data={
        "name": "John2",
        "age": 302,
        "city": "New York",
        "sex2": "1"
    })
    mq.get_all_data()
