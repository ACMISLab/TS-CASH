import os
import sqlite3
import time

from pylibs.utils.util_common import UtilComm
from pylibs.utils.util_log import get_logger
import traceback
log = get_logger()


def create_table(conn, sql=None):
    if sql is None:
        sql = '''
        CREATE TABLE tasks (
            job_id TEXT NOT NULL,
            exp_id TEXT NOT NULL,
            status INTEGER NOT NULL);
    '''
    if check_table_exists(conn, "tasks"):
        return True
    else:
        c = conn.cursor()  # 获取一个数据库游标
        # 使用 execute() 方法执行 SQL 命令
        c.execute(sql)

        # 提交事务
        conn.commit()


def check_table_exists(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
    result = cursor.fetchone()

    if result:
        UtilSys.is_debug_mode() and log.info(f"Table '{table_name}' exists.")
        return True
    else:
        UtilSys.is_debug_mode() and log.info(f"Table '{table_name}' does not exist.")
        return False


def get_connection_from_file(file, check_same_thread=False):
    # if not os.path.exists(file):
    #     raise ValueError(f"File [{os.path.abspath(file)}] is not existed")

    return sqlite3.connect(file, check_same_thread=check_same_thread)


def get_table_names_and_types(conn: sqlite3.Connection, table_name):
    """
    Get the columns and type of the tables.

    return names,types, e.g.,
    ['timestamp', 'trialJobId', 'parameterId', 'type', 'sequence', 'data'] and
    ['integer', 'text', 'text', 'text', 'integer', 'text)']

    Parameters
    ----------
    conn : sqlite3.Connection
        The connection.
    table_name : str
        The table names

    Returns
    -------
    list
        names
    list
        types

    """
    c = conn.cursor()
    c.execute(f"select * from sqlite_master where type='table' and name='{table_name}'")
    recored = c.fetchone()
    c.close()
    names = []
    types = []
    for line in recored[-1].split("(")[-1].split(','):
        line = str(line).strip()
        name, n_type = line.split(" ")
        names.append(f"{table_name}_{name}")
        types.append(n_type)
    return names, types


def exec_sql_by_sqlite_file(file, sql):
    """
    Execute SQL given a sqlite file.

    return records, columns_name

    Parameters
    ----------
    file : str
        A sqlite database file, e.g., nni.sqlite
    sql : str
        The SQL statement to execute

    Returns
    -------

    """
    return exec_sql(get_connection_from_file(file), sql)


def db_insert(conn: sqlite3.Connection, sql):
    """

    Parameters
    ----------
    conn :
    sql :

    Returns
    -------

    """
    if conn is not None:
        c = conn.cursor()
        exe_result = c.execute(sql)
        c.close()
        conn.commit()
        row_count = exe_result.rowcount
        UtilSys.is_debug_mode() and log.info(f"INSERT SQL: {sql}, row count:{row_count}")
        return row_count
    else:
        raise RuntimeError(f"Conn is None")


def db_update(conn: sqlite3.Connection, sql):
    if conn is not None:
        c = conn.cursor()
        exe_result = c.execute(sql)
        c.close()
        UtilSys.is_debug_mode() and log.info(f"UPDATE SQL: {sql}, result:{exe_result}")
        conn.commit()
        return exe_result.rowcount

    else:
        raise RuntimeError(f"Execute statement error: {sql}")


def exec_sql(conn: sqlite3.Connection, sql):
    """
    Execute sql and return the records and column names.

    Parameters
    ----------
    conn :
    sql :

    Returns
    -------
    list
        records. e.g.,
        [
            (1667467838155, 'tp0oL', '0', 'FINAL', 0, ... , 'vx7m2tus'),
            (1667467838156, 'tp0oL', '0', 'FINAL', 0, ... , 'vx7m2tus'),
            ...
        ]

    list
        names of each column. e.g.,
        ['timestamp', 'trialJobId', 'parameterId', 'type', 'sequence', 'metrics', 'hyperparameter', 'logpath', 'event', 'expinfo', 'expid']

    """
    if conn is not None:
        try:
            c = conn.cursor()
            c.execute(sql)
            recored = c.fetchall()
            names = [col[0] for col in c.description]
            c.close()
            return recored, names
        except Exception as e:
            log.error(traceback.format_exc())
            return None, None
    else:
        return None, None


def db_select(conn: sqlite3.Connection, sql):
    """
    Execute sql and return the records

    Parameters
    ----------
    conn :
    sql :

    Returns
    -------
    list
        records. e.g.,
        [
            (1667467838155, 'tp0oL', '0', 'FINAL', 0, ... , 'vx7m2tus'),
            (1667467838156, 'tp0oL', '0', 'FINAL', 0, ... , 'vx7m2tus'),
            ...
        ]

    """
    if conn is not None:
        c = conn.cursor()
        c.execute(sql)
        recored = c.fetchall()
        UtilSys.is_debug_mode() and log.info(f"SELECT SQL: {sql}, result:{recored}")
        c.close()
        return recored
    else:
        raise RuntimeError(f"Execute statement error: {sql}")


if __name__ == '__main__':
    db = get_connection_from_file(UtilComm.get_file_name("aaa.db"))
    create_table(db)
    UtilSys.is_debug_mode() and log.info(exec_sql(db, "select * from tasks"))
    insert = f"INSERT INTO tasks (job_id, exp_id, status) VALUES ('1', '{time.time()}', 3);"
    UtilSys.is_debug_mode() and log.info(db_insert(db, insert))

    UtilSys.is_debug_mode() and log.info(
        exec_sql(db, "select * from tasks where job_id='1' and exp_id='1689321668.170961'  "))
