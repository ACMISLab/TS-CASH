from pylibs.utils.util_common import UtilComm
from pylibs.utils.util_sqlite import get_connection_from_file, create_table, db_insert, exec_sql, db_update
from pylibs.utils.util_log import get_logger

log = get_logger()


class JobManager:
    CREATE_TABLE_SQL = '''
        CREATE TABLE tasks (
            exp_id TEXT PRIMARY KEY ,
            status INTEGER NOT NULL);
    '''
    DONE_STATUS = 0
    ERROR_STATUS = -1

    def __init__(self, id):
        self._db = get_connection_from_file(UtilComm.get_file_name(f"{id}.sqlite"))
        create_table(self._db, JobManager.CREATE_TABLE_SQL)

    def set_exp_done(self, exp_id):
        insert = f"INSERT INTO tasks ( exp_id, status) VALUES ('{exp_id}', {JobManager.DONE_STATUS});"
        try:
            db_insert(self._db, insert)
        except Exception as e:
            db_update(self._db, f"UPDATE tasks SET status  = {JobManager.DONE_STATUS} WHERE exp_id = '{exp_id}';")
        return

    def get_exp_status(self, exp_id):
        res, header = exec_sql(self._db, f"select * from tasks where exp_id='{exp_id}'  ")
        if len(res) == 0:
            return JobManager.ERROR_STATUS
        else:
            return res[0][1]

    def is_exp_run(self, exp_id):
        """
        任务是否执行过
        Returns
        -------

        """
        res, header = exec_sql(self._db, f"select * from tasks where exp_id='{exp_id}'  ")
        return len(res) > 0


if __name__ == '__main__':
    jb = JobManager("df")
    jb.set_exp_done("3a")
    assert jb.is_exp_run("3a") is True
    assert jb.is_exp_run("3b") is False
