import threading
import time


class _MyThread(threading.Thread):
    def __init__(self, func, args):
        super(_MyThread, self).__init__()
        self.result = None
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


class UThread:

    @staticmethod
    def run_task_and_wait(fun_, args):
        """
        启动一个线程执行任务，等待完成并返回执行结果。

         def fun(aaa):
        time.sleep(1)
        return aaa + "~~~~~~~"


        th = UThread.run_task_and_wait(fun, ("sfs",))
        print("执行结果：", th)
        print("Done!")

        Parameters
        ----------
        fun_ :
        args :

        Returns
        -------

        """
        thread = _MyThread(fun_, args)

        # 启动线程
        thread.start()

        # 等待线程执行完成
        thread.join()
        return thread.get_result()


class UtilThreads:
    def __init__(self):
        self.jobs = []

    def append(self, fun_, args_):
        """

        Parameters
        ----------
        fun_ : function
        args_ :

        Returns
        -------

        """
        t = threading.Thread(target=fun_, args=(args_,))
        t.start()
        self.jobs.append(t)

    def append_without_args(self, fun_):
        """

        Parameters
        ----------
        fun_ : function
        args_ :

        Returns
        -------

        """
        t = threading.Thread(target=fun_)
        t.start()
        self.jobs.append(t)

    def start(self):
        for t in self.jobs:
            t.join()


if __name__ == '__main__':
    def fun(aaa):
        time.sleep(1)
        print("sdfsdfs")
        return str(aaa) + "~~~~~~~"


    ut = UtilThreads()
    ut.append(fun, "222")
    ut.append(fun, "222")
    ut.append(fun, "222")
    ut.append(fun, "222")
    ut.start()
