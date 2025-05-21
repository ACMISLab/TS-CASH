import time
import unittest

from pylibs.utils.util_bash import exec_cmd


class MyTestCase(unittest.TestCase):
    def test_file(self):
        exec_cmd("echo ''>temp.txt")
        while True:
            with open("temp.txt", "r") as f:
                print(f.read())
                time.sleep(1)
