import unittest

from pylibs.utils.util_hash import get_str_hash


class MyTestCase(unittest.TestCase):
    def test_something(self):
        print(get_str_hash("aaa"))
        assert get_str_hash("aaa")==get_str_hash("aaa")
        assert get_str_hash("aaa")!=get_str_hash("aab")
        print(get_str_hash("bb"))


if __name__ == '__main__':
    unittest.main()
