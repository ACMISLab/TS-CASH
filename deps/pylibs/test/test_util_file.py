import os.path
import sys
from unittest import TestCase

from pylibs.utils.util_file import generate_random_file


class TestUtilFile(TestCase):
    def test_user_home(self):
        generate_random_file()
