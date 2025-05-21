import argparse
from unittest import TestCase

from pylibs.utils.util_argparse import get_number_of_experiments


class TestArgs(TestCase):
    def test_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", nargs="+")
        parser.add_argument("--seed", nargs="+")
        args = parser.parse_args("--model IF COCA --seed 1 2 3".split())
        self.assertEqual(get_number_of_experiments(args), 6)

        parser = argparse.ArgumentParser()
        parser.add_argument("--model", nargs="+")
        parser.add_argument("--seed", nargs="+")
        args = parser.parse_args("--model IF --seed 1 2 3".split())
        self.assertEqual(get_number_of_experiments(args), 3)
