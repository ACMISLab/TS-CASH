from unittest import TestCase
from pylibs.utils.util_resume import ResumeHelper


class TestResumeHelper(TestCase):

    def test_nni_tools(self):
        r = ResumeHelper("32")
        r.record(32)
        assert r.get_latest_index() == 32

        r = ResumeHelper("32")
        r.record(26)
        assert r.get_latest_index() == 26
