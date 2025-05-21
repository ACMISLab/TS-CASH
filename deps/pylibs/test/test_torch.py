from unittest import TestCase

import torch

from pylibs.utils.util_log import get_logger
from pylibs.utils.util_pytorch import enable_pytorch_reproduciable, \
    get_l_out_of_max_pool_1d, random_select_one_gpu

log = get_logger()


class TestTF(TestCase):

    def test_l_out(self):
        # l_in, padding, dilation, kernel_size, stride
        kernel_size = 4
        l1 = get_l_out_of_max_pool_1d(l_in=54, padding=kernel_size // 2, kernel_size=kernel_size, stride=1)
        l1 = get_l_out_of_max_pool_1d(l1, kernel_size=2, stride=2, padding=1)
        assert l1 == 28

    def test_gpu_selected(self):
        print(random_select_one_gpu(avaliable_gpus="1"))

    def test_1dconv(self):
        m = torch.nn.Conv1d(16, 2, 3, stride=2)
        input = torch.randn(3, 16, 3)
        output = m(input)
        print(output)

    def test_aaa(self):
        m = torch.nn.Conv1d(22, 32, 4, 1)
        input = torch.randn(64, 1, 22)
        print(m(input))

    def test_patt(self):
        import re
        text = "在足球比赛中，横踢和腿法是两种重要的进攻方式。[('xx', 0.5714), ('腿法', 0.4286), ('进攻', 0.2857), ('比赛', 0.2857)],xxksjfl [('yy', 0.5714), ('腿法', 0.4286), ('进攻', 0.2857)]"

        _arr = []
        for _t in re.findall("\[(.+?)\]", text):
            _name = re.findall("'(.+?)'\s*,\s*[0-9.]+?", _t)
            _val = re.findall("'.+?'\s*,\s*([0-9.]+)", _t)
            _key_val = list(zip(_name, _val))
            _arr = _arr + _key_val
        print(_arr)
