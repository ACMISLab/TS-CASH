from unittest import TestCase

from pylibs.utils.util_args import parse_cmd_to_dict


class TestCmdToDict(TestCase):
    def test1(self):
        # {'python': 'models/deepsvdd.py', 'm_dataset': 'SHUTTLE', 'seed': '0', 'data_sample_method': 'RS', 'data_sample_rate': ''}
        res=parse_cmd_to_dict("python models/deepsvdd.py  --m_dataset SHUTTLE  --seed 0 --data_sample_method RS --data_sample_rate  1.0")
        assert res["python"]=="models/deepsvdd.py"
        assert res["m_dataset"]=="SHUTTLE"
        assert res["seed"]=="0"
        assert res["data_sample_method"]=="RS"
        assert res["data_sample_rate"]=="1.0"
        print(res)


        res=parse_cmd_to_dict("python models/deepsvdd.py               --m_dataset SHUTTLE  --seed         0     --data_sample_method         RS --data_sample_rate  1.0x")
        assert res["python"]=="models/deepsvdd.py"
        assert res["m_dataset"]=="SHUTTLE"
        assert res["seed"]=="0"
        assert res["data_sample_method"]=="RS"
        assert res["data_sample_rate"]=="1.0x"
