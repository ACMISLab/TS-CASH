# pip install --upgrade pip
# pip install cyberdb
import time

import cyberdb
import numpy as np

from pylibs.application.datatime_helper import DateTimeHelper
from pylibs.config import GCF, ExpServerConf, ServerName
from pylibs.utils.util_common import UtilComm
from pylibs.utils.util_log import logwarn, loginfo
from pylibs.utils.util_system import UtilSys


class CyberDB:

    def __init__(self, conf: ExpServerConf = GCF.get_server_conf(ServerName.CYBERDB_LAN)):
        # 生成客户端实例并连接。
        client = cyberdb.connect(host=conf.ip, port=int(conf.port), password=conf.password)
        self.client = client

    def save_dict(self, id, data: dict):
        UtilSys.is_debug_mode() and loginfo(f"Data with [{id}] = \n{data}")
        proxy = self.client.get_proxy()
        proxy.connect()
        try:

            proxy.create_cyberdict(id)
            pdict = proxy.get_cyberdict(id)
            pdict.update(data)
            loginfo("Data updated")
        except Exception as e:
            # traceback.print_exc()
            # raise e
            logwarn(f"key [{id}] is existed")
        finally:
            proxy.close()

    def get_dict(self, id):
        proxy = self.client.get_proxy()
        proxy.connect()
        _data = proxy.get_cyberdict(id)
        ret = dict(_data)
        del _data
        proxy.close()
        return ret


if __name__ == '__main__':
    from pylibs.experiments.exp_config import ExpConf

    conf = ExpConf(model_name="as", dataset_name="IOPS",
                   data_id="KPI-6d1114ae-be04-3c46-b5aa-be1a003a57cd.train.out",
                   data_sample_method="random",
                   data_sample_rate=time.time(),
                   exp_index=1, exp_total=10,
                   exp_name="djls",
                   metrics_save_home=UtilComm.get_system_runtime(),
                   test_rate=time.time_ns()
                   )
    dth = DateTimeHelper()
    saved_metrics = conf.save_score_dth_exp_conf_to_redis(score=np.asarray([1, 234]), dt=dth)
    load_data = CyberDB().get_dict(conf.get_exp_id())
    print(load_data)
    assert load_data == saved_metrics
