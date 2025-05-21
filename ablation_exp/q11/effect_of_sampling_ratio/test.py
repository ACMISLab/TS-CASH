from pylibs.utils.util_rsync import Rsync
from pyutils.util_servers import Servers

Rsync.upload_file(Servers.GPU1009,
                  local_file="/Users/sunwu/SW-Research/AutoML-Benchmark/ablation_exp/q11/effect_of_sampling_ratio/rs_top2_alg.sqlite")
