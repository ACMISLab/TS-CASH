import os
import re
import subprocess
from unittest import TestCase

from pylibs.experiments.NNIExpGenerator import NNIExpGenerator
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_nnictl import create_nni_experiment
from pylibs.utils.util_system import UtilSys

log = get_logger()


class TestExpGen(TestCase):
    def test_exp_resume(self):
        max_trial_number = 30
        search_space_file = "nni_test_search_space.json"
        search_space = """{
              "batch_size": {
                    "_type": "uniform",
                    "_value": [
                      256,
                      1024
                    ]
               }
            }
        """
        with open(search_space_file, 'w') as f:
            f.write(search_space)
        model_file = """
import nni
import numpy as np
nni.get_next_parameters()
nni.report_final_result({
    "default": np.random.random(),
    "test_auc": np.random.random()
})"""
        nni_debug_model_file = "nni_debug_model.py"
        with open(nni_debug_model_file, 'w') as f:
            f.write(model_file)

        neg = NNIExpGenerator(
            exp_name=f"test_resume_{max_trial_number}",
            home=os.getcwd(),
            max_trial_number=max_trial_number,
            search_space_file=search_space_file,
            model_entry_file=nni_debug_model_file,
            dry_run=False
        )
        neg.start_or_restart()

    def test_start(self):
        cmd = "nnictl create  --port 49493 --config /Users/sunwu/SW-Research/p1_ch05/experiments/configs/DEBUG_1672124080472839000_DID-DATA_ID_DAT-DEBUG_NTR-RANDOM_MTN-1_MTC-1_RET-0_SED-0_DSM-RANDOM_DSR-1.0_MON-DEFAULT_MEF-MAIN.yaml"
        # cmd = "where nnictl"
        ret = subprocess.run(cmd, capture_output=True, shell=True, encoding="utf-8")
        print(ret.stderr)

        # stdout, stderr = exec_cmd_return_stdout_and_stderr(cmd, check=False)
        # print(stdout, stderr)

    def test_reg(self):
        message = """
          File "/Users/sunwu/SW-Research/nni-2.10/nni/experiment/launcher.py", line 105, in start_experiment
    _ensure_port_idle(port)
  File "/Users/sunwu/SW-Research/nni-2.10/nni/experiment/launcher.py", line 185, in _ensure_port_idle
    raise RuntimeError(f'Port {port} is not idle {message}')
RuntimeError: Port 49493 is not idle """
        g = re.search("(RuntimeError: Port\s+(\d+)\sis not idle)", message)
        UtilSys.is_debug_mode() and log.info(f"===>re:\n{g.groups()}")

    def test_create_experiment(self):
        cmd = "nnictl create  --port 49493 --config /Users/sunwu/SW-Research/p1_ch05/experiments/configs/DEBUG_1672124080472839000_DID-DATA_ID_DAT-DEBUG_NTR-RANDOM_MTN-1_MTC-1_RET-0_SED-0_DSM-RANDOM_DSR-1.0_MON-DEFAULT_MEF-MAIN.yaml"
        # create_nni_experiment(cmd)
        create_nni_experiment(cmd)

    def test_create_experiment2(self):
        cmd = "nnictl create --port 57191 --config /Users/sunwu/SW-Research/p1_ch05/experiments/configs/DEBUG_1675858846692219000_DID-A07AC296-DE40-3A7C-8DF3-91F642CC14D0_DAT-IOPSCOMPETITION_NTR-RANDOM_MTN-4_MTC-1_RET-0_SED-0_DSM-RANDOM_DSR-1.0_MON-DEFAULT_MEF-MAIN_SK.yaml --debug "
        # create_nni_experiment(cmd)
        create_nni_experiment(cmd)

    def test_add_parameters(self):
        ne = NNIExpGenerator(_mp_hpy=2, model="VAE")
        print(ne._generate_model_args())
        assert ne._generate_model_args().find("--hpy 2") > -1
