
import nni
import numpy as np
nni.get_next_parameters()
nni.report_final_result({
    "default": np.random.random(),
    "test_auc": np.random.random()
})