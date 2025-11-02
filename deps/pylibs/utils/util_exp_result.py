import numpy as np
import pandas as pd


class ExpResult:

    @staticmethod
    def format_float(val, decimal=2):
        return np.round(val, decimals=decimal)

    @staticmethod
    def load_merge_metrics_bz2(filename):
        _data = pd.read_pickle(filename)
        _data = _data.reset_index(drop=True)
        return _data

    @staticmethod
    def format_perf_mean_and_std(mean_val: float, std_val: float, scale: float = 1, decimal: int = 2,
                                 prefix_sign=r'\pm'):
        mean_val = mean_val * scale
        std_val = std_val * scale

        if decimal == 0:
            return f"${int(mean_val)}{prefix_sign}{int(std_val)}$"
        else:
            # f"%.{decimal}f".format(mean_val)
            return f"${mean_val:.{decimal}f}{prefix_sign}{std_val:.{decimal}f}$"
        # if decimal == 0:
        #     return f"{int(mean_val)}({prefix_sign}{int(std_val)})"
        # else:
        #     f"%.{decimal}f".format(mean_val)
        #     return f"{mean_val:.{decimal}f}({prefix_sign}{std_val:.{decimal}f})"
        #

class ER(ExpResult):
    pass
