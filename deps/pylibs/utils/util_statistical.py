import traceback

import numpy as np
import scipy
from pylibs.utils.util_log import get_logger
from pylibs.utils.util_numpy import enable_numpy_reproduce

log = get_logger()


def is_significant_decrease_fix_test(baseline_metrics: np.ndarray, sampled_metrics: np.ndarray, alpha: float = 0.01,
                                     return_t_and_p: bool = False):
    """
    这个函数修复了假设检验前判断数据分布的问题
    Return:
    is_sig_decrease, t, p = is_significant_decrease(baseline_metrics, sampled_metrics)  if return_t_and_p=True

    is_significant_decrease if return_t_and_p=False (default)

    Parameters
    ----------
    baseline_metrics :
    sampled_metrics :
    alpha :
    return_t_and_p :

    Returns
    -------

    """
    if (baseline_metrics == sampled_metrics).all():
        log.warning("👍 two sets is absolutely equal")
        t_, p_ = 1, 1

    # fixed distribution checking: whether the difference of the paired sets is normalized.
    elif is_normal_distribution(baseline_metrics - sampled_metrics):
        # fisrst to test whether it is equal, then to test whether (alternative hypothesis) baseline metrics is greater the optimized metrics.
        # we except the optimized metrics is not less than baseline metrics (baseline <= optimized metrics).
        # so the alternative hypothesis is: baseline metrics > optimized metrics
        # if p <= 0.01, we think that baseline metrics  is greater than baseline metrics.
        # so we expect p > 0.01

        # To test whether two sets have significant differences
        t_, p_ = scipy.stats.ttest_rel(baseline_metrics, sampled_metrics)

        # To test whether the sampled set has a significant decrease compared to the baseline.
        if p_ < alpha:
            t_, p_ = scipy.stats.ttest_rel(baseline_metrics, sampled_metrics, alternative="greater")
    else:
        # # To test whether two sets have significant differences
        wilcoxon_res = scipy.stats.wilcoxon(baseline_metrics, sampled_metrics)
        t_, p_ = wilcoxon_res.statistic, wilcoxon_res.pvalue

        if p_ < alpha:
            wilcoxon_res = scipy.stats.wilcoxon(baseline_metrics, sampled_metrics,
                                                alternative="greater")
            t_, p_ = wilcoxon_res.statistic, wilcoxon_res.pvalue
    if return_t_and_p:
        return p_ < alpha, t_, p_
    else:
        return p_ < alpha


def is_normal_distribution(data: np.ndarray,
                           alpha=0.01, check_type="shapiro",
                           return_pvalue: bool = True):
    """

    Args:
        data:
        alpha: float
            Default 0.01
        check_type:
            one of  shapiro,  normaltest
        return_pvalue: boolean

    Returns:

    """
    if data.shape[0] <= 3:
        t, p = 0, 0
    elif check_type == "shapiro":
        t, p = scipy.stats.shapiro(data)
    elif check_type == "normaltest":
        t, p = scipy.stats.normaltest(data)
    else:
        t, p = scipy.stats.shapiro(data)
    if return_pvalue:
        return p >= alpha, p
    else:
        return p >= alpha


def is_significant_decrease(baseline_metrics: np.ndarray,
                            sampled_metrics: np.ndarray, alpha: float = 0.01,
                            return_t_and_p: bool = False):
    """
    Return:
    is_sig_decrease, t, p = is_significant_decrease(baseline_metrics, sampled_metrics)  if return_t_and_p=True

    is_significant_decrease if return_t_and_p=False (default)

    Parameters
    ----------
    baseline_metrics :
    sampled_metrics :
    alpha :
    return_t_and_p :

    Returns
    -------

    """
    if (baseline_metrics == sampled_metrics).all():
        log.warning("👍 two sets is absolutely equal")
        t_, p_ = 1, 1
    elif is_normal_distribution(baseline_metrics):
        # To test whether two sets have significant differences
        t_, p_ = scipy.stats.ttest_rel(baseline_metrics, sampled_metrics)

        # To test whether the sampled set has a significant decrease compared to the baseline.
        if p_ < alpha:
            t_, p_ = scipy.stats.ttest_rel(baseline_metrics, sampled_metrics, alternative="greater")
    else:
        # # To test whether two sets have significant differences
        wilcoxon_res = scipy.stats.wilcoxon(baseline_metrics, sampled_metrics)
        t_, p_ = wilcoxon_res.statistic, wilcoxon_res.pvalue

        if p_ < alpha:
            wilcoxon_res = scipy.stats.wilcoxon(baseline_metrics, sampled_metrics,
                                                alternative="greater")
            t_, p_ = wilcoxon_res.statistic, wilcoxon_res.pvalue
    if return_t_and_p:
        return p_ < alpha, t_, np.round(p_,decimals=6)
    else:
        return p_ < alpha


class UtilHypo:
    @staticmethod
    def welchs_test(baseline_metrics, sampled_metrics):
        wilcoxon_res = scipy.stats.ttest_ind(baseline_metrics, sampled_metrics,
                                             equal_var=False)
        t_, p_ = wilcoxon_res.statistic, wilcoxon_res.pvalue
        return p_

    @staticmethod
    def wilcoxon_test(baseline_metrics, sampled_metrics):
        wilcoxon_res = scipy.stats.wilcoxon(baseline_metrics, sampled_metrics,
                                            alternative="greater")
        t_, p_ = wilcoxon_res.statistic, wilcoxon_res.pvalue
        return p_


if __name__ == '__main__':
    enable_numpy_reproduce(42)
    assert is_significant_decrease(np.ones((10,)), np.zeros((10,)) + 0.5) == True
    assert is_significant_decrease(np.ones((10,)) + 100, np.zeros((10,)) + 0.5) == True
    assert is_significant_decrease(np.random.normal(loc=1.0, size=100), np.random.normal(loc=0.5, size=100)) == True
    assert is_significant_decrease(np.asarray([1, 1, 1]), np.asarray([1, 1, 1])) == False
    assert is_normal_distribution(np.random.uniform(1, 10, 100), return_pvalue=False) == False
    assert is_normal_distribution(np.random.normal(size=100), return_pvalue=False) == True
