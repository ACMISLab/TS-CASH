from tshpo.lib_class import HPTrimHelper
from tshpo.tshpo_common import TSHPOCommon

if __name__ == '__main__':
    HPTrimHelper.generate_outputs(
        TSHPOCommon.get_result_data("c09_select_optimal_alg_v2_original_20241031_1352.csv.gz"))
