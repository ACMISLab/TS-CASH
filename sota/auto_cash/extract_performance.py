from tshpo.automl_libs import *

metircs = AnaHelper.get_all_metrics()
if __name__ == '__main__':
    df_baseline = AnaHelper.load_csv_file("c00_baseline_n500_madelon_original_20241029_1434.csv.gz")
    print(df_baseline)
