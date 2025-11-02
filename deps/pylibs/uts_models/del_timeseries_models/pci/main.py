from hyper_speed.models.pci.algorithm import AlgorithmArgs, CustomParameters
from hyper_speed.models.pci.pci.model import PCIAnomalyDetector
from pylibs.dataset.DatasetAIOPS2018 import DatasetAIOps2018, AIOpsKPIID
from pylibs.utils.util_univariate_time_series_view import UnivariateTimeSeriesView

cp = CustomParameters()
cp.window_size = 30
pci = PCIAnomalyDetector(
    k=cp.window_size,
    p=cp.thresholding_p,
    calculate_labels=False
)

# x, y = get_test_time_series()
da = DatasetAIOps2018(kpi_id=AIOpsKPIID.D7,
                      windows_size=32,
                      is_include_anomaly_windows=True,
                      sampling_rate=1,
                      valid_rate=0.5)

train_x, train_y, valid_x, valid_y, test_x, test_y = da.get_three_splits()
y = train_y
x = train_x.reshape(-1)
anomaly_scores, label = pci.detect(x)
uv = UnivariateTimeSeriesView()
uv.plot_kpi_with_anomaly_score_row2(x, y, anomaly_scores)
