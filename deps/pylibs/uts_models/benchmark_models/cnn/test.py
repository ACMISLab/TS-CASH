import unittest

from pylibs.uts_models.benchmark_models.cnn.cnn import CNNModel
from pylibs.uts_models.benchmark_models.lstm.lstm import LSTMModel
from pylibs.uts_dataset.dataset_loader import DatasetLoader


class TestLSTM(unittest.TestCase):
    def test_transform(self):
        data = DatasetLoader.debug_dataset()
        print("source: \n", data)
        lm = CNNModel(slidingwindow=10)
        x, y = lm.create_dataset(data, 9, 1)
        print("x:\n", x)
        print("y:\n", y)
        """output:
        
source: 
 [[  1  11  21  31  41  51  61  71  81  91]
 [101 111 121 131 141 151 161 171 181 191]
 [201 211 221 231 241 251 261 271 281 291]
 [301 311 321 331 341 351 361 371 381 391]
 [401 411 421 431 441 451 461 471 481 491]
 [501 511 521 531 541 551 561 571 581 591]
 [601 611 621 631 641 651 661 671 681 691]
 [701 711 721 731 741 751 761 771 781 791]
 [801 811 821 831 841 851 861 871 881 891]
 [901 911 921 931 941 951 961 971 981 991]]
x:
 [[[ 1]
  [11]
  [21]
  [31]
  [41]
  [51]
  [61]
  [71]
  [81]]]
y:
 [[[ 91]
  [101]
  [111]
  [121]
  [131]
  [141]
  [151]
  [161]
  [171]
  [181]
  [191]
  [201]
  [211]
  [221]
  [231]
  [241]
  [251]
  [261]
  [271]
  [281]
  [291]
  [301]
  [311]
  [321]
  [331]
  [341]
  [351]
  [361]
  [371]
  [381]
  [391]
  [401]
  [411]
  [421]
  [431]
  [441]
  [451]
  [461]
  [471]
  [481]
  [491]
  [501]
  [511]
  [521]
  [531]
  [541]
  [551]
  [561]
  [571]
  [581]
  [591]
  [601]
  [611]
  [621]
  [631]
  [641]
  [651]
  [661]
  [671]
  [681]
  [691]
  [701]
  [711]
  [721]
  [731]
  [741]
  [751]
  [761]
  [771]
  [781]
  [791]
  [801]
  [811]
  [821]
  [831]
  [841]
  [851]
  [861]
  [871]
  [881]
  [891]
  [901]
  [911]
  [921]
  [931]
  [941]
  [951]
  [961]
  [971]
  [981]
  [991]]]
  """

    def test_2(self):
        window_size = 100
        dl = DatasetLoader("NAB", 1, window_size=window_size, is_include_anomaly_window=False)
        train_x, train_y = dl.get_sliding_windows()
        print(train_x)
        lm = LSTMModel(slidingwindow=window_size)
        lm.fit(train_x)
