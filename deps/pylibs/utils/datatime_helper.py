import pprint
import time
from dataclasses import dataclass

from numpy.testing import assert_almost_equal
import logging

log = logging.getLogger(__name__)

@dataclass
class MetricsCollector:
    """
    A class to help recording the running time of an application.

    When app is start_or_restart, call start_or_restart() and the timestamp of current is recorded in self._app_start_dt

    If the app mark_as_finished, call end() and the timestamp of current is recorded in self._app_end_dt, etc.

    """

    def __post_init__(self):
        self._evaluate_start_timestamp: float = -1
        self._evaluate_end_timestamp: float = -1
        self._train_end_timestamp: float = -1
        self._train_start_timestamp: float = -1
        # The datatime of app to -1
        self._app_start_timestamp: float = -1
        # The datatime of app start_or_restart to run
        self._app_end_timestamp: float = -1

    def start(self):
        """
        Record the timestamp of the app start_or_restart

        Returns
        -------

        """
        self._app_start_timestamp = self.get_current_timestamp()
        log.info(f"App start_or_restart at {self._app_start_timestamp}")

    def end(self):
        """
        Record the timestamp as soon as the app finishes (mark_as_finished)

        Returns
        -------

        """
        self._app_end_timestamp = self.get_current_timestamp()
        log.info(f"App end at {self._app_end_timestamp}")

    def train_start(self):
        self._train_start_timestamp = self.get_current_timestamp()
        log.info(f"Train start_or_restart at {self._train_start_timestamp}")

    def train_end(self):
        assert self._train_start_timestamp != -1, "Train start must be called before call train_end"
        self._train_end_timestamp = self.get_current_timestamp()
        log.info(
            f"Train end at {self._train_start_timestamp}, elapsed: "
            f"{self.get_elapse_train()} s")

    def evaluate_end(self):
        self._evaluate_end_timestamp = self.get_current_timestamp()

    def evaluate_start(self):
        self._evaluate_start_timestamp = self.get_current_timestamp()

    @classmethod
    def get_current_timestamp(cls):
        return time.time()

    def collect_metrics(self):
        return {
            "time_app_start": self._app_start_timestamp,
            "time_app_end": self._app_end_timestamp,
            "time_train_start": self._train_start_timestamp,
            'time_train_end': self._train_end_timestamp,
            'time_eval_start': self._evaluate_start_timestamp,
            'time_eval_end': self._evaluate_end_timestamp,
            'elapsed_train_seconds': self.get_elapse_train(),
            'elapsed_evaluate_seconds': self.get_elapse_test(),
        }

    def get_elapse_train(self):
        return self._train_end_timestamp - self._train_start_timestamp

    def get_elapse_test(self):
        return self._evaluate_end_timestamp - self._evaluate_start_timestamp

    @property
    def app_start_timestamp(self):
        return self._app_start_timestamp

    @property
    def app_end_timestamp(self):
        return self._app_end_timestamp

    @property
    def train_start_timestamp(self):
        return self._train_start_timestamp

    @property
    def train_end_timestamp(self):
        return self._train_end_timestamp

    @property
    def evaluate_start_timestamp(self):
        return self._evaluate_start_timestamp

    @property
    def evaluate_end_timestamp(self):
        return self._evaluate_end_timestamp

    def __str__(self):
        return str(self.__dict__)


class DTH(MetricsCollector):
    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    dt = MetricsCollector()
    dt.train_start()
    # 经过2.11秒
    sleep_time = 2.13
    print(f"except {sleep_time} s")
    time.sleep(sleep_time)
    dt.train_end()

    pprint.pprint(dt.collect_metrics())
    assert_almost_equal(sleep_time, dt.collect_metrics()['elapsed_train_seconds'], decimal=2)
