from dataclasses import dataclass
from prometheus_api_client import PrometheusConnect
import pprint


@dataclass
class Prometheus:
    # http://your_server_ip:32715/graph?g0.expr=container_cpu_usage_seconds_total%7Bpod%3D~%22php-apache.*%22%7D&g0.tab=1&g0.display_mode=lines&g0.show_exemplars=0&g0.range_input=1h
    url = "http://your_server_ip:32067"

    def __post_init__(self):
        self.prom = PrometheusConnect(url=self.url, disable_ssl=True)

    def get_all_metrics(self):
        # Get the list of all the metrics that the Prometheus host scrapes
        print(self.prom.all_metrics())

    def query(self, promQL):
        """ promQL= container_cpu_usage_seconds_total{pod=~'php-apache.*'} @"""
        metric_data = self.prom.custom_query(query=promQL)
        # pprint.pprint(metric_data)
        return metric_data


if __name__ == "__main__":
    # http://your_server_ip:32715
    print(Prometheus().query())

    # for i in range(1000):
    #   print("{}".format(i))
    #   query = 'rate(container_cpu_usage_seconds_total[5s])'
    #   Prometheus().query(query=query)
