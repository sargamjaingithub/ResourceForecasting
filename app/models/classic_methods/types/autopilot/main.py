# Modification of the Autopilot implementation from Claudia Herron Mulet
import numpy as np

from app.models.classic_methods.method_interface import MethodInterface
from app.models.classic_methods.types.autopilot.auxiliary import get_aggregated_signal, get_granular_rec, get_peak_rec, \
    get_wavg_rec, get_jp_rec, get_ml_rec


class AutopilotWindowRecommenderPeak(MethodInterface):
    def __init__(
            self,
            device: str,
            min_cpu: float,
            max_cpu: float,
            mean_cpu: float
    ):
        super(AutopilotWindowRecommenderPeak, self).__init__(device)

        num_buckets = 400
        self.resolution = 20
        self.n = 12

        self.mean_cpu = mean_cpu
        bucket_size_cpu = (max_cpu - abs(min_cpu)) / num_buckets
        self.cpu_buckets = np.linspace(min_cpu + bucket_size_cpu, max_cpu, num=num_buckets)
        self.cpu_bins = np.linspace(min_cpu, max_cpu, num=num_buckets + 1)

    def train(self, time_series_list, y=None):
        return

    def compute_loss(self, time_series_list, y):
        return

    def predict(self, time_series_list):
        output = []
        for time_series in time_series_list:
            aggregated_signal = get_aggregated_signal(time_series, self.resolution, self.cpu_bins)
            output.append(get_granular_rec(get_peak_rec(aggregated_signal, self.cpu_buckets, self.n), self.mean_cpu, self.resolution)[:time_series.shape[0]])
        return output


class AutopilotWindowRecommenderWeighted(MethodInterface):
    def __init__(
            self,
            device: str,
            min_cpu: float,
            max_cpu: float,
            mean_cpu: float
    ):
        super(AutopilotWindowRecommenderWeighted, self).__init__(device)

        self.num_buckets = 400
        self.resolution = 20
        self.n = 12
        self.half_life_cpu = 12

        self.mean_cpu = mean_cpu
        bucket_size_cpu = (max_cpu - abs(min_cpu)) / self.num_buckets
        self.cpu_buckets = np.linspace(min_cpu + bucket_size_cpu, max_cpu, num=self.num_buckets)
        self.cpu_bins = np.linspace(min_cpu, max_cpu, num=self.num_buckets + 1)

    def train(self, time_series_list, y=None):
        return

    def compute_loss(self, time_series_list, y):
        return

    def predict(self, time_series_list):
        output = []
        for time_series in time_series_list:
            aggregated_signal = get_aggregated_signal(time_series, self.resolution, self.cpu_bins)
            output.append(get_granular_rec(get_wavg_rec(aggregated_signal, self.cpu_buckets, self.half_life_cpu, self.num_buckets, self.n), self.mean_cpu, self.resolution)[:time_series.shape[0]])
        return output


class AutopilotWindowRecommenderPercentile(MethodInterface):
    def __init__(
            self,
            device: str,
            min_cpu: float,
            max_cpu: float,
            mean_cpu: float
    ):
        super(AutopilotWindowRecommenderPercentile, self).__init__(device)

        self.num_buckets = 400
        self.resolution = 20
        self.n = 12
        self.j_cpu = 95
        self.half_life_cpu = 12

        self.mean_cpu = mean_cpu
        bucket_size_cpu = (max_cpu - abs(min_cpu)) / self.num_buckets
        self.cpu_buckets = np.linspace(min_cpu + bucket_size_cpu, max_cpu, num=self.num_buckets)
        self.cpu_bins = np.linspace(min_cpu, max_cpu, num=self.num_buckets + 1)

    def train(self, time_series_list, y=None):
        return

    def compute_loss(self, time_series_list, y):
        return

    def predict(self, time_series_list):
        output = []
        for time_series in time_series_list:
            aggregated_signal = get_aggregated_signal(time_series, self.resolution, self.cpu_bins)
            output.append(get_granular_rec(get_jp_rec(aggregated_signal, self.cpu_buckets, self.half_life_cpu, self.j_cpu, self.num_buckets), self.mean_cpu, self.resolution)[:time_series.shape[0]])
        return output


class AutopilotMLRecommender(MethodInterface):
    def __init__(
            self,
            device: str,
            lag_size: int,
            min_cpu: float,
            max_cpu: float,
            mean_cpu: float
    ):
        super(AutopilotMLRecommender, self).__init__(device)

        self.num_buckets = 400
        self.resolution = 20
        self.n = 12
        self.j_cpu = 95
        self.half_life_cpu = 12

        self.mean_cpu = mean_cpu
        bucket_size_cpu = (max_cpu - abs(min_cpu)) / self.num_buckets
        self.cpu_buckets = np.linspace(min_cpu + bucket_size_cpu, max_cpu, num=self.num_buckets)
        self.cpu_bins = np.linspace(min_cpu, max_cpu, num=self.num_buckets + 1)

        self.w_o, self.w_u, self.w_delta_L, self.w_delta_m = 0.5, 0.25, 0.1, 0.1
        self.d = 0.75
        self.dm_min, self.dm_max, self.d_n_step = 0.1, 1.0, 10
        self.Mm_min, self.Mm_max, self.M_n_step = 0, 1, 2

    def train(self, train_data_loader, val_data_loader):
        return

    def compute_loss(self, data_loader):
        return None

    def predict(self, data_loader):
        output_time_series_ids = []
        output_time_series_values = []
        for index_batch, (ids, time_series_batch_list) in enumerate(data_loader):
            for index_time_series, time_series in enumerate(time_series_batch_list):
                aggregated_signal = get_aggregated_signal(time_series, self.resolution, self.cpu_bins)
                forecasted_request = get_granular_rec(
                    get_ml_rec(
                        aggregated_signal,
                        self.cpu_buckets,
                        self.dm_min,
                        self.dm_max,
                        self.d_n_step,
                        self.Mm_min,
                        self.Mm_max,
                        self.M_n_step,
                        self.w_delta_m,
                        self.w_delta_L,
                        self.w_o,
                        self.w_u,
                        self.d
                    ),
                    self.mean_cpu,
                    self.resolution
                )
                output_time_series_values.append(forecasted_request)
            output_time_series_ids += ids
        return output_time_series_ids, output_time_series_values