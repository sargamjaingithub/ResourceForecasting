from app.models.classic_methods.method_interface import MethodInterface
from app.models.classic_methods.types.theta_scan.theta_scan import ThetaScan as PackageThetaScan


class ThetaScan(MethodInterface):

    # ---> Main classic_methods

    def __init__(self, device: str, lag_size: int):
        super().__init__(device)
        self.device = device
        self.lag_size = lag_size

        self.theta_scan = PackageThetaScan()

    def train(self, train_data_loader, val_data_loader):
        return

    def compute_loss(self, data_loader):
        return None

    def predict(self, data_loader):
        output_time_series_ids = []
        output_time_series_values = []
        for index_batch, (ids, time_series_batch_list) in enumerate(data_loader):
            for index_time_series, time_series in enumerate(time_series_batch_list):
                forecasted_request, _ = self.theta_scan.dynamic_recommend(time_series, observation_window=self.lag_size)
                output_time_series_values.append(forecasted_request)
            output_time_series_ids += ids
        return output_time_series_ids, output_time_series_values
