import logging
from time import time

import numpy as np
import torch
import torch.nn as nn

from app.auxiliary_files.other_methods.visualize import compare_multiple_lines_rectangles_color
from app.models.model_type_interface import ModelTypeInterface

# from captum.attr import DeepLiftShap

logger = logging.getLogger(__name__)


# define the default workflow for neural networks training
class ModelTypeXAI(ModelTypeInterface):

    def __init__(self, config: dict, data_model, network: nn.Module, output_path: str, device: str):
        super().__init__(config, data_model, network, output_path, device)

        # save input params
        self.config = config
        self.data_model = data_model
        self.output_path = output_path
        self.device = device

    # ---> Main classic_methods

    def show_info(self):
        pass

    def train_test(self):
        raise NotImplementedError('Method not implemented')

    def inference(self, pretrained_model):
        init_time = time()
        dl = DeepLiftShap(pretrained_model.network)
        test_data_loader = self.data_model.get_test_data_loader()

        pretrained_model.network.eval()

        # 1st -> nothing: 2000; spikes:

        positions_to_show = {1234, 1436, 2000}
        # """
        time_series_attributions = {}
        # """
        time_series_output = {}

        for index, (ids, values, target) in enumerate(test_data_loader, 0):  # iterate data loader
            print('!!!!!!!!!!!!!!!!!!!!', index)
            (real_target, weighted_target) = target

            values = values.to(self.device)
            real_target = real_target.to(self.device)
            weighted_target = weighted_target.to(self.device)

            # """
            baseline_dist = torch.randn(values.shape) * 0.001
            attributions, delta = dl.attribute(values, baseline_dist, target=real_target, return_convergence_delta=True)
            deltas_per_example = torch.mean(delta.reshape(values.shape[0], -1), dim=1)

            attributions = attributions.detach().numpy()
            deltas_per_example = deltas_per_example.detach().numpy()
            # """

            output = pretrained_model.network.predict(values).detach().numpy()

            for attribution_index, (time_series_id, position_index) in enumerate(ids):
                # attributions
                # """
                if position_index in positions_to_show:
                    if time_series_id not in time_series_attributions:
                        time_series_attributions[time_series_id] = {}
                    time_series_attributions[time_series_id][position_index] = attributions[attribution_index]
                # """

                # output
                if time_series_id not in time_series_output:
                    time_series_output[time_series_id] = {}
                time_series_output[time_series_id][position_index] = output[attribution_index]

        # load time series
        time_series_ids_list, time_series_times_list, time_series_initial_values_list, time_series_values_values_list, time_series_target_values_list = self.data_model.data_source.load_time_series_list(list(time_series_output.keys()))

        for time_series_index, time_series_id in enumerate(time_series_ids_list):
            # generate charts
            initial_values_time_series = time_series_initial_values_list[time_series_index]
            values_values_time_series = time_series_values_values_list[time_series_index]
            target_values_time_series = time_series_target_values_list[time_series_index]

            # complete
            target_feature_index = self.data_model.data_source.get_target_feature_index()
            lines = [
                (
                    initial_values_time_series[target_feature_index, :],
                    np.arange(initial_values_time_series.shape[1]),
                    "#c0e0de",
                    3,
                    'raw'
                ),
                (
                    values_values_time_series[target_feature_index, :],
                    np.arange(values_values_time_series.shape[1]),
                    "gray",
                    3,
                    'values series'
                ),
                (
                    target_values_time_series,
                    np.arange(target_values_time_series.shape[0]),
                    "#3c474b",
                    3,
                    'target series'
                )
            ]

            rectangles = []
            init_position_interval = self.data_model.lag_size
            current_interval = np.argmax(time_series_output[time_series_id][self.data_model.lag_size])
            for (prediction_time_position, prediction_values) in time_series_output[time_series_id].items():
                interval_index = np.argmax(prediction_values)
                if interval_index != current_interval:
                    height = self.data_model.intervals[current_interval + 1] - self.data_model.intervals[current_interval] \
                        if (current_interval + 1) < len(self.data_model.intervals) \
                        else self.data_model.intervals[current_interval] - self.data_model.intervals[current_interval - 1]
                    rectangles.append(
                        (
                            init_position_interval,  # x
                            self.data_model.intervals[current_interval],  # y
                            # if not self.log_transformation else np.exp(self.intervals[interval_index]),  # y
                            prediction_time_position - init_position_interval,  # width
                            height,  # if not self.log_transformation else np.exp(height),  # height
                            0,  # angle
                            {'facecolor': (1, 0, 0, 0.2)}  # args
                        )
                    )
                    init_position_interval = prediction_time_position
                    current_interval = interval_index
            height = self.data_model.intervals[interval_index + 1] - self.data_model.intervals[interval_index] \
                if (interval_index + 1) < len(self.data_model.intervals) \
                else self.data_model.intervals[interval_index] - self.data_model.intervals[interval_index - 1]
            rectangles.append(
                (
                    init_position_interval,  # x
                    self.data_model.intervals[interval_index],  # y
                    prediction_time_position - init_position_interval,  # width
                    height,  # height
                    0,  # angle
                    {'facecolor': (1, 0, 0, 0.2)}  # args
                )
            )


            compare_multiple_lines_rectangles_color(
                False,
                lines,
                rectangles,
                'y',
                'time',
                f'',
                f'{self.output_path}/xai_{time_series_id}_complete',
                legend=True
            )

            # partial
            for position_index in positions_to_show:
                start_point = position_index - self.data_model.lag_size
                end_point = position_index
                lines = [
                    (
                        initial_values_time_series[target_feature_index, start_point:(end_point + 1)],
                        np.arange(position_index - self.data_model.lag_size, position_index + 1),
                        "#c0e0de",
                        3,
                        'raw'
                    ),
                    (
                        values_values_time_series[target_feature_index, start_point:(end_point + 1)],
                        np.arange(position_index - self.data_model.lag_size, position_index + 1),
                        "gray",
                        3,
                        'values series'
                    ),
                    (
                        target_values_time_series[start_point:(end_point + 1)],
                        np.arange(position_index - self.data_model.lag_size, position_index + 1),
                        "#3c474b",
                        3,
                        'target series'
                    )
                ]

                rectangles = []
                # """
                for feature_index in range(len(self.data_model.data_source.get_features_labels())):
                    lines.append(
                        (
                            time_series_attributions[time_series_id][position_index][feature_index],
                            np.arange(position_index - self.data_model.lag_size, position_index),
                            None,
                            3,
                            f'attributed importance feature index {self.data_model.data_source.get_features_labels()[feature_index]}'
                        )
                    )

                interval_index = np.argmax(time_series_output[time_series_id][position_index])
                height = self.data_model.intervals[interval_index + 1] - self.data_model.intervals[interval_index] \
                    if (interval_index + 1) < len(self.data_model.intervals) \
                    else self.data_model.intervals[interval_index] - self.data_model.intervals[interval_index - 1]
                rectangles = [
                    (
                        position_index,  # x
                        self.data_model.intervals[interval_index],  # y
                        2,  # width
                        height,  # if not self.log_transformation else np.exp(height),  # height
                        0,  # angle
                        {'facecolor': (1, 0, 0, 0.2)}  # args
                    )
                ]
                # """

                compare_multiple_lines_rectangles_color(
                    False,
                    lines,
                    rectangles,
                    'y',
                    'time',
                    f'',
                    f'{self.output_path}/xai_{time_series_id}_partial_{position_index}',
                    legend=True
                )

        return time() - init_time





    def save_results(self, visualize):
        pass

    def save_model(self):
        pass

    # --> Auxiliary classic_methods
