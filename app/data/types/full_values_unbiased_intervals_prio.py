import logging
import random
from bisect import bisect
from functools import partial
from typing import List

import numpy as np
import torch
from torch import LongTensor
from torch.utils.data import DataLoader, TensorDataset

from app.data.types.full_values_unbiased_intervals import DataTypeFullValuesUnbiasedIntervals

logger = logging.getLogger(__name__)


class DataTypeFullValuesUnbiasedIntervalsPrio(DataTypeFullValuesUnbiasedIntervals):

    def __init__(self, config, *args):
        super().__init__(config, *args)
        self.shuffle = config['shuffle'] if 'shuffle' in config else True
        self.data_loader_args = config['data_loader'] if 'data_loader' in config else {}
        self.intervals = config['intervals']
        self.log_transformation = config['log_transformation'] if 'log_transformation' in config else False
        self.prediction_scope = config['prediction_scope'] if 'prediction_scope' in config else 1

        if self.log_transformation:
            self.intervals = list(np.log(np.asarray(self.intervals) + 0.00000000001))

    def get_prediction_size(self) -> int:
        return len(self.intervals)

    def create_data_loader(
            self,
            time_series_ids: List[str],
            time_series_lengths: List[int],
            is_train: bool
    ) -> DataLoader:
        WINDOW_SIZE = 100

        # load all time series in memory
        _, _, _, time_series_values_values_list, time_series_target_values_list = self.data_source.load_time_series_list(
            time_series_ids,
            times=False,
            init=False
        )
        _, target_feature_index_values, target_feature_index_target = self.data_source.get_target_feature_index()

        # create data loader with the indexes of all possible values
        # we do not create a new array to not waste memory (the transformation is done in collate_fn)
        # the indexes are in the way -> (time_series_index, time_position)
        maximum_per_time_series = min(time_series_lengths) - self.lag_size - self.prediction_size
        indexes = []
        for time_series_id_index in range(len(time_series_ids)):
            time_series_length = time_series_lengths[time_series_id_index]

            if is_train:
                time_series_values = time_series_target_values_list[time_series_id_index]  # TODO check index coincides
                # time_series_values = time_series_values[target_feature_index_target]

                possible_time_values_default = []
                possible_time_values_spikes = []
                last_spike_position = None
                for time_position in range(self.lag_size, time_series_length - self.prediction_size + 1):
                    value = time_series_values[time_position]
                    if last_spike_position:
                        window_size = min(WINDOW_SIZE, time_position - last_spike_position)
                    else:
                        window_size = WINDOW_SIZE
                    lag_mean = np.mean(time_series_values[(time_position - window_size):time_position])
                    lag_std = np.std(time_series_values[(time_position - window_size):time_position])
                    if (value - lag_mean) > 1 * lag_std:
                        last_spike_position = time_position
                        possible_time_values_spikes.append((time_series_id_index, time_position))
                    else:
                        possible_time_values_default.append((time_series_id_index, time_position))

                '''
                compare_multiple_lines_points_color_enhanced(
                    False,
                    [
                        (
                            time_series_target_values_list[time_series_id_index][self.lag_size:(self.lag_size * 5)],
                            np.arange(self.lag_size, self.lag_size * 5),
                            "#3c474b",
                            9,
                            'target series'
                        )
                    ],
                    [
                        (time_series_target_values_list[time_series_id_index][time_position], time_position)
                        for (time_series_id_index, time_position) in possible_time_values_spikes
                        if self.lag_size < time_position < (self.lag_size * 5)
                    ],
                    'CPU consumption',
                    'time',
                    f'{self.output_path}/{time_series_id_index}_complete'
                )
                '''

                if maximum_per_time_series < len(possible_time_values_spikes):
                    possible_time_values = [(time_series_id_index, time_position) for time_position in
                                            range(self.lag_size, time_series_length - self.prediction_size + 1)]
                    possible_time_values = random.sample(possible_time_values, maximum_per_time_series)
                    indexes += possible_time_values
                else:
                    indexes += possible_time_values_spikes
                    indexes += random.sample(possible_time_values_default, maximum_per_time_series - len(possible_time_values_spikes))
            else:
                possible_time_values = [(time_series_id_index, time_position) for time_position in
                                        range(self.lag_size, time_series_length - self.prediction_size + 1)]
                indexes += possible_time_values

        logger.info(f'Total possible values {len(indexes)}')

        return DataLoader(
            dataset=TensorDataset(LongTensor(indexes)),
            shuffle=(is_train and self.shuffle),
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=partial(
                self.collate_fn,
                time_series_ids=time_series_ids,
                time_series_values_values_list=time_series_values_values_list,
                time_series_target_values_list=time_series_target_values_list
            ),
            **self.data_loader_args
        )

    def collate_fn(
            self,
            samples,
            time_series_ids,
            time_series_values_values_list,
            time_series_target_values_list
    ):
        values = torch.zeros((len(samples), time_series_values_values_list[0].shape[0], self.lag_size))  # TODO does not work if list empty
        target = torch.zeros(len(samples), dtype=torch.long)

        ids = []
        for index, sample in enumerate(samples):
            time_series_index, time_position = sample[0][0].item(), sample[0][1].item()
            ids.append((time_series_ids[time_series_index], time_position))
            values[index] = torch.from_numpy(time_series_values_values_list[time_series_index][:, (time_position - self.lag_size):time_position])
            max_target_value = np.max(time_series_target_values_list[time_series_index][time_position:(time_position + self.prediction_size)])
            interval_index = bisect(self.intervals, max_target_value)
            if interval_index >= len(self.intervals):
                raise Exception(f'Found value outside of defined intervals: {max_target_value}')
            target[index] = interval_index

        return ids, values, target
