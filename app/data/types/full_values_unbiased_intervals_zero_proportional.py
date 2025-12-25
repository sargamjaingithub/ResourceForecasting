import logging
from bisect import bisect

import numpy as np
import torch

from app.data.types.full_values_unbiased_intervals import DataTypeFullValuesUnbiasedIntervals

logger = logging.getLogger(__name__)


class DataTypeFullValuesUnbiasedIntervalsZeroProportional(DataTypeFullValuesUnbiasedIntervals):

    def __init__(self, config, *args):
        super().__init__(config, *args)
        reduction_magnitude = config['reduction_magnitude']
        self.proportion = torch.arange(1, 0, reduction_magnitude / len(self.intervals), dtype=torch.float)

    def collate_fn(
            self,
            samples,
            time_series_ids,
            time_series_values_values_list,
            time_series_target_values_list
    ):
        values = torch.zeros((len(samples), time_series_values_values_list[0].shape[0], self.lag_size), dtype=torch.float)  # TODO does not work if list empty
        real_target = torch.zeros(len(samples), dtype=torch.long)
        weighted_target = torch.zeros(len(samples), len(self.intervals), dtype=torch.float)

        ids = []
        for index, sample in enumerate(samples):
            time_series_index, time_position = sample[0][0].item(), sample[0][1].item()
            ids.append((time_series_ids[time_series_index], time_position))
            values[index] = torch.from_numpy(time_series_values_values_list[time_series_index][:, (time_position - self.lag_size):time_position])

            max_target_value = np.max(time_series_target_values_list[time_series_index][time_position:(time_position + self.prediction_scope)])
            interval_index = bisect(self.intervals, max_target_value)

            real_target[index] = interval_index
            weighted_target[index, interval_index:(interval_index + self.proportion.shape[0])] = self.proportion[:(len(self.intervals) - interval_index)]

        return ids, values, (real_target, weighted_target)

    def to_device(self, values, target):
        real_target, weighted_target = target
        values = values.to(self.device)
        real_target = real_target.to(self.device)
        weighted_target = weighted_target.to(self.device)

        return values, (real_target, weighted_target)
