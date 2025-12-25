import logging
import random
from datetime import datetime
from functools import partial
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import LongTensor
from torch.utils.data import DataLoader, TensorDataset

from app.data.types.full_values_unbiased import DataTypeFullValuesUnbiased
from app.models.networks.types.informer.auxiliary import time_features

logger = logging.getLogger(__name__)


class DataTypeFullValuesUnbiasedInformer(DataTypeFullValuesUnbiased):

    def __init__(self, config, *args):
        super().__init__(config, *args)
        self.transformer_label_len = config['transformer_label_len']
        self.time_in_microseconds = False if 'time_in_microseconds' not in config else config['time_in_microseconds']
        assert(self.transformer_label_len < self.lag_size)

    def create_data_loader(
            self,
            time_series_ids: List[str],
            time_series_lengths: List[int],
            is_train: bool
    ) -> DataLoader:
        # create data loader with the indexes of all possible values
        # we do not create a new array to not waste memory (the transformation is done in collate_fn)
        # the indexes are in the way -> (time_series_index, time_position)
        maximum_per_time_series = min(time_series_lengths) - self.lag_size - self.prediction_size
        indexes = []
        for time_series_id_index in range(len(time_series_ids)):
            time_series_length = time_series_lengths[time_series_id_index]
            possible_time_values = [(time_series_id_index, time_position) for time_position in range(self.lag_size, time_series_length - self.prediction_size + 1)]
            if is_train:
                possible_time_values = random.sample(possible_time_values, maximum_per_time_series)
            indexes += possible_time_values
        logger.info(f'Total possible values {len(indexes)}')

        # load all time series in memory
        _, time_series_times_list, _, time_series_values_values_list, time_series_target_values_list = self.data_source.load_time_series_list(
            time_series_ids,
            times=False,
            init=False
        )

        # encode time
        for time_series_times_index, time_series_times in enumerate(time_series_times_list):
            if self.time_in_microseconds:
                time_series_times /= 1000000
            time_series_times = [datetime.fromtimestamp(time_series_times[timestamp_index]) for timestamp_index in range(time_series_times.shape[0])]
            encoded_time_series_times = pd.DataFrame({'date': time_series_times})
            encoded_time_series_times = time_features(encoded_time_series_times, timeenc=1, freq='T')
            time_series_times_list[time_series_times_index] = encoded_time_series_times

        return DataLoader(
            dataset=TensorDataset(LongTensor(indexes)),
            shuffle=(is_train and self.shuffle),
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=partial(
                self.collate_fn,
                time_series_ids=time_series_ids,
                time_series_times_list=time_series_times_list,
                time_series_values_values_list=time_series_values_values_list,
                time_series_target_values_list=time_series_target_values_list
            ),
            **self.data_loader_args
        )

    def collate_fn(
            self,
            samples,
            time_series_ids,
            time_series_times_list,
            time_series_values_values_list,
            time_series_target_values_list
    ):
        values_seq_x = torch.zeros((len(samples), time_series_values_values_list[0].shape[0], self.lag_size), dtype=torch.float)  # TODO does not work if list empty
        values_seq_y = torch.zeros((len(samples), time_series_values_values_list[0].shape[0], self.transformer_label_len + self.prediction_size), dtype=torch.float)  # TODO does not work if list empty
        values_seq_x_mark = torch.zeros((len(samples), self.lag_size, 5), dtype=torch.float)
        values_seq_y_mark = torch.zeros((len(samples), self.transformer_label_len + self.prediction_size, 5), dtype=torch.float)

        target = torch.zeros(len(samples), self.prediction_size, dtype=torch.float)  # TODO only works with one feature as target

        ids = []
        for index, sample in enumerate(samples):
            time_series_index, time_position = sample[0][0].item(), sample[0][1].item()
            ids.append((time_series_ids[time_series_index], time_position))

            seq_x = time_series_values_values_list[time_series_index][:, (time_position - self.lag_size):time_position]
            seq_y = np.zeros((seq_x.shape[0], self.transformer_label_len + self.prediction_size))  # zero padding
            seq_y[:, :-1] = time_series_values_values_list[time_series_index][:, (time_position - self.transformer_label_len):time_position]
            values_seq_x[index] = torch.from_numpy(seq_x)
            values_seq_y[index] = torch.from_numpy(seq_y)

            seq_x_mark = time_series_times_list[time_series_index][(time_position - self.lag_size):time_position]
            seq_y_mark = time_series_times_list[time_series_index][(time_position - self.transformer_label_len):(time_position + self.prediction_size)]
            values_seq_x_mark[index] = torch.from_numpy(seq_x_mark)
            values_seq_y_mark[index] = torch.from_numpy(seq_y_mark)

            target[index] = torch.from_numpy(time_series_target_values_list[time_series_index][time_position:(time_position + self.prediction_size)])

        return ids, (values_seq_x, values_seq_x_mark, values_seq_y, values_seq_y_mark), target

    def to_device(self, values, target):
        values_seq_x, values_seq_x_mark, values_seq_y, values_seq_y_mark = values

        values_seq_x = values_seq_x.to(self.device)
        values_seq_y = values_seq_y.to(self.device)
        values_seq_x_mark = values_seq_x_mark.to(self.device)
        values_seq_y_mark = values_seq_y_mark.to(self.device)
        target = target.to(self.device)

        return (values_seq_x, values_seq_x_mark, values_seq_y, values_seq_y_mark), target
