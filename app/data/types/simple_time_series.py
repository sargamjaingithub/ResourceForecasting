import logging
from functools import partial
from typing import List

import numpy as np
from torch import LongTensor
from torch.utils.data import DataLoader, TensorDataset

from app.auxiliary_files.other_methods.util_functions import timeit
from app.data.data_type_abstract import DataTypeAbstract

logger = logging.getLogger(__name__)


class DataTypeSimpleTimeSeries(DataTypeAbstract):

    def __init__(self, config, *args):
        super().__init__(config, *args)
        self.shuffle = config['shuffle'] if 'shuffle' in config else True
        self.data_loader_args = config['data_loader'] if 'data_loader' in config else {}
        self.univariate = config['univariate'] if 'univariate' in config else True

    def create_data_loader(
            self,
            time_series_ids: List[str],
            time_series_lengths: List[int],
            is_train: bool
    ) -> DataLoader:
        indexes = list(range(len(time_series_ids)))
        logger.info(f'Total possible values {len(indexes)}')
        _, _, target_feature_index = self.data_source.get_target_feature_index()

        # load all time series in memory
        _, _, time_series_list, _, _ = self.data_source.load_time_series_list(
            time_series_ids,
            times=False,
            init=False
        )

        if self.univariate:
            time_series_list = [time_series[target_feature_index] for time_series in time_series_list]

        return DataLoader(
            dataset=TensorDataset(LongTensor(indexes)),
            shuffle=(is_train and self.shuffle),
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=partial(
                self.collate_fn,
                time_series_ids=time_series_ids,
                time_series_list=time_series_list
            ),
            **self.data_loader_args
        )

    @timeit
    def collate_fn(
            self,
            samples,
            time_series_ids,
            time_series_list
    ):
        ids = []
        time_series_batch_list = []
        for index, sample in enumerate(samples):
            time_series_index = sample[0].item()
            ids.append(time_series_ids[time_series_index])
            time_series_batch_list.append(time_series_list[time_series_index])

        return ids, time_series_batch_list

    @timeit
    def get_final_predictions(self, all_predictions: np.ndarray):
        _, _, time_series_initial_values_list, _, _ = self.data_source.load_time_series_list(all_predictions[0])
        target_feature_index_global, target_feature_index_values, _ = self.data_source.get_target_feature_index()
        time_series_initial_values_list = [initial_values[target_feature_index_global] for initial_values in time_series_initial_values_list]
        return all_predictions[0], time_series_initial_values_list, all_predictions[1]
