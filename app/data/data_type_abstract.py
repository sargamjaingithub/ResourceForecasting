import logging
from typing import List

from torch.utils.data import DataLoader

from app.auxiliary_files.other_methods.util_functions import print_pretty_json
from app.data.data_source_abstract import DataSourceAbstract
from app.data.data_type_interface import DataTypeInterface

logger = logging.getLogger(__name__)


class DataTypeAbstract(DataTypeInterface):

    def __init__(
            self,
            config: dict,
            data_source: DataSourceAbstract,
            output_path: str,
            device: str
    ):
        # --> save input params
        self.lag_size = config['lag_size']
        self.prediction_size = config['prediction_size']
        self.batch_size = config['batch_size']
        self.data_source = data_source
        self.config = config
        self.output_path = output_path
        self.device = device

    # ---> Main classic_methods

    def load_data(self) -> None:
        # --> load time series
        (self.train_ids, self.train_lengths), (self.val_ids, self.val_lengths), (self.test_ids, self.test_lengths) = self.data_source.load_split()

        # --> show some info
        logger.info(f'Number of training samples: {len(self.train_ids)}')
        logger.info(f'Number of validation samples:{len(self.val_ids)}')
        logger.info(f'Number of testing samples:{len(self.test_ids)}')

    def show_info(self) -> None:
        print_pretty_json(self.config)
        self.data_source.show_info()

    def get_train_data_loader(self, for_training: bool) -> DataLoader:
        return self.create_data_loader(
            self.train_ids,
            self.train_lengths,
            is_train=for_training
        )

    def get_val_data_loader(self) -> DataLoader:
        return self.create_data_loader(
            self.val_ids,
            self.val_lengths,
            is_train=False
        )

    def get_test_data_loader(self) -> DataLoader:
        return self.create_data_loader(
            self.test_ids,
            self.test_lengths,
            is_train=False
        )

    def get_number_samples(self) -> int:
        return len(self.labels_time_series_dict)

    def get_number_features(self) -> int:
        return self.data_source.get_number_features()

    def get_lag_size(self) -> int:
        return self.lag_size

    def get_prediction_size(self) -> int:
        return self.prediction_size

    def visualize(self, output_path: str):
        raise NotImplementedError()

    # ---> Auxiliary classic_methods

    def create_data_loader(self,
                           time_series_indexes: List[str],
                           time_series_lengths: List[int],
                           is_train: bool
                           ) -> DataLoader:
        raise Exception('Not implemented in abstract class')

    def collate_fn(self, *args):
        raise Exception('Not implemented in abstract class')
