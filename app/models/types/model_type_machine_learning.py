import logging
from functools import reduce
from time import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.auxiliary_files.other_methods.util_functions import print_pretty_json, save_json, load_json
from app.evaluation.evaluation import compute_evaluation_list
from app.factories.machine_learning_algorithm_factory import MachineLearningAlgorithmFactory
from app.models.model_type_interface import ModelTypeInterface

logger = logging.getLogger(__name__)


# define the default workflow for neural networks training
class ModelTypeMachineLearningAlgorithm(ModelTypeInterface):

    def __init__(self, config: dict, data_model, output_path: str, device: str):
        super().__init__(config, data_model, output_path, device)

        # save input params
        self.config = config
        self.data_model = data_model
        self.output_path = output_path
        self.device = device

        # load network
        self.algorithm = MachineLearningAlgorithmFactory.select_algorithm(
            config['algorithm'],
            self.data_model.get_lag_size(),
            self.data_model.get_prediction_size(),
            self.data_model.get_number_features(),
            self.device
        )

    # ---> Main classic_methods

    def show_info(self):
        pass

    def train_test(self, save_charts=True, compute_evaluation=True):
        # -> train model
        init_time = time()

        # obtain data loaders
        train_data_loader = self.data_model.get_train_data_loader(for_training=True)
        _, train_x, train_y = self.__data_loader_to_dataset(train_data_loader)

        # train algorithm
        algorithm_args = self.algorithm.train(train_x, train_y)
        save_json(self.output_path + '/args', algorithm_args)
        total_train_time = time() - init_time

        # -> predict train, val and test splits
        init_time = time()

        # obtain val and test data loaders
        val_data_loader = self.data_model.get_val_data_loader()
        val_ids, val_x, val_y = self.__data_loader_to_dataset(val_data_loader)
        test_data_loader = self.data_model.get_test_data_loader()
        test_ids, test_x, test_y = self.__data_loader_to_dataset(test_data_loader)

        # recreate train data loader without shuffling
        train_data_loader = self.data_model.get_train_data_loader(for_training=True)
        train_ids, train_x, train_y = self.__data_loader_to_dataset(train_data_loader)

        # compute loss
        train_loss = self.algorithm.score(train_x, train_y)
        val_loss = self.algorithm.score(val_x, val_y)
        test_loss = self.algorithm.score(test_x, test_y)

        total_test_time = time() - init_time
        final_model_losses_dict = {
            'final_model_losses': {
                'final_train_loss': train_loss,
                'final_val_loss': val_loss,
                'final_test_loss': test_loss
            }
        }
        print_pretty_json(final_model_losses_dict)
        save_json(self.output_path + '/final_model_losses.json', final_model_losses_dict)

        # -> do visualization and evaluation
        if compute_evaluation:
            train_data_loader = self.data_model.get_train_data_loader(for_training=False)
            train_ids, train_x, train_y = self.__data_loader_to_dataset(train_data_loader)

            # predict
            train_data_predictions = self.algorithm.predict(train_x)
            val_data_predictions = self.algorithm.predict(val_x)
            test_data_predictions = self.algorithm.predict(test_x)

            train_data_predictions = [(train_ids, np.expand_dims(train_data_predictions, axis=1))]
            val_data_predictions = [(val_ids, np.expand_dims(val_data_predictions, axis=1))]
            test_data_predictions = [(test_ids, np.expand_dims(test_data_predictions, axis=1))]

            # -> do evaluation
            train_final_predictions_ids, train_initial_traces, train_final_predictions = self.data_model.get_final_predictions(train_data_predictions)
            val_final_predictions_ids, val_initial_traces, val_final_predictions = self.data_model.get_final_predictions(val_data_predictions)
            test_final_predictions_ids, test_initial_traces, test_final_predictions = self.data_model.get_final_predictions(test_data_predictions)

            train_evaluation = compute_evaluation_list(
                'train',
                train_final_predictions_ids,
                train_initial_traces,
                train_final_predictions,
                self.output_path,
                self.data_model.get_lag_size()
            )
            val_evaluation = compute_evaluation_list(
                'val',
                val_final_predictions_ids,
                val_initial_traces,
                val_final_predictions,
                self.output_path,
                self.data_model.get_lag_size()
            )
            test_evaluation = compute_evaluation_list(
                'test',
                test_final_predictions_ids,
                test_initial_traces,
                test_final_predictions,
                self.output_path,
                self.data_model.get_lag_size()
            )
            print_pretty_json(train_evaluation)
            print_pretty_json(val_evaluation)
            print_pretty_json(test_evaluation)
            save_json(self.output_path + '/train_evaluation.json', train_evaluation)
            save_json(self.output_path + '/val_evaluation.json', val_evaluation)
            save_json(self.output_path + '/test_evaluation.json', test_evaluation)

        return total_train_time, total_test_time

    def test(self, visualize: int, lag_size: None, evaluation_args: Optional[dict] = {}):
        # obtain data loaders
        test_data_loader = self.data_model.get_test_data_loader()
        test_ids, test_x, test_y = self.__data_loader_to_dataset(test_data_loader)

        # compute predictions
        test_data_predictions = self.algorithm.predict(test_x)
        test_data_predictions = [(test_ids, np.expand_dims(test_data_predictions, axis=1))]

        # do evaluation
        test_final_predictions_ids, test_initial_traces, test_final_predictions = self.data_model.get_final_predictions(test_data_predictions)
        test_evaluation = compute_evaluation_list(
            'test',
            test_final_predictions_ids,
            test_initial_traces,
            test_final_predictions,
            self.output_path,
            self.data_model.get_lag_size() if lag_size is None else lag_size,
            evaluation_args=evaluation_args,
            visualize=visualize
        )
        print_pretty_json(test_evaluation)
        save_json(self.output_path + '/test_evaluation.json', test_evaluation)

        return test_evaluation

    def inference(self):
        raise NotImplementedError('Method not implemented')

    def save_results(self, visualize):
        pass

    def save_model(self):
        pass

    def load_model(self, model_root_path: str):
        algorithm_args = load_json(model_root_path + '/args.json')

        train_data_loader = self.data_model.get_train_data_loader(for_training=True)
        _, train_x, train_y = self.__data_loader_to_dataset(train_data_loader)
        self.algorithm.load(algorithm_args, train_x, train_y)

    # --> Auxiliary classic_methods

    def __data_loader_to_dataset(self, data_loader: DataLoader):
        all_ids = None
        x = None
        y = None
        with torch.no_grad():
            for index, (ids, values, target) in enumerate(data_loader, 0):  # iterate data loader
                if x is None:
                    all_ids = ids
                    x = values
                    y = target
                else:
                    all_ids += ids
                    x = torch.cat((x, values))
                    y = torch.cat((y, target))
        x = x.detach().numpy()
        y = y.detach().numpy()
        if x.ndim > 2:
            x = np.reshape(x, (x.shape[0], reduce(lambda j, k: j * k, x.shape[1:])))
        y = np.ravel(y)
        return all_ids, x, y
