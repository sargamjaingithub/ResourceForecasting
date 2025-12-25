import json
import logging
from time import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.auxiliary_files.model_methods.loss_functions import select_loss_function
from app.auxiliary_files.model_methods.model_operations import model_arq_to_json
from app.auxiliary_files.model_methods.model_preprocessing import preprocess_model
from app.auxiliary_files.model_methods.optimizers import select_optimizer
from app.auxiliary_files.other_methods.util_functions import print_pretty_json, save_json
from app.auxiliary_files.other_methods.visualize import compare_multiple_lines
from app.evaluation.evaluation import compute_evaluation_list
from app.factories.network_factory import NetworkFactory
from app.models.model_type_interface import ModelTypeInterface

logger = logging.getLogger(__name__)


# define the default workflow for neural networks training
class ModelTypeNeuralNetwork(ModelTypeInterface):

    def __init__(self, config: dict, data_model, output_path: str, device: str):
        super().__init__(config, data_model, output_path, device)

        # save input params
        self.config = config
        self.data_model = data_model
        self.output_path = output_path
        self.device = device

        # load network
        self.network = NetworkFactory.select_network(
            config['network'],
            self.data_model.get_lag_size(),
            self.data_model.get_prediction_size(),
            self.data_model.get_number_features(),
            self.device
        )

        # read config
        config = config['type']
        self.early_stopping = True if 'early_stopping' not in config['train_info'] else config['train_info']['early_stopping']
        self.number_epochs = config['train_info']['number_epochs']

        # load components
        self.optimizer = select_optimizer(config['train_info']['optimizer'], self.network)
        self.loss_function = select_loss_function(config['train_info']['loss_function'], self.device)

        # prepare network
        if 'pretrained' in config:
            self.network.load_state_dict(torch.load(config['pretrained'], map_location=device))
        else:
            preprocess_model(config['transforms'], self.network)  # TODO check it does its job

        self.network = self.network.float()
        self.network = self.network.to(self.device)

        # auxiliary variables
        self.train_loss = torch.zeros((self.number_epochs, 1)).to(self.device).detach()
        self.val_loss = torch.zeros((self.number_epochs, 1)).to(self.device).detach()

    # ---> Main classic_methods

    def show_info(self):
        logger.info(f'Model architecture: {json.dumps(model_arq_to_json(self.network), indent=2)}')

    def get_network(self):
        return self.network

    def train_test(self, save_charts=True, compute_evaluation=True):
        # -> train model
        init_time = time()

        # obtain data loaders
        train_data_loader = self.data_model.get_train_data_loader(for_training=True)
        val_data_loader = self.data_model.get_val_data_loader()

        # train network
        self.__train_network(train_data_loader, val_data_loader)
        total_train_time = time() - init_time

        # show training evolution
        losses = [
            (self.train_loss.to('cpu').numpy(), np.arange(self.train_loss.shape[0]), 'train loss'),
            (self.val_loss.to('cpu').numpy(), np.arange(self.val_loss.shape[0]), 'val loss')
        ]
        compare_multiple_lines(False, losses, 'Loss', 'epoch', 'Train evolution', self.output_path + '/train_evolution')
        best_train_scores_dict = {
            'best_model_losses': {
                'best_train_loss': {
                    'value': float(torch.min(self.train_loss).to('cpu').item()),
                    'epoch': int(torch.argmin(self.train_loss).to('cpu').item())
                },
                'best_val_loss': {
                    'value': float(torch.min(self.val_loss).to('cpu').item()),
                    'epoch': int(torch.argmin(self.val_loss).to('cpu').item()),
                    'corresponding_train_loss': float(self.train_loss[int(torch.argmin(self.val_loss).to('cpu').item())].to('cpu').item())  # TODO refactor
                }
            }
        }
        print_pretty_json(best_train_scores_dict)
        save_json(self.output_path + '/best_train_scores.json', best_train_scores_dict)

        # -> predict train, val and test splits
        init_time = time()

        # obtain test data loader
        test_data_loader = self.data_model.get_test_data_loader()

        # recreate train data loader without shuffling
        train_data_loader = self.data_model.get_train_data_loader(for_training=False)

        # compute loss
        train_loss = self.__compute_loss(train_data_loader)
        val_loss = self.__compute_loss(val_data_loader)
        test_loss = self.__compute_loss(test_data_loader)

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
            # predict
            train_data_predictions = self.predict_network(train_data_loader)
            val_data_predictions = self.predict_network(val_data_loader)
            test_data_predictions = self.predict_network(test_data_loader)

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
        # obtain test data loader
        test_data_loader = self.data_model.get_test_data_loader()

        # compute predictions
        test_data_predictions = self.predict_network(test_data_loader)

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
        self.network = self.network.to('cpu')
        torch.save(self.network.state_dict(), self.output_path + '/network_weights.pt')
        self.network = self.network.to(self.device)

    def load_model(self, model_root_path: str):
        self.network.load_state_dict(torch.load(model_root_path + '/network_weights.pt'))

    # --> Auxiliary classic_methods

    def __train_network(self, train_data_loader: DataLoader, val_data_loader: DataLoader):
        count_not_increased_val = 0

        for number_epoch in range(self.number_epochs):
            t = time()
            self.__train_epoch(number_epoch, train_data_loader, self.train_loss)
            train_time = time() - t
            t = time()

            self.__not_train_epoch(number_epoch, val_data_loader, self.val_loss)
            val_time = time() - t

            logger.info(
                str('====> Epoch: {} Train set loss: {:.6f}; time {}. Val set loss: {:.6f}; time: {} \n').format(
                    number_epoch,
                    self.train_loss[number_epoch][0],
                    train_time,
                    self.val_loss[number_epoch][0],
                    val_time
                    )
                )

            if self.early_stopping:
                if number_epoch > 5 and self.val_loss[number_epoch] > torch.mean(self.val_loss[(number_epoch - 5):number_epoch]):
                    count_not_increased_val += 1
                    if count_not_increased_val >= 5:
                        self.train_loss = self.train_loss[:(number_epoch + 1)]
                        self.val_loss = self.val_loss[:(number_epoch + 1)]
                        return
                else:
                    count_not_increased_val = 0

    def __compute_loss(self, data_loader: DataLoader):
        loss = torch.zeros((1, 1)).detach().to(self.device)
        self.__not_train_epoch(0, data_loader, loss)
        return loss.to('cpu').item()

    def predict_network(self, data_loader: DataLoader):
        self.network.eval()

        all_predictions = []

        with torch.no_grad():
            for index, (ids, values, target) in enumerate(data_loader, 0):  # iterate data loader
                values, target = self.data_model.to_device(values, target)
                prediction = self.network.predict(values)

                prediction = prediction.detach().to('cpu').numpy()
                all_predictions.append((ids, prediction))

        return all_predictions

    def __train_epoch(self, number_epoch: int, train_data_loader: DataLoader, losses_array: torch.Tensor):
        self.network.train()

        for index, (_, train_values, train_target) in enumerate(train_data_loader, 0):  # iterate data loader
            train_values, train_target = self.data_model.to_device(train_values, train_target)

            self.optimizer.zero_grad()
            train_output = self.network(train_values, train_target)
            loss = self.loss_function.run(train_output, train_target)
            loss.backward()
            self.optimizer.step()

            losses_array[number_epoch][0] = losses_array[number_epoch][0].add(loss.detach().view(1))  # update loss array

        losses_array[number_epoch][0] = losses_array[number_epoch][0].div(len(train_data_loader))  # update loss array

    def __not_train_epoch(self, number_epoch: int, data_loader: DataLoader, losses_array: torch.Tensor):
        self.network.eval()

        with torch.no_grad():
            for index, (_, values, target) in enumerate(data_loader, 0):  # iterate data loader
                values, target = self.data_model.to_device(values, target)
                output = self.network(values, target)
                loss = self.loss_function.run(output, target)

                losses_array[number_epoch][0] = losses_array[number_epoch][0].add(loss.detach().view(1))  # update loss array

        losses_array[number_epoch][0] = losses_array[number_epoch][0].div(len(data_loader))  # update loss array
