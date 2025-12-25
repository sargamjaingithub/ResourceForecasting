import logging
from time import time

from app.auxiliary_files.other_methods.util_functions import print_pretty_json, save_json
from app.evaluation.evaluation import compute_evaluation_list
from app.factories.method_factory import MethodFactory
from app.models.model_type_interface import ModelTypeInterface

logger = logging.getLogger(__name__)


# define the default workflow for neural networks training
class ModelTypeClassicMethod(ModelTypeInterface):

    def __init__(self, config: dict, data_model, output_path: str, device: str):
        super().__init__(config, data_model, output_path, device)
        self.data_model = data_model
        self.output_path = output_path
        self.device = device

        self.method = MethodFactory.select_method(config['method'], self.device, self.data_model.get_lag_size())

    # ---> Main classic_methods

    def show_info(self):
        pass

    def train_test(self, save_charts=True, compute_evaluation=True):
        # -> train model
        init_time = time()

        # obtain data loaders
        train_data_loader = self.data_model.get_train_data_loader(for_training=True)
        val_data_loader = self.data_model.get_val_data_loader()

        # train
        self.method.train(train_data_loader, val_data_loader)
        total_train_time = time() - init_time

        # -> predict train, val and test splits
        init_time = time()

        # obtain test data loader
        test_data_loader = self.data_model.get_test_data_loader()

        # recreate train data loader without shuffling
        train_data_loader = self.data_model.get_train_data_loader(for_training=False)

        # compute loss
        train_loss = self.method.compute_loss(train_data_loader)
        val_loss = self.method.compute_loss(val_data_loader)
        test_loss = self.method.compute_loss(test_data_loader)

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
            train_data_predictions = self.method.predict(train_data_loader)
            val_data_predictions = self.method.predict(val_data_loader)
            test_data_predictions = self.method.predict(test_data_loader)

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

    def test(self, visualize: int):
        raise NotImplementedError('Method not implemented')

    def inference(self):
        raise NotImplementedError('Method not implemented')

    def save_results(self, visualize):
        pass

    def save_model(self):
        pass

    def load_model(self, model_root_path: str):
        raise NotImplementedError('Method not implemented')
