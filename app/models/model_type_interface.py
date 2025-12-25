import abc


# Interface for configs
from typing import Optional


class ModelTypeInterface(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):  # to check that the subclasses follow the interface
        return (hasattr(subclass, 'show_info') and
                callable(subclass.show_info) and
                hasattr(subclass, 'train_test') and
                callable(subclass.train_test) and
                hasattr(subclass, 'test') and
                callable(subclass.test) and
                hasattr(subclass, 'inference') and
                callable(subclass.inference) and
                hasattr(subclass, 'save_results') and
                callable(subclass.save_results) and
                hasattr(subclass, 'save_model') and
                callable(subclass.save_model) and
                hasattr(subclass, 'load_model') and
                callable(subclass.load_model) or
                NotImplemented)

    # ---> Main classic_methods

    @abc.abstractmethod
    def __init__(self, config: dict, data_model, output_path: str, device: str):
        pass

    @abc.abstractmethod
    def show_info(self):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def train_test(self, save_charts: bool):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def test(self, visualize: int, lag_size: None, evaluation_args: Optional[dict] = {}):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def inference(self):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def save_results(self, visualize: bool):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def save_model(self):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def load_model(self, model_root_path: str):
        raise NotImplementedError('Method not implemented in interface class')
