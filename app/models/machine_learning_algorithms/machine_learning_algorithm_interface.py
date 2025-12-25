import abc


# Interface for networks
class MachineLearningAlgorithmInterface(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):  # to check that the subclasses follow the interface
        return (hasattr(subclass, 'load') and
                callable(subclass.load) and
                hasattr(subclass, 'train') and
                callable(subclass.train) and
                hasattr(subclass, 'score') and
                callable(subclass.score) and
                hasattr(subclass, 'predict') and
                callable(subclass.predict) or
                NotImplemented)

    # ---> Main classic_methods

    def __init__(self, lag_size: int, prediction_size: int, number_features: int, device: str):
        super().__init__()
        self.lag_size = lag_size
        self.prediction_size = prediction_size
        self.number_features = number_features
        self.device = device

    @abc.abstractmethod
    def load(self, algorithm_args, x, y):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def train(self, x, y):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def score(self, x, y):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def predict(self, x):
        raise NotImplementedError('Method not implemented in interface class')
