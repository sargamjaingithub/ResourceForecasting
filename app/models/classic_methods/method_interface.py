import abc


# Interface for classic_methods
class MethodInterface(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):  # to check that the subclasses follow the interface
        return (hasattr(subclass, 'train') and
                callable(subclass.train) and
                hasattr(subclass, 'compute_loss') and
                callable(subclass.compute_loss) and
                hasattr(subclass, 'predict') and
                callable(subclass.predict) or
                NotImplemented)

    # ---> Main classic_methods

    def __init__(self, device: str):
        super().__init__()
        self.device = device

    @abc.abstractmethod
    def train(self, train_data_loader, val_data_loader):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def compute_loss(self, data_loader):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def predict(self, data_loader):
        raise NotImplementedError('Method not implemented in interface class')
