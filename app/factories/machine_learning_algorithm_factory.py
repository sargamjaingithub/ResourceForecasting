from app.models.machine_learning_algorithms.machine_learning_algorithm_interface import MachineLearningAlgorithmInterface
from app.models.machine_learning_algorithms.types.random_forest.main import RandomForest
from app.models.machine_learning_algorithms.types.svm.main import SVM


class MachineLearningAlgorithmFactory:

    def __init__(self):
        raise Exception('This class can not be instantiated')

    @staticmethod
    def select_algorithm(config, *args) -> MachineLearningAlgorithmInterface:
        name = config['name']
        if name == 'random_forest':
            algorithm = RandomForest(*args, **config['args'])
        elif name == 'svm':
            algorithm = SVM(*args, **config['args'])
        else:
            raise Exception('The algorithm with name ' + name + ' does not exist')
        if issubclass(type(algorithm), MachineLearningAlgorithmInterface):
            return algorithm
        else:
            raise Exception('The algorithm does not follow the interface definition')
