from app.models.model_type_interface import ModelTypeInterface
from app.models.types.model_type_classic_method import ModelTypeClassicMethod
from app.models.types.model_type_machine_learning import ModelTypeMachineLearningAlgorithm
from app.models.types.model_type_neural_network import ModelTypeNeuralNetwork
from app.models.types.model_type_xai import ModelTypeXAI


class ModelTypeFactory:

    def __init__(self):
        raise Exception('This class can not be instantiated')

    @staticmethod
    def select_model_type(config, *args) -> ModelTypeInterface:
        name = config['type']['name']
        if name == 'neural_network':
            model = ModelTypeNeuralNetwork(config, *args)
        elif name == 'classic_method':
            model = ModelTypeClassicMethod(config, *args)
        elif name == 'machine_learning_algorithm':
            model = ModelTypeMachineLearningAlgorithm(config, *args)
        elif name == 'xai':
            model = ModelTypeXAI(config, *args)
        else:
            raise Exception('The model type with name ' + name + ' does not exist')
        if issubclass(type(model), ModelTypeInterface):
            return model
        else:
            raise Exception('The model type does not follow the interface definition')
