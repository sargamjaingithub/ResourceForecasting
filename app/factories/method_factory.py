from app.models.classic_methods.method_interface import MethodInterface
from app.models.classic_methods.types.autopilot.main import AutopilotMLRecommender
from app.models.classic_methods.types.theta_scan.main import ThetaScan


class MethodFactory:

    def __init__(self):
        raise Exception('This class can not be instantiated')

    @staticmethod
    def select_method(config, *args) -> MethodInterface:
        name = config['name']
        if name == 'autopilot':  # TODO put all types of autopilot
            method = AutopilotMLRecommender(*args, **config['args'])
        elif name == 'theta_scan':
            method = ThetaScan(*args, **config['args'])
        else:
            raise Exception('The method with name ' + name + ' does not exist')
        if issubclass(type(method), MethodInterface):
            return method
        else:
            raise Exception('The method does not follow the interface definition')
