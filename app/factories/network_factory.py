from app.models.networks.network_interface import NetworkInterface
from app.models.networks.types.dlinear.main import DLinear
from app.models.networks.types.informer.main import Informer
from app.models.networks.types.scinet.main import SCINet
from app.models.networks.types.temporal_inception.main import \
    DeepConvolutionalClassifierInceptionBranch3NoEncBigBottleneckComposed


class NetworkFactory:

    def __init__(self):
        raise Exception('This class can not be instantiated')

    @staticmethod
    def select_network(config, *args) -> NetworkInterface:
        name = config['name']
        if name == 'temporal_inception':
            network = DeepConvolutionalClassifierInceptionBranch3NoEncBigBottleneckComposed(*args, **config['args'])
        elif name == 'scinet':
            network = SCINet(*args, **config['args'])
        elif name == 'informer':
            network = Informer(*args, **config['args'])
        elif name == 'dlinear':
            network = DLinear(*args, **config['args'])
        else:
            raise Exception('The network with name ' + name + ' does not exist')
        if issubclass(type(network), NetworkInterface):
            return network
        else:
            raise Exception('The network does not follow the interface definition')
