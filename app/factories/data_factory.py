from app.data.data_source_abstract import DataSourceAbstract
from app.data.data_type_interface import DataTypeInterface
from app.data.sources.alibaba_2018 import DataSourceAlibaba2018
from app.data.sources.google_2019 import DataSourceGoogle2019
from app.data.types.data_process import DataTypeProcess
from app.data.types.full_values_unbiased import DataTypeFullValuesUnbiased
from app.data.types.full_values_unbiased_informer import DataTypeFullValuesUnbiasedInformer
from app.data.types.full_values_unbiased_intervals import DataTypeFullValuesUnbiasedIntervals
from app.data.types.full_values_unbiased_intervals_informer import DataTypeFullValuesUnbiasedIntervalsInformer
from app.data.types.full_values_unbiased_intervals_prio import DataTypeFullValuesUnbiasedIntervalsPrio
from app.data.types.full_values_unbiased_intervals_prio_balanced import DataTypeFullValuesUnbiasedIntervalsPrioBalanced
from app.data.types.full_values_unbiased_intervals_zero_binary import DataTypeFullValuesUnbiasedIntervalsZeroBinary
from app.data.types.full_values_unbiased_intervals_zero_exponential import \
    DataTypeFullValuesUnbiasedIntervalsZeroExponential
from app.data.types.full_values_unbiased_intervals_zero_proportional import \
    DataTypeFullValuesUnbiasedIntervalsZeroProportional
from app.data.types.full_values_unbiased_intervals_zero_single import DataTypeFullValuesUnbiasedIntervalsZeroSingle
from app.data.types.simple_time_series import DataTypeSimpleTimeSeries


class DataFactory:

    def __init__(self):
        raise Exception('This class can not be instantiated')

    @staticmethod
    def select_data(config: dict, output_path: str, device: str) -> DataTypeInterface:
        source = DataFactory.select_data_source(config['source'], device)
        return DataFactory.select_data_type(config['type'], source, output_path, device)

    @staticmethod
    def select_data_source(config, *args) -> DataSourceAbstract:
        name = config['name']
        if name == 'google_2019':
            data = DataSourceGoogle2019(config, *args)
        elif name == 'alibaba_2018':
            data = DataSourceAlibaba2018(config, *args)
        else:
            raise Exception('The data source with name ' + name + ' does not exist')
        if issubclass(type(data), DataSourceAbstract):
            return data
        else:
            raise Exception('The data source does not follow the interface definition')

    @staticmethod
    def select_data_type(config, *args) -> DataTypeInterface:
        name = config['name']
        if name == 'full_values_unbiased':
            data = DataTypeFullValuesUnbiased(config, *args)
        elif name == 'full_values_unbiased_informer':
            data = DataTypeFullValuesUnbiasedInformer(config, *args)
        elif name == 'full_values_unbiased_intervals':
            data = DataTypeFullValuesUnbiasedIntervals(config, *args)
        elif name == 'full_values_unbiased_intervals_informer':
            data = DataTypeFullValuesUnbiasedIntervalsInformer(config, *args)
        elif name == 'full_values_unbiased_intervals_prio':
            data = DataTypeFullValuesUnbiasedIntervalsPrio(config, *args)
        elif name == 'full_values_unbiased_intervals_prio_balanced':
            data = DataTypeFullValuesUnbiasedIntervalsPrioBalanced(config, *args)
        elif name == 'full_values_unbiased_intervals_zero_single':
            data = DataTypeFullValuesUnbiasedIntervalsZeroSingle(config, *args)
        elif name == 'full_values_unbiased_intervals_zero_binary':
            data = DataTypeFullValuesUnbiasedIntervalsZeroBinary(config, *args)
        elif name == 'full_values_unbiased_intervals_zero_proportional':
            data = DataTypeFullValuesUnbiasedIntervalsZeroProportional(config, *args)
        elif name == 'full_values_unbiased_intervals_zero_exponential':
            data = DataTypeFullValuesUnbiasedIntervalsZeroExponential(config, *args)
        elif name == 'data_preprocess':
            data = DataTypeProcess(config, *args)
        elif name == 'simple_time_series':
            data = DataTypeSimpleTimeSeries(config, *args)
        else:
            raise Exception('The data type with name ' + name + ' does not exist')
        if issubclass(type(data), DataTypeInterface):
            return data
        else:
            raise Exception('The data type does not follow the interface definition')
