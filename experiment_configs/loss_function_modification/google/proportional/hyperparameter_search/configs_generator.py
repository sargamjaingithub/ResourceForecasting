import json

OUTPUT_PATH = './full_configs/'

BASE_CONFIGS = [
    './base_config.json',
]
BASE_NAMES = [
    '1_set_experiments_google_type_4_proportional_hyperparameter_search_temporal_inception',
]
PARAMETERS_TO_TEST = [
    [
        {'keys': ['model', 'type', 'train_info', 'optimizer', 'args', 'lr'], 'values': [0.1, 0.01, 0.001]},
        {'keys': ['model', 'type', 'train_info', 'loss_function', 'alpha'], 'values': [0.99, 1]},
        {'keys': ['data', 'type', 'reduction_magnitude'], 'values': [-2, -4, -8]}
    ]
]


def __load_json(path):
    with open(path) as file:
        data = json.load(file)
    return data


def __save_json(path, data):
    path += '.json'
    with open(path, 'w') as file:
        json.dump(data, file, indent=2)


def __nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def create_config_rec(base_config: dict, base_name: str, left_parameters_to_set: list, already_set_parameters: list):
    if len(left_parameters_to_set) == 0:
        new_config = base_config.copy()
        new_config_name = base_name
        for already_set_parameter in already_set_parameters:
            keys = already_set_parameter['keys']
            value = already_set_parameter['value']
            __nested_set(new_config, keys, value)
            new_config_name += f'_{keys[-1]}_{value}'
        __save_json(f'{OUTPUT_PATH}/{new_config_name}', new_config)
    else:
        left_parameter_to_set = left_parameters_to_set[0]
        keys = left_parameter_to_set['keys']
        values = left_parameter_to_set['values']

        for value in values:
            auxiliary_list = already_set_parameters.copy()
            auxiliary_list.append({
                'keys': keys,
                'value': value
            })
            create_config_rec(base_config, base_name, left_parameters_to_set[1:], auxiliary_list)


def main():
    for index, base_config in enumerate(BASE_CONFIGS):
        base_config = __load_json(base_config)
        create_config_rec(base_config, BASE_NAMES[index], PARAMETERS_TO_TEST[index], [])


if __name__ == '__main__':
    main()

