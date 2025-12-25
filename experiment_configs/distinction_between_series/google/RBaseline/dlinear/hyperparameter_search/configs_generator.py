import json

OUTPUT_PATH = './full_configs'
BASE_CONFIG = './base_config.json'
BASE_NAME = '1_set_experiments_google_type_1_hyperparameter_search_dlinear'


PARAMETERS_TO_TEST = [
    {'keys': ['data', 'type', 'lag_size'], 'values': [32, 128, 512]},
    {'keys': ['model', 'type', 'train_info', 'optimizer', 'args', 'lr'], 'values': [0.1, 0.01, 0.001]}
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


def create_config_rec(base_config: dict, left_parameters_to_set: list, already_set_parameters: list):
    if len(left_parameters_to_set) == 0:
        new_config = base_config.copy()
        new_config_name = f'{BASE_NAME}'
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
            create_config_rec(base_config, left_parameters_to_set[1:], auxiliary_list)


def main():
    base_config = __load_json(BASE_CONFIG)
    create_config_rec(base_config, PARAMETERS_TO_TEST, [])


if __name__ == '__main__':
    main()

