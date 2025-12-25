import argparse
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn

from app.evaluation.evaluation import compute_evaluation_list
from factories.data_factory import DataFactory


# plt.rcParams.update({'font.size': 45})


def parse_arguments():
    def parse_bool(s: str):
        if s.casefold() in ['1', 'true', 'yes']:
            return True
        if s.casefold() in ['0', 'false', 'no']:
            return False
        raise ValueError()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment', type=str, help='Path of the model to use', required=True)
    parser.add_argument('--data', type=str, help='Path of the data to use', required=True)
    parser.add_argument('--output', type=str, help='Path to the output directory', required=True)
    parser.add_argument('--device', type=str, help='Device to use', required=False, default='cpu')
    parser.add_argument('--debug', type=parse_bool, help='Run as debug', required=False, default=False)
    args = parser.parse_args()
    return args


def filters_evaluation_comparation(experiment_path: str, data_path: str, output_path: str, device: str, debug: bool):
    def compare_multiple_lines(lines, y_label, x_label, title, path):
        fig, ax = plt.subplots(figsize=(60, 20))
        for line in lines:
            y, x, color, linewidth, label = line
            ax.plot(x, y, color=color, linewidth=linewidth, label=label)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # TODO automatically define position of axis labels
        # plt.text(0.1, 0.3, y_label, rotation=90)
        # plt.text(70, -1.1, x_label)
        plt.title(title)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, loc='best', shadow=True)
        ax.get_xaxis().set_ticklabels([])
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticklabels([])
        ax.get_yaxis().set_ticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.savefig(path, bbox_inches='tight')

    def compute_ce(network, data_model, device):
        network.eval()

        time_series_ids_list = data.test_ids
        time_series_lengths_list = data.test_lengths

        ce_list = []
        criterion = nn.CrossEntropyLoss(reduction='sum')
        for time_series_index in range(len(time_series_lengths_list)):
            time_series_id = time_series_ids_list[time_series_index]
            time_series_length = time_series_lengths_list[time_series_index]
            data_loader = data.create_data_loader(
                [time_series_id],
                [time_series_length],
                is_train=False
            )

            count = 0
            ce = 0
            for index, (ids, values, target) in enumerate(data_loader, 0):  # iterate data loader
                values, target = data_model.to_device(values, target)

                output = network.predict(values)

                ce += criterion(output, target).detach().item()
                count += values.shape[0]

            ce /= count
            ce_list.append(ce)

        np.save(f'{output_path}/ce_results', np.asarray(ce_list))

    # config
    if data_path == 'alibaba':
        config = {
            'data': {
                'source': {
                    'name': 'alibaba_2018',
                    'directory_path': '../data/alibaba_2018/data/third_preprocess_compressed_standarized',
                    'time_feature': 'time_stamp',
                    'target_feature': 'maximum_cpu_average_usage_relative',
                    'batch_size': 512,
                    'subsample': 10,
                    'preprocessing_steps': {
                        'global': [
                            {
                                'name': 'sum',
                                'values': [
                                    0.8293231410685978,
                                    0.9069997126030365,
                                    0.8777236007469805,
                                    0.7067615631752456,
                                    0.7360793569458134,
                                    0.7849484188206659,
                                    2.858544167686376,
                                    2.8397778636204842,
                                    2.919976251757168,
                                    2.879644460096365,
                                    2.8391935283060676,
                                    2.8385477189535084,
                                    0.5219586660230582,
                                    0.5223685403149223,
                                    0.5422420611292673,
                                    0.5423664695192236,
                                    1.4252378830568786,
                                    1.3246437619650067
                                ]
                            }
                        ],
                        'values': [
                            {
                                'name': 'select_feature',
                                'feature': 'maximum_cpu_average_usage_relative'
                            }
                        ],
                        'target': []
                    },
                    'split': {
                        'name': 'random',
                        'train_size': 0.7,
                        'val_size': 0.15
                    }
                }
            },
            'manual_seed': 0
        }
    elif data_path == 'google':
        config = {
            'data': {
                'source': {
                    'name': 'google_2019',
                    'directory_path': '../data/google_2019_a/global_data/third_preprocess_cluster_selection_pca_standarized',
                    'time_feature': 'time_stamp',
                    'target_feature': 'cpu_maximum_usage',
                    'batch_size': 512,
                    # 'subsample': 10,
                    'preprocessing_steps': {
                        'global': [
                            {
                                'name': 'sum',
                                'values': [
                                    1.6968,
                                    1.875,
                                    1.3773,
                                    3.3659,
                                    0.7373,
                                    0.7384,
                                    0.3806,
                                    0.7783,
                                    2.7883,
                                    1.9111,
                                    1.8452,
                                    1.6333,
                                    1.4973,
                                    1.7893
                                ]
                            }
                        ],
                        'values': [
                            {
                                'name': 'select_feature',
                                'feature': 'cpu_maximum_usage'
                            }
                        ],
                        'target': []
                    },
                    'split': {
                        'name': 'random',
                        'train_size': 0.7,
                        'val_size': 0.15
                    }
                }
            },
            'manual_seed': 0
        }
    else:
        raise NotImplementedError

    # config setup
    if int(config['manual_seed']) == -1:
        print('Random seed not provided!!!')
        config['manual_seed'] = random.randint(1, 10000)
    random.seed(config['manual_seed'])
    np.random.seed(config['manual_seed'])
    torch.manual_seed(config['manual_seed'])
    torch.cuda.manual_seed(config['manual_seed'])
    cudnn.benchmark = False
    torch.set_default_tensor_type(torch.FloatTensor)

    # load data
    data = DataFactory.select_data_source(config['data']['source'], device)
    _, _, (test_indexes, test_lengths) = data.load_split()
    test_final_predictions_ids, _, _, test_value_traces, _ = data.load_time_series_list(
        test_indexes,
        times=False,
        init=False
    )

    # compute predictions
    window_size = 3
    elevation_factor = 0.1
    test_final_predictions_max_past = []
    test_final_predictions_max_future = []
    test_final_predictions_mean_present = []
    test_final_predictions_mean_past = []
    test_final_predictions_past_value = []
    for index, time_series_values in enumerate(test_value_traces):  # iterate data loader

        # max past
        time_series_values_max_past = deepcopy(time_series_values)
        for time_position in range(window_size, time_series_values.shape[0]):
            max_value = np.max(time_series_values[(time_position - window_size):time_position])
            time_series_values_max_past[time_position] = max_value + max_value * elevation_factor
        test_final_predictions_max_past.append(time_series_values_max_past)

        # max future
        time_series_values_max_future = deepcopy(time_series_values)
        for time_position in range(time_series_values.shape[0] - window_size):
            max_value = np.max(time_series_values[time_position:(time_position + window_size)])
            time_series_values_max_future[time_position] = max_value + max_value * elevation_factor
        test_final_predictions_max_future.append(time_series_values_max_future)

        # mean present
        time_series_values_mean_present = deepcopy(time_series_values)
        for time_position in range(window_size // 2, time_series_values.shape[0] - window_size // 2):
            mean_value = np.mean(
                time_series_values[(time_position - window_size // 2):(time_position + window_size // 2)])
            time_series_values_mean_present[time_position] = mean_value
        test_final_predictions_mean_present.append(time_series_values_mean_present)

        # mean past
        time_series_values_mean_past = deepcopy(time_series_values)
        for time_position in range(window_size, time_series_values.shape[0]):
            mean_value = np.mean(time_series_values[(time_position - window_size):time_position])
            time_series_values_mean_past[time_position] = mean_value
        test_final_predictions_mean_past.append(time_series_values_mean_past)

        # past value
        time_series_values_past_value = deepcopy(time_series_values)
        for time_position in range(1, time_series_values.shape[0]):
            past_value = time_series_values[time_position - 1]
            time_series_values_past_value[time_position] = past_value
        test_final_predictions_past_value.append(time_series_values_past_value)

        # remove last part of trace, because is not predicted in some of the methods because of the window_sze
        test_value_traces[index] = test_value_traces[index][:-window_size]
        test_final_predictions_max_past[index] = test_final_predictions_max_past[index][:-window_size]
        test_final_predictions_max_future[index] = test_final_predictions_max_future[index][:-window_size]
        test_final_predictions_mean_present[index] = test_final_predictions_mean_present[index][:-window_size]
        test_final_predictions_mean_past[index] = test_final_predictions_mean_past[index][:-window_size]
        test_final_predictions_past_value[index] = test_final_predictions_past_value[index][:-window_size]

    # visualize
    id_to_visualize = 0
    range_to_visualize = [320, 370]
    lines = [
        (
            test_value_traces[id_to_visualize][range_to_visualize[0]:range_to_visualize[1]],
            np.arange(range_to_visualize[0], range_to_visualize[1]),
            '#7eb3af',
            15,
            'original series'
        ),
        #(
        #    test_final_predictions_max_past[id_to_visualize][range_to_visualize[0]:range_to_visualize[1]],
        #    np.arange(range_to_visualize[0], range_to_visualize[1]),
        #    "#618264",
        #    9,
        #    'max past filter'
        #),
        (
            test_final_predictions_max_future[id_to_visualize][range_to_visualize[0]:range_to_visualize[1]],
            np.arange(range_to_visualize[0], range_to_visualize[1]),
            "#618264",
            12,
            'max future filter'
        ),
        #(
        #    test_final_predictions_mean_present[id_to_visualize][range_to_visualize[0]:range_to_visualize[1]],
        #    np.arange(range_to_visualize[0], range_to_visualize[1]),
        #    "gray",
        #    12,
        #    'mean present filter'
        #),
        (
            test_final_predictions_mean_past[id_to_visualize][range_to_visualize[0]:range_to_visualize[1]],
            np.arange(range_to_visualize[0], range_to_visualize[1]),
            "gray",
            12,
            'mean past filter'
        ),
        (
            test_final_predictions_past_value[id_to_visualize][range_to_visualize[0]:range_to_visualize[1]],
            np.arange(range_to_visualize[0], range_to_visualize[1]),
            "#3c474b",
            12,
            'past value filter'
        )
    ]

    compare_multiple_lines(
        lines,
        'CPU consumption',
        'time',
        f'',
        f'{output_path}/different_preprocessings.png'
    )

    # do evaluation
    evaluation_max_past = compute_evaluation_list(
        None,
        test_final_predictions_ids,
        test_value_traces,
        test_final_predictions_max_past,
        output_path,
        window_size,
        visualize=0
    )
    evaluation_max_future = compute_evaluation_list(
        None,
        test_final_predictions_ids,
        test_value_traces,
        test_final_predictions_max_future,
        output_path,
        window_size,
        visualize=0
    )
    evaluation_mean_present = compute_evaluation_list(
        None,
        test_final_predictions_ids,
        test_value_traces,
        test_final_predictions_mean_present,
        output_path,
        window_size,
        visualize=0
    )
    evaluation_mean_past = compute_evaluation_list(
        None,
        test_final_predictions_ids,
        test_value_traces,
        test_final_predictions_mean_past,
        output_path,
        window_size,
        visualize=0
    )
    evaluation_past_value = compute_evaluation_list(
        None,
        test_final_predictions_ids,
        test_value_traces,
        test_final_predictions_past_value,
        output_path,
        window_size,
        visualize=0
    )
    # compute_ce(model.get_network(), data, device)

    # plot
    print('Max Past:', evaluation_max_past)
    print('Max Future:', evaluation_max_future)
    print('Mean Present:', evaluation_mean_present)
    print('Mean Past:', evaluation_mean_past)
    print('Past Value:', evaluation_past_value)


if __name__ == '__main__':
    args = parse_arguments()
    filters_evaluation_comparation(args.experiment, args.data, args.output, args.device, args.debug)
