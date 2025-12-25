import argparse
import os
import random
from typing import List, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from matplotlib.lines import Line2D
from numpy.lib.stride_tricks import sliding_window_view

from app.evaluation.evaluation import move_over_0, find_spikes
from auxiliary_files.other_methods.util_functions import load_json, save_json
from factories.data_factory import DataFactory
from factories.model_factory import ModelTypeFactory

plt.rcParams.update({'font.size': 45})


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


def example_explainability(experiment_path: str, output_path: str, device: str):
    from captum.attr import DeepLiftShap

    def predict_attributions(network, data_loader, data_model, spikes_ids, device):
        network.eval()

        dl = DeepLiftShap(network)

        spikes_ids = set(spikes_ids)
        attributions_ids = []
        attributions_values = []

        for index, (ids, values, target) in enumerate(data_loader, 0):  # iterate data loader
            values, _ = data_model.to_device(values, target)

            prediction = network.predict(values)
            prediction_target = prediction.argmax(dim=1)

            prediction_target = prediction_target.tolist()  # requirement of captum (quite strange behaviour of _select_targets function)
            baselines = torch.randn(values.shape) * 0.001
            baselines = baselines.to(device)
            attributions = dl.attribute(values, baselines, target=prediction_target).detach()
            attributions = torch.abs(attributions)

            for index_batch, (time_series_id, time_series_position) in enumerate(ids):
                if time_series_position in spikes_ids:
                    attributions_ids.append(time_series_position)
                    attributions_values.append(attributions[index_batch].to('cpu').detach().numpy())

        return attributions_ids, attributions_values

    # load config
    config_path = os.path.join(experiment_path, 'initial_config.json')
    config = load_json(config_path)

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
    config['data']['type']['batch_size'] = 32
    config['data']['source']['directory_path'] = '../data/alibaba_2018/data/third_preprocess_compressed_standarized_chart_xai/'
    data = DataFactory.select_data(config['data'], output_path, device)
    data.load_data()

    # load model
    model = ModelTypeFactory.select_model_type(config['model'], data, output_path, device)
    model.load_model(experiment_path)

    # test model
    test_data_loader = data.get_test_data_loader()
    test_data_predictions = model.predict_network(test_data_loader)
    test_final_predictions_ids, test_initial_traces, test_final_predictions = data.get_final_predictions(test_data_predictions)

    test_initial_trace = test_initial_traces[0][512:]
    test_final_prediction = test_final_predictions[0][512:]

    spike_list = find_spikes(
        test_initial_trace,
        512,
        250,
        2
    )
    print(spike_list)

    fig, ax = plt.subplots(figsize=(40, 15))
    ax.plot(np.arange(test_initial_trace.shape[0]), test_initial_trace, color='#7eb3af', linewidth=5)
    ax.plot(np.arange(test_initial_trace.shape[0]), test_final_prediction, color='#3c474b', linewidth=5)
    for point in [556, 844, 1132, 1420]:
        y, x = test_initial_trace[point], point
        ax.plot(
            x,
            y,
            marker='o',
            markersize=16,
            markeredgewidth=5,
            markerfacecolor='red',
            markeredgecolor='red'
        )
    plt.xlabel('time', labelpad=50)
    plt.ylabel('CPU consumption', labelpad=50)
    plt.xlim(0, 1420)
    line_1 = Line2D([], [], label='real', color='#7eb3af', linewidth=5)
    line_2 = Line2D([], [], label='pred', color='#3c474b', linewidth=5)
    line_3 = Line2D([], [], label='lag for one prediction', color='orange', linestyle='dashed', linewidth=5)
    point_1 = Line2D([], [], label='linked spikes', marker='o', linewidth=0, markersize=16, markeredgewidth=5, markerfacecolor='red', markeredgecolor='red', color='red')
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([line_1, line_2, point_1, line_3])
    rect = patches.Rectangle((1420 - 512, np.min(test_initial_trace)), 512, np.max(test_initial_trace) + 0.25, linewidth=5, linestyle='dashed', edgecolor='orange', facecolor='none', zorder=2)
    ax.add_patch(rect)
    # ax.arrow()
    ax.legend(handles=handles, loc='upper left', shadow=True)
    ax.get_xaxis().set_ticks([0, 288, 556, 844, 1132, 1420, 1708])
    ax.get_xaxis().set_ticklabels(['day 0', 'day 1', 'day 2', 'day 3', 'day 4', 'day 5', 'day 6'])
    ax.get_yaxis().set_ticklabels([])
    ax.get_yaxis().set_ticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.savefig(f'{output_path}/attributions_example_full', bbox_inches='tight')

    spike_positions = [556, 844, 1132, 1420]
    if not os.path.exists(f'{output_path}/spikes_attributions_values.npy'):
        spikes_attributions_ids, spikes_attributions_values = predict_attributions(model.get_network(), test_data_loader, data, spike_positions, device)
        print(spikes_attributions_ids)
        np.save(f'{output_path}/spikes_attributions_values', spikes_attributions_values)
    spikes_attributions_values = np.load(f'{output_path}/spikes_attributions_values.npy')
    print(data.data_source.features_values)
    spike_index = 3
    init_range = spike_positions[spike_index] - 512
    end_range = spike_positions[spike_index]
    attributions_full = spikes_attributions_values[spike_index]
    attributions_average = np.mean(attributions_full, axis=0)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(40, 15))
    axs[0].plot(np.arange(attributions_average.shape[0]), test_initial_trace[init_range:(end_range)], color='#7eb3af', linewidth=5)
    axs[0].plot(np.arange(attributions_average.shape[0]), test_final_prediction[init_range:(end_range)], color='#3c474b', linewidth=5)

    rect = patches.Rectangle((512, np.min(test_initial_trace[init_range:(end_range + 1)])), 6, np.max(test_initial_trace[init_range:(end_range + 1)]) + 0.25, linewidth=5, linestyle='dashed', edgecolor='orange', facecolor='none', zorder=2)
    axs[0].add_patch(rect)

    for point in [spike_positions[spike_index - 1] - init_range, spike_positions[spike_index] - init_range]:
        if point == spike_positions[spike_index - 1] - init_range:
            y, x = test_initial_trace[point + init_range], point
            axs[0].plot(
                x,
                y,
                marker='o',
                markersize=16,
                markeredgewidth=5,
                markerfacecolor='red',
                markeredgecolor='red'
            )
        else:
            y, x = test_initial_trace[point + init_range], point
            axs[0].plot(
                x + 3,
                y,
                marker='o',
                markersize=16,
                markeredgewidth=5,
                markerfacecolor='red',
                markeredgecolor='red'
            )

    cpu_attributions = np.max(attributions_full.take([0, 1, 2, 3, 4, 5], axis=0), axis=0)
    memory_attributions = np.max(attributions_full.take([6, 7, 8, 9, 10, 11], axis=0), axis=0)
    network_attributions = np.max(attributions_full.take([12, 13, 14, 15], axis=0), axis=0)
    disk_attributions = np.max(attributions_full.take([16, 17], axis=0), axis=0)
    # axs[1].plot(np.arange(attributions_average.shape[0]), attributions_average, color='orange', linewidth=5, label='average attributions')
    axs[1].plot(np.arange(attributions_average.shape[0]), cpu_attributions, linewidth=5, label='cpu attributions')
    axs[1].plot(np.arange(attributions_average.shape[0]), memory_attributions, linewidth=5, label='memory attributions')
    axs[1].plot(np.arange(attributions_average.shape[0]), network_attributions, linewidth=5, label='network attributions')
    axs[1].plot(np.arange(attributions_average.shape[0]), disk_attributions, linewidth=5, label='disk attributions')

    # for feature_index, feature in enumerate(data.data_source.features_values):
    #     axs[1].plot(np.arange(attributions_average.shape[0]), attributions_full[feature_index], linewidth=5, label=feature)

    plt.xlabel('lag', labelpad=50)
    axs[0].set_ylabel('CPU consumption', labelpad=50)
    axs[1].set_ylabel('Past importance', labelpad=50)
    line_1 = Line2D([], [], label='real', color='#7eb3af', linewidth=5)
    line_2 = Line2D([], [], label='pred', color='#3c474b', linewidth=5)
    line_3 = Line2D([], [], label='future', color='orange', linestyle='dashed', linewidth=5)
    point_1 = Line2D([], [], label='linked spikes', marker='o', linewidth=0, markersize=16, markeredgewidth=5, markerfacecolor='red', markeredgecolor='red', color='red')
    handles, labels = axs[0].get_legend_handles_labels()
    handles.extend([line_1, line_2, point_1, line_3])
    axs[0].legend(handles=handles, loc='upper left', shadow=True)
    line_3 = Line2D([], [], label='attributions', color='orange', linewidth=5)
    handles, labels = axs[1].get_legend_handles_labels()
    # handles.extend([line_3])
    axs[1].legend(handles=handles, loc='upper left', shadow=True)
    axs[0].get_xaxis().set_ticklabels([])
    axs[0].get_xaxis().set_ticks(np.arange(512, 0, -48))
    axs[0].get_yaxis().set_ticklabels([])
    axs[0].get_yaxis().set_ticks([])
    axs[1].get_xaxis().set_ticklabels([f'-{(512 - units) * 5 // 60}h' for units in np.arange(512, 0, -48)])
    axs[1].get_xaxis().set_ticks(np.arange(512, 0, -48))
    axs[1].get_yaxis().set_ticklabels([])
    axs[1].get_yaxis().set_ticks([])
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    # axs[0].spines['bottom'].set_visible(False)
    axs[0].spines['left'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    # axs[1].spines['bottom'].set_visible(False)
    axs[1].spines['left'].set_visible(False)
    axs[1].set_xlim(0, 519)
    plt.savefig(f'{output_path}/attributions_example_specific', bbox_inches='tight')


def compute_attributions(experiment_path: str, data_path: str, output_path: str, device: str, debug: bool):
    from captum.attr import DeepLiftShap

    def compute_correctly_predicted_spikes_list(
            title: str,
            traces_ids: List[str],
            traces_values: List[np.ndarray],
            traces_predictions: List[np.ndarray],
            output_path: str,
            model_lag_size: int,
            spikes_window_size: Optional[int] = 250,
            spikes_std_factor: Optional[float] = 2,
            spikes_allowed_time_separation: Optional[int] = 4,
            spikes_allowed_over_factor: Optional[float] = 0.6,
            spikes_allowed_under_factor: Optional[float] = 0.2,
    ):
        # move over 0
        traces_values, traces_predictions = move_over_0(traces_values, traces_predictions)

        # compute metrics for all time series
        correctly_predicted_spikes = []
        not_correctly_predicted_spikes = []
        for time_series_id_index, time_series_id in enumerate(traces_ids):
            values = traces_values[time_series_id_index]
            predictions = traces_predictions[time_series_id_index]
            values_spike_list = find_spikes(
                values,
                model_lag_size,
                spikes_window_size,
                spikes_std_factor
            )

            correct_values_spike_list = []
            not_correct_values_spike_list = []
            for spike_position in values_spike_list:
                spike_value = values[spike_position]
                lookup_window = [
                    max(spike_position - spikes_allowed_time_separation, 0, model_lag_size),  # init position
                    spike_position + 1  # end position (not included)
                ]
                found_closed = False
                found_closed_position = None
                index = lookup_window[0]
                while not found_closed and index < lookup_window[1]:
                    value = predictions[index]
                    found_closed = \
                        (spike_value - spike_value * spikes_allowed_under_factor) < \
                        value < \
                        (spike_value + spike_value * spikes_allowed_over_factor)
                    if found_closed:
                        found_closed_position = index
                    index += 1
                if found_closed:
                    correct_values_spike_list.append(spike_position)
                else:
                    not_correct_values_spike_list.append(spike_position)

            correct_values_spike_list = [(time_series_id, position) for position in correct_values_spike_list]
            not_correct_values_spike_list = [(time_series_id, position) for position in not_correct_values_spike_list]
            correctly_predicted_spikes += correct_values_spike_list
            not_correctly_predicted_spikes += not_correct_values_spike_list

        return correctly_predicted_spikes, not_correctly_predicted_spikes


    def predict_all_attributions(network, data_loader, data_model, cor_ids, not_cor_ids, device):
        network.eval()

        dl = DeepLiftShap(network)

        cor_ids = set(cor_ids)
        all_attributions = torch.zeros((data_model.get_number_features(), data_model.get_lag_size())).detach().to(device)
        cor_attributions = torch.zeros((data_model.get_number_features(), data_model.get_lag_size())).detach().to(device)  # correctly predicted spikes attributions
        cor_attributions_full_ids, cor_attributions_full_values = [], []
        not_cor_attributions = torch.zeros((data_model.get_number_features(), data_model.get_lag_size())).detach().to(device)  # not correctly predicted spikes attributions
        count_all = 0
        count_cor = 0
        count_not_cor = 0

        for index, (ids, values, target) in enumerate(data_loader, 0):  # iterate data loader
            values, _ = data_model.to_device(values, target)

            prediction = network.predict(values)
            prediction_target = prediction.argmax(dim=1)

            prediction_target = prediction_target.tolist()  # requirement of captum (quite strange behaviour of _select_targets function)
            baselines = torch.randn(values.shape) * 0.001
            baselines = baselines.to(device)
            attributions = dl.attribute(values, baselines, target=prediction_target).detach()
            abs_attributions = torch.abs(attributions)

            all_attributions += abs_attributions.sum(0)
            count_all += len(ids)

            cor_indexes = []
            for index_batch, _id in enumerate(ids):
                _id = (_id[0], _id[1])
                if _id in cor_ids:
                    cor_indexes.append(index_batch)
                    cor_attributions_full_ids.append(_id)
                    cor_attributions_full_values.append(attributions[index_batch].to('cpu').detach().numpy())
            if len(cor_indexes) > 0:
                count_cor += len(cor_indexes)
                cor_attributions += torch.index_select(abs_attributions, 0, torch.tensor(cor_indexes).detach().to(device)).sum(0)

            not_cor_indexes = []
            for index_batch, _id in enumerate(ids):
                _id = (_id[0], _id[1])
                if _id in not_cor_ids:
                    not_cor_indexes.append(index_batch)
            if len(not_cor_indexes) > 0:
                count_not_cor += len(not_cor_indexes)
                not_cor_attributions += torch.index_select(abs_attributions, 0, torch.tensor(not_cor_indexes).detach().to(device)).sum(0)

        all_attributions = all_attributions.div(count_all)
        cor_attributions = cor_attributions.div(count_cor)
        not_cor_attributions = not_cor_attributions.div(count_not_cor)
        print(f'Count all {count_all}. Count core {count_cor}. Not count core {count_not_cor}')

        return all_attributions.to('cpu').numpy(), cor_attributions.to('cpu').numpy(), not_cor_attributions.to('cpu').numpy(), cor_attributions_full_ids, np.asarray(cor_attributions_full_values)

    # load config
    config_path = os.path.join(experiment_path, 'initial_config.json')
    config = load_json(config_path)

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

    if not os.path.exists(f'{output_path}/all_attributions.npy'):
        # load data
        if debug:
            config['data']['source']['subsample'] = 10
        config['data']['type']['batch_size'] = 32
        config['data']['source']['directory_path'] = data_path
        data = DataFactory.select_data(config['data'], output_path, device)
        data.load_data()

        # load model
        model = ModelTypeFactory.select_model_type(config['model'], data, output_path, device)
        model.load_model(experiment_path)

        # test model
        test_data_loader = data.get_test_data_loader()
        test_data_predictions = model.predict_network(test_data_loader)
        test_final_predictions_ids, test_initial_traces, test_final_predictions = data.get_final_predictions(test_data_predictions)
        correctly_predicted_spikes, not_correctly_predicted_spikes = compute_correctly_predicted_spikes_list(
            'none',
            test_final_predictions_ids,
            test_initial_traces,
            test_final_predictions,
            output_path,
            data.get_lag_size()
        )
        print(f'Number of correctly predicted spikes: {len(correctly_predicted_spikes)}')
        print(f'Number of not correctly predicted spikes: {len(not_correctly_predicted_spikes)}')

        # compute all attributions
        all_attributions, cor_attributions, not_cor_attributions, cor_attributions_full_ids, cor_attributions_full_values = predict_all_attributions(model.get_network(), test_data_loader, data, correctly_predicted_spikes, not_correctly_predicted_spikes, device)
        np.save(f'{output_path}/all_attributions', all_attributions)
        print(f'All attributions output contains Nan? {np.isnan(all_attributions).any()}')
        print(f'All attributions output contains Inf? {np.isinf(all_attributions).any()}')
        np.save(f'{output_path}/cor_attributions', cor_attributions)
        print(f'Cor attributions output contains Nan? {np.isnan(cor_attributions).any()}')
        print(f'Cor attributions output contains Inf? {np.isinf(cor_attributions).any()}')
        np.save(f'{output_path}/not_cor_attributions', not_cor_attributions)
        print(f'Not cor attributions output contains Nan? {np.isnan(not_cor_attributions).any()}')
        print(f'Not cor attributions output contains Inf? {np.isinf(not_cor_attributions).any()}')
        np.save(f'{output_path}/cor_attributions_full_values', cor_attributions_full_values)
        save_json(f'{output_path}/cor_attributions_full_ids', cor_attributions_full_ids)
        print(f'Cor attributions full output contains Nan? {np.isnan(cor_attributions_full_values).any()}')
        print(f'Cor attributions full output contains Inf? {np.isinf(cor_attributions_full_values).any()}')

    # plot
    all_attributions = np.load(f'{output_path}/all_attributions.npy')
    cor_attributions = np.load(f'{output_path}/cor_attributions.npy')
    not_cor_attributions = np.load(f'{output_path}/not_cor_attributions.npy')

    all_attributions_averaged_features = np.mean(all_attributions, axis=0)
    cor_attributions_averaged_features = np.mean(cor_attributions, axis=0)
    not_cor_attributions_averaged_features = np.mean(not_cor_attributions, axis=0)

    fig, ax = plt.subplots(figsize=(40, 20))
    ax.plot(np.arange(-512, 0), cor_attributions_averaged_features, color='#3c474b', linewidth=15, label='correctly predicted spikes')
    # ax.plot(np.arange(-512, 0), not_cor_attributions_averaged_features, color='#7eb3af', linewidth=15, label='not correctly predicted spikes')
    ax.plot(np.arange(-512, 0), all_attributions_averaged_features, color='#7eb3af', linewidth=15, label='all')
    plt.xlabel('lag')
    plt.ylabel('Past importance')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, loc='best', shadow=True)
    #ax.get_xaxis().set_ticks([])
    #ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.get_yaxis().set_ticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.savefig(f'{output_path}/attributions_comparison', bbox_inches='tight')


def visualize_attributions(experiment_path: str, data_path: str, output_path: str, device: str, debug: bool):
    def __visualize_attributions(attributions, cpu_indexes, memory_indexes, network_indexes, disk_indexes, dataset):
        fig, axs = plt.subplots(figsize=(40, 20))
        cpu_attributions = np.max(attributions.take(cpu_indexes, axis=0), axis=0)
        axs.plot(np.arange(attributions.shape[1]), cpu_attributions, linewidth=10, label='cpu attributions')
        memory_attributions = np.max(attributions.take(memory_indexes, axis=0), axis=0)
        axs.plot(np.arange(attributions.shape[1]), memory_attributions, linewidth=10, label='memory attributions')
        if network_indexes:
            network_attributions = np.max(attributions.take(network_indexes, axis=0), axis=0)
            axs.plot(np.arange(attributions.shape[1]), network_attributions, linewidth=10, label='network attributions')
        if disk_indexes:
            disk_attributions = np.max(attributions.take(disk_indexes, axis=0), axis=0)
            axs.plot(np.arange(attributions.shape[1]), disk_attributions, linewidth=10, label='disk attributions')
        plt.xlabel('lag')
        plt.ylabel('Past importance')
        handles, labels = axs.get_legend_handles_labels()
        axs.legend(handles=handles, loc='best', shadow=True)
        axs.get_xaxis().set_ticklabels([f'-{(512 - units) * 5 // 60}h' for units in np.arange(512, 0, -48)])
        axs.get_xaxis().set_ticks(np.arange(512, 0, -48))
        axs.get_yaxis().set_ticklabels([])
        axs.get_yaxis().set_ticks([])
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        # axs[0].spines['bottom'].set_visible(False)
        axs.spines['left'].set_visible(False)
        plt.savefig(f'{output_path}/attributions_{dataset}', bbox_inches='tight')

    def __select_attributions_bigger_spikes(number_spikes, attributions_ids, attributions_values, data, feature_index):
        if number_spikes is not None and number_spikes < attributions_values.shape[0]:
            initial_shape = attributions_values.shape
            distances = np.zeros(attributions_values.shape[0])
            for time_series_index in range(attributions_values.shape[0]):
                time_series_id, time_series_position = attributions_ids[time_series_index]
                _, time_series_values = data.data_source.load_time_series(time_series_id)
                lag_mean = np.mean(time_series_values[feature_index, (time_series_position - 512):time_series_position])
                distances[time_series_index] = time_series_values[feature_index, time_series_position] - lag_mean
            sorted_indices = np.argsort(-distances)
            sorted_indices = sorted_indices[:number_spikes]

            attributions_ids_new = [attributions_ids[index] for index in sorted_indices]
            attributions_ids = attributions_ids_new
            attributions_values = attributions_values.take(sorted_indices, axis=0)

            print(f'Spikes reduction from {initial_shape} to {attributions_values.shape}')

        return attributions_ids, attributions_values

    # load config
    config_path = os.path.join(experiment_path, 'initial_config.json')
    config = load_json(config_path)

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
    config['data']['type']['batch_size'] = 32
    config['data']['source']['directory_path'] = data_path
    data = DataFactory.select_data(config['data'], output_path, device)

    # plot
    all_attributions_values = np.load(f'{output_path}/all_attributions.npy')

    if 'alibaba' in experiment_path:
        dataset = 'alibaba'
        cpu_indexes = [0, 1, 2, 3, 4, 5]
        memory_indexes = [6, 7, 8, 9, 10, 11]
        network_indexes = [12, 13, 14, 15]
        disk_indexes = [16, 17]
    elif 'google' in experiment_path:
        dataset = 'google'
        cpu_indexes = [0, 1, 2, 3, 9, 10, 11, 12, 13]
        memory_indexes = [4, 5, 6, 7, 8]
        network_indexes = None
        disk_indexes = None
    else:
        raise NotImplementedError()

    __visualize_attributions(
        all_attributions_values,
        cpu_indexes,
        memory_indexes,
        network_indexes,
        disk_indexes,
        dataset
    )

    number_spikes = 5000
    feature_index = 1
    cor_attributions_full_ids = load_json(f'{output_path}/cor_attributions_full_ids.json')
    cor_attributions_full_values = np.load(f'{output_path}/cor_attributions_full_values.npy')

    _, cor_attributions_full_values = __select_attributions_bigger_spikes(
        number_spikes,
        cor_attributions_full_ids,
        cor_attributions_full_values,
        data,
        feature_index
    )

    cor_attributions_full_values = np.mean(cor_attributions_full_values, axis=0)
    __visualize_attributions(
        cor_attributions_full_values,
        cpu_indexes,
        memory_indexes,
        network_indexes,
        disk_indexes,
        f'{dataset}-spikes-{number_spikes}'
    )


def clusterize_attributions(experiment_path: str, data_path: str, output_path: str, device: str, debug: bool):
    import random
    feature_index = 1
    number_components = 40
    number_spikes = 7000
    window_size = 5
    window_stride = 3
    normalize = False
    clustering_algorithm = 'KMeans'
    clustering_algorithm_args = {'n_clusters': 8}
    '''if 'alibaba' in experiment_path:
        clustering_algorithm = 'KMeans'
        clustering_algorithm_args = {'n_clusters': 8}
    else:
        clustering_algorithm = 'HDBSCAN'
        clustering_algorithm_args = {
            'cluster_selection_method': 'eom',
            'alpha': 1.5,
            'max_cluster_size': 1000
        }'''
    clusters_to_select = None
    attributions_ids = load_json(f'{output_path}/cor_attributions_full_ids.json')
    attributions_values = np.load(f'{output_path}/cor_attributions_full_values.npy')

    config_path = os.path.join(experiment_path, 'initial_config.json')
    config = load_json(config_path)
    config['data']['type']['batch_size'] = 32
    config['data']['source']['directory_path'] = data_path
    data = DataFactory.select_data(config['data'], output_path, device)

    if int(config['manual_seed']) == -1:
        print('Random seed not provided!!!')
        config['manual_seed'] = random.randint(1, 10000)
    random.seed(config['manual_seed'])
    np.random.seed(config['manual_seed'])
    torch.manual_seed(config['manual_seed'])
    torch.cuda.manual_seed(config['manual_seed'])
    cudnn.benchmark = False
    torch.set_default_tensor_type(torch.FloatTensor)

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    import seaborn as sns
    import pandas as pd
    import random
    import matplotlib.pyplot as plt

    time_series_ids = [(time_series_id, time_series_position) for (time_series_id, time_series_position) in attributions_ids]

    #if debug:
    #    time_series_ids = time_series_ids[:50]
    #    attributions_values = attributions_values[:50]

    # Select bigger spikes
    if number_spikes is not None and number_spikes < attributions_values.shape[0]:
        initial_shape = attributions_values.shape
        distances = np.zeros(attributions_values.shape[0])
        for time_series_index in range(attributions_values.shape[0]):
            time_series_id, time_series_position = time_series_ids[time_series_index]
            _, time_series_values = data.data_source.load_time_series(time_series_id)
            lag_mean = np.mean(time_series_values[feature_index, (time_series_position - 512):time_series_position])
            distances[time_series_index] = time_series_values[feature_index, time_series_position] - lag_mean

        # visualize spikes sizes
        plt.figure(figsize=(20, 20))
        plt.hist(distances, 50, cumulative=True)
        plt.xlabel('size')
        plt.ylabel('Frequency')
        plt.title('Spike sizes')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{output_path}/spike_sizes.png')

        sorted_indices = np.argsort(-distances)
        sorted_indices = sorted_indices[:number_spikes]

        time_series_ids = [time_series_ids[index] for index in sorted_indices]
        attributions_values = attributions_values.take(sorted_indices, axis=0)

        print(f'Spikes reduction from {initial_shape} to {attributions_values.shape}')

    # NaNs imputation
    time_series_characteristics = attributions_values
    print(f'Number of samples: {time_series_characteristics.shape[0]}')
    print(f'Number of features: {time_series_characteristics.shape[1]}')
    print('Counts of NaNs', np.count_nonzero(np.isnan(time_series_characteristics)))
    time_series_characteristics = np.nan_to_num(time_series_characteristics)

    # Compact lag in windows
    initial_shape = time_series_characteristics.shape
    if window_size > 1:
        time_series_characteristics = np.flip(np.average(sliding_window_view(np.flip(time_series_characteristics, axis=2), window_shape=window_size, axis=2)[:, :, ::window_stride], axis=3), axis=2)
    print(f'Time reduction from {initial_shape} to {time_series_characteristics.shape}')
    time_series_characteristics = time_series_characteristics.reshape((time_series_characteristics.shape[0], time_series_characteristics.shape[1] * time_series_characteristics.shape[2]))

    # standarized
    # time_series_characteristics = (time_series_characteristics - time_series_characteristics.mean(axis=0)) / (time_series_characteristics.std(axis=0))

    # normalization
    if normalize:
        sc = StandardScaler()
        time_series_characteristics = sc.fit_transform(time_series_characteristics)

    # show variance
    pca = PCA(n_components=min(time_series_characteristics.shape[0], time_series_characteristics.shape[1]))
    pca.fit(time_series_characteristics)
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    plt.figure(figsize=(20, 20))
    plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'{output_path}/pca_features_variance')

    # transformation
    pca = PCA(n_components=number_components)
    input_data = pca.fit_transform(time_series_characteristics)
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    plt.figure(figsize=(20, 20))
    plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'{output_path}/pca_selected_features_variance')

    # clusterization
    cluster_algorithm = eval(clustering_algorithm)(**clustering_algorithm_args)
    clustering_result = cluster_algorithm.fit_predict(input_data)
    clusters, counts = np.unique(clustering_result, return_counts=True)
    print(f'Number clusters: {clusters.shape[0]}. Clusters: {clusters}. Counts: {counts}')

    clusterized_time_series = {clusters[cluster_index]: [] for cluster_index in range(clusters.shape[0])}
    for time_series_index in range(time_series_characteristics.shape[0]):
        time_series_id, time_series_position = time_series_ids[time_series_index]
        time_series_cluster = clustering_result[time_series_index]
        clusterized_time_series[time_series_cluster].append(((time_series_id, time_series_position), time_series_index))

    # visualization
    visualized_time_series = set()
    for cluster_index in range(clusters.shape[0]):
        if cluster_index > 20:
            continue
        examples_per_cluster = 5
        cluster = clusters[cluster_index]
        if len(clusterized_time_series[cluster]) < examples_per_cluster:
            print(f'Warning cluster {cluster} with less samples than examples to show')
            continue
        while examples_per_cluster > 0:
            random_index = random.randint(0, len(clusterized_time_series[cluster]) - 1)
            (time_series_id, time_series_position), time_series_index = clusterized_time_series[cluster][random_index]
            aux_id = f'{time_series_id}_{time_series_position}'
            if aux_id not in visualized_time_series:
                _, time_series_values = data.data_source.load_time_series(time_series_id)
                time_series_values = time_series_values[:, (time_series_position - 512):(time_series_position + 1)]

                fig, axs = plt.subplots(2, 1, sharex=True, figsize=(40, 15))

                if 'alibaba' in experiment_path:
                    cpu_indexes = [0, 1, 2, 3, 4, 5]
                    memory_indexes = [6, 7, 8, 9, 10, 11]
                    network_indexes = [12, 13, 14, 15]
                    disk_indexes = [16, 17]
                elif 'google' in experiment_path:
                    cpu_indexes = [0, 1, 2, 3, 9, 10, 11, 12, 13]
                    memory_indexes = [4, 5, 6, 7, 8]
                    network_indexes = None
                    disk_indexes = None
                else:
                    raise Exception('Unrecognized dataset')

                if disk_indexes:
                    disk_values = np.mean(time_series_values.take(disk_indexes, axis=0), axis=0)
                    disk_values = (disk_values - np.min(disk_values)) / (np.max(disk_values) - np.min(disk_values))
                    axs[0].plot(np.arange(attributions_values.shape[2]), disk_values[:-1], linewidth=5, color='#d62728', label='disk values')
                if network_indexes:
                    network_values = np.mean(time_series_values.take(network_indexes, axis=0), axis=0)
                    network_values = np.divide(network_values - np.min(network_values), np.max(network_values) - np.min(network_values), out=np.zeros_like(network_values), where=( np.max(network_values) - np.min(network_values))!=0)
                    axs[0].plot(np.arange(attributions_values.shape[2]), network_values[:-1], linewidth=5, color='#2ca02c', label='network values')
                memory_values = np.mean(time_series_values.take(memory_indexes, axis=0), axis=0)
                memory_values = (memory_values - np.min(memory_values)) / (np.max(memory_values) - np.min(memory_values))
                axs[0].plot(np.arange(attributions_values.shape[2]), memory_values[:-1], linewidth=5, color='#ff7f0e', label='memory values')
                cpu_values = np.max(time_series_values.take(cpu_indexes, axis=0), axis=0)
                cpu_values = (cpu_values - np.min(cpu_values)) / (np.max(cpu_values) - np.min(cpu_values))
                axs[0].plot(np.arange(attributions_values.shape[2]), cpu_values[:-1], linewidth=5, label='cpu values')

                rect = patches.Rectangle((512, 0), 6, 1, linewidth=5, linestyle='dashed', edgecolor='#17becf', facecolor='none', zorder=2)
                axs[0].add_patch(rect)

                axs[0].plot(
                    512 + 3,
                    cpu_values[-1],
                    marker='o',
                    markersize=16,
                    markeredgewidth=5,
                    markerfacecolor='#1f77b4',
                    markeredgecolor='#1f77b4'
                )

                if disk_indexes:
                    disk_attributions = np.mean(attributions_values[time_series_index].take(disk_indexes, axis=0), axis=0)
                    axs[1].plot(np.arange(attributions_values.shape[2]), disk_attributions, linewidth=5, color='#d62728', label='disk attributions')
                if network_indexes:
                    network_attributions = np.mean(attributions_values[time_series_index].take(network_indexes, axis=0), axis=0)
                    axs[1].plot(np.arange(attributions_values.shape[2]), network_attributions, linewidth=5, color='#2ca02c', label='network attributions')
                memory_attributions = np.mean(attributions_values[time_series_index].take(memory_indexes, axis=0), axis=0)
                axs[1].plot(np.arange(attributions_values.shape[2]), memory_attributions, linewidth=5, color='#ff7f0e', label='memory attributions')
                cpu_attributions = np.max(attributions_values[time_series_index].take(cpu_indexes, axis=0), axis=0)
                axs[1].plot(np.arange(attributions_values.shape[2]), cpu_attributions, linewidth=5, label='cpu attributions')

                plt.xlabel('lag', labelpad=25)
                axs[0].set_ylabel('Resource usage', labelpad=25)
                axs[1].set_ylabel('Past importance', labelpad=25)
                handles, labels = axs[0].get_legend_handles_labels()
                line_3 = Line2D([], [], label='future', color='#17becf', linestyle='dashed', linewidth=5)
                point_1 = Line2D([], [], label='future cpu value', marker='o', linewidth=0, markersize=30, markeredgewidth=5, markerfacecolor='#1f77b4', markeredgecolor='#1f77b4')
                handles.extend([line_3, point_1])
                axs[0].legend(handles=handles, loc='upper left', shadow=True, prop={'size': 37})
                handles, labels = axs[1].get_legend_handles_labels()
                axs[1].legend(handles=handles, loc='upper left', shadow=True, prop={'size': 37})
                axs[0].get_xaxis().set_ticklabels([])
                axs[0].get_xaxis().set_ticks(np.arange(512, 0, -48))
                axs[0].get_yaxis().set_ticklabels([])
                axs[0].get_yaxis().set_ticks([])
                axs[1].get_xaxis().set_ticklabels([f'-{(512 - units) * 5 // 60}h' for units in np.arange(512, 0, -48)])
                axs[1].get_xaxis().set_ticks(np.arange(512, 0, -48))
                axs[1].get_yaxis().set_ticklabels([])
                axs[1].get_yaxis().set_ticks([])
                axs[0].spines['top'].set_visible(False)
                axs[0].spines['right'].set_visible(False)
                # axs[0].spines['bottom'].set_visible(False)
                axs[0].spines['left'].set_visible(False)
                axs[1].spines['top'].set_visible(False)
                axs[1].spines['right'].set_visible(False)
                # axs[1].spines['bottom'].set_visible(False)
                axs[1].spines['left'].set_visible(False)
                axs[1].set_xlim(0, 519)
                plt.savefig(f'{output_path}/attributions_example_cluster_{cluster}_{examples_per_cluster}_id_{time_series_id}', bbox_inches='tight')
                visualized_time_series.add(aux_id)
                examples_per_cluster -= 1

    # pca visualization
    pca = PCA(n_components=2)
    pca_result_2 = pca.fit_transform(time_series_characteristics)
    # clustering_result = cluster_algorithm.fit_predict(pca_result)
    pca_result_2 = pd.DataFrame(pca_result_2, columns=["pca-0", "pca-1"])
    pca_result_2['clusters'] = clustering_result

    plt.figure(figsize=(20, 20))
    sns.scatterplot(
        x="pca-0", y="pca-1",
        hue="clusters",
        palette=sns.color_palette("hls", np.unique(clustering_result).shape[0]),
        data=pca_result_2,
        legend="full",
        alpha=0.3
    )
    plt.savefig(f'{output_path}/time_series_clusterization_pca_2_components')

    pca = PCA(n_components=3)
    pca_result_3 = pca.fit_transform(time_series_characteristics)
    # clustering_result = cluster_algorithm.fit_predict(pca_result)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(
        xs=pca_result_3[:, 0],
        ys=pca_result_3[:, 1],
        zs=pca_result_3[:, 2],
        c=clustering_result,
        cmap='tab10'
    )
    ax.set_xlabel('pca-0')
    ax.set_ylabel('pca-1')
    ax.set_zlabel('pca-2')
    plt.savefig(f'{output_path}/time_series_clusterization_pca_3_components')

    # t-sne visualization
    input_data = time_series_characteristics
    if input_data.shape[1] > 50:
        pca = PCA(n_components=number_components)
        input_data = pca.fit_transform(input_data)
    tsne = TSNE(n_components=2, verbose=1)
    tsne_result_2 = tsne.fit_transform(input_data)
    # clustering_result = cluster_algorithm.fit_predict(tsne_result)
    tsne_result_2 = pd.DataFrame(tsne_result_2, columns=["tsne-0", "tsne-1"])
    tsne_result_2['clusters'] = clustering_result

    plt.figure(figsize=(20, 20))
    sns.scatterplot(
        x="tsne-0", y="tsne-1",
        hue="clusters",
        palette=sns.color_palette("hls", np.unique(clustering_result).shape[0]),
        data=tsne_result_2,
        legend="full",
        alpha=0.3
    )
    plt.savefig(f'{output_path}/time_series_clusterization_tsne_2_components')

    tsne = TSNE(n_components=3, verbose=1)
    tsne_result_3 = tsne.fit_transform(input_data)
    # clustering_result = cluster_algorithm.fit_predict(tsne_result)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(
        xs=tsne_result_3[:, 0],
        ys=tsne_result_3[:, 1],
        zs=tsne_result_3[:, 2],
        c=clustering_result,
        cmap='tab10'
    )
    ax.set_xlabel('tsne-0')
    ax.set_ylabel('tsne-1')
    ax.set_zlabel('tsne-2')
    plt.savefig(f'{output_path}/time_series_clusterization_tsne_3_components')

    if clusters_to_select:
        selected_ids = []
        for index in range(clustering_result.shape[0]):
            cluster_index = clustering_result[index]
            if cluster_index not in clusters_to_select:
                clustering_result[index] = -2
            else:
                selected_ids.append(time_series_ids[index])
        save_json(f'{output_path}/selected_ids', {'selected_ids': selected_ids})

        pca_result_2['clusters'] = clustering_result
        plt.figure(figsize=(20, 20))
        sns.scatterplot(
            x="pca-0", y="pca-1",
            hue="clusters",
            palette=sns.color_palette("hls", np.unique(clustering_result).shape[0]),
            data=pca_result_2,
            legend="full",
            alpha=0.3
        )
        plt.savefig(f'{output_path}/time_series_clusterization_pca_2_components_selection')

        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(
            xs=pca_result_3[:, 0],
            ys=pca_result_3[:, 1],
            zs=pca_result_3[:, 2],
            c=clustering_result,
            cmap='tab10'
        )
        ax.set_xlabel('pca-0')
        ax.set_ylabel('pca-1')
        ax.set_zlabel('pca-2')
        plt.savefig(f'{output_path}/time_series_clusterization_pca_3_components_selection')

        tsne_result_2['clusters'] = clustering_result
        plt.figure(figsize=(20, 20))
        sns.scatterplot(
            x="tsne-0", y="tsne-1",
            hue="clusters",
            palette=sns.color_palette("hls", np.unique(clustering_result).shape[0]),
            data=tsne_result_2,
            legend="full",
            alpha=0.3
        )
        plt.savefig(f'{output_path}/time_series_clusterization_tsne_2_components_selection')

        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(
            xs=tsne_result_3[:, 0],
            ys=tsne_result_3[:, 1],
            zs=tsne_result_3[:, 2],
            c=clustering_result,
            cmap='tab10'
        )
        ax.set_xlabel('tsne-0')
        ax.set_ylabel('tsne-1')
        ax.set_zlabel('tsne-2')
        plt.savefig(f'{output_path}/time_series_clusterization_tsne_3_components_selection')


if __name__ == '__main__':
    args = parse_arguments()
    compute_attributions(args.experiment, args.data, args.output, args.device, args.debug)
    visualize_attributions(args.experiment, args.data, args.output, args.device, args.debug)
    clusterize_attributions(args.experiment, args.data, args.output, args.device, args.debug)
    example_explainability(args.experiment, args.output, args.device)
