import math
import os
import random
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import mean_squared_error

from app.auxiliary_files.other_methods.util_functions import save_json


def move_over_0(
        traces_values: List[np.ndarray],
        traces_predictions: List[np.ndarray],
):
    min_value = min([np.min(values) for values in traces_values])

    traces_predictions = [
        np.where(predictions > min_value, predictions, min_value)
        for predictions in traces_predictions
    ]

    if min_value <= 0:
        value_to_sum = abs(min_value) + 1e-10
        traces_values = [values + value_to_sum for values in traces_values]
        traces_predictions = [predictions + value_to_sum for predictions in traces_predictions]

    return traces_values, traces_predictions


def compute_mse(
        values: np.ndarray,
        predictions: np.ndarray,
        model_lag_size: int
):
    values = values[model_lag_size:]
    predictions = predictions[model_lag_size:]

    return mean_squared_error(values, predictions)


def find_spikes(
        values: np.ndarray,  # min > 0
        model_lag_size: int,
        spikes_window_size: int,
        spikes_std_factor: float
):
    def weighted_mean_and_std(values, weights):
        average = np.average(values, weights=weights)
        variance = np.average((values - average) ** 2, weights=weights)
        return average, math.sqrt(variance)

    spikes_position_indexes = []
    static_weights = np.arange(1 / spikes_window_size, 1 + 1 / spikes_window_size, 1 / spikes_window_size)

    last_spike_position = None
    for time_position in range(spikes_window_size + model_lag_size, values.shape[0]):
        value = values[time_position]
        if last_spike_position:
            window_size = min(spikes_window_size, time_position - last_spike_position)
        else:
            window_size = spikes_window_size
        window = values[(time_position - window_size):time_position]
        weights = static_weights[-window_size:]
        lag_mean, lag_std = weighted_mean_and_std(window, weights)
        if (value - lag_mean) > spikes_std_factor * lag_std:
            last_spike_position = time_position
            spikes_position_indexes.append(time_position)

    return spikes_position_indexes


def compute_over_provisioning(
        values: np.ndarray,  # min > 0
        predictions: np.ndarray,  # min > 0
        model_lag_size: int
):
    over_provisioning = sum(
        [
            (max(predictions[index] - values[index], 0) / values[index]) ** 2
            for index in range(model_lag_size, values.shape[0])
        ]
    ) / (values.shape[0] - model_lag_size)
    return over_provisioning


def compute_under_provisioning(
        values: np.ndarray,  # min > 0
        predictions: np.ndarray,  # min > 0
        model_lag_size: int
):
    under_provisioning = sum(
        [
            (min(predictions[index] - values[index], 0) / values[index]) ** 2
            for index in range(model_lag_size, values.shape[0])
        ]
    ) / (values.shape[0] - model_lag_size)
    return under_provisioning


def compute_spikes_precision(
        values: np.ndarray,  # min > 0
        predictions: np.ndarray,  # min > 0
        model_lag_size: int,
        spikes_window_size: int,
        spikes_std_factor: float,
        spikes_allowed_time_separation: int,
        spikes_allowed_over_factor: float,
        spikes_allowed_under_factor: float
):
    prediction_spike_list = find_spikes(
        predictions,
        model_lag_size,
        spikes_window_size,
        spikes_std_factor
    )

    close_prediction_spike_list = []
    for spike_position in prediction_spike_list:
        spike_value = predictions[spike_position]
        lookup_window = [
            max(spike_position - spikes_allowed_time_separation, 0, model_lag_size),  # init position
            min(spike_position + spikes_allowed_time_separation + 1, values.shape[0])  # end position (not included)
        ]
        found_closed = False
        index = lookup_window[0]
        while not found_closed and index < lookup_window[1]:
            value = values[index]
            found_closed = \
                (spike_value - spike_value * spikes_allowed_under_factor) < \
                value < \
                (spike_value + spike_value * spikes_allowed_over_factor)
            index += 1
        if found_closed:
            close_prediction_spike_list.append(spike_position)

    if len(prediction_spike_list) > 0:
        spikes_precision = len(close_prediction_spike_list) / len(prediction_spike_list)
    else:
        spikes_precision = 0

    return spikes_precision, prediction_spike_list, close_prediction_spike_list


def compute_spikes_recall(
        values: np.ndarray,  # min > 0
        predictions: np.ndarray,  # min > 0
        model_lag_size: int,
        spikes_window_size: int,
        spikes_std_factor: float,
        spikes_allowed_time_separation: int,
        spikes_allowed_over_factor: float,
        spikes_allowed_under_factor: float
):
    values_spike_list = find_spikes(
        values,
        model_lag_size,
        spikes_window_size,
        spikes_std_factor
    )

    correct_values_spike_list = []
    for spike_position in values_spike_list:
        spike_value = values[spike_position]
        lookup_window = [
            max(spike_position - spikes_allowed_time_separation, 0, model_lag_size),  # init position
            spike_position + 1  # end position (not included)
        ]
        found_closed = False
        index = lookup_window[0]
        while not found_closed and index < lookup_window[1]:
            value = predictions[index]
            found_closed = \
                (spike_value - spike_value * spikes_allowed_under_factor) < \
                value < \
                (spike_value + spike_value * spikes_allowed_over_factor)
            index += 1
        if found_closed:
            correct_values_spike_list.append(spike_position)

    if len(values_spike_list) > 0:
        spikes_recall = len(correct_values_spike_list) / len(values_spike_list)
    else:
        spikes_recall = 0

    return spikes_recall, values_spike_list, correct_values_spike_list


def compute_evaluation(
        title: str,
        id: str,
        values: np.ndarray,  # min > 0
        predictions: np.ndarray,  # min > 0
        output_path: str,
        model_lag_size: int,
        spikes_window_size: Optional[int] = 250,
        spikes_std_factor: Optional[float] = 2,
        spikes_allowed_time_separation: Optional[int] = 4,
        spikes_allowed_over_factor: Optional[float] = 0.6,
        spikes_allowed_under_factor: Optional[float] = 0.2,
        visualize: Optional[bool] = False,
        **visualize_args
):
    # over-provisioning
    over_provisioning = compute_over_provisioning(
        values,
        predictions,
        model_lag_size
    )

    # under-provisioning
    under_provisioning = compute_under_provisioning(
        values,
        predictions,
        model_lag_size
    )

    # spikes-precision
    spikes_precision, prediction_spike_list, close_prediction_spike_list = compute_spikes_precision(
        values,
        predictions,
        model_lag_size,
        spikes_window_size,
        spikes_std_factor,
        spikes_allowed_time_separation,
        spikes_allowed_over_factor,
        spikes_allowed_under_factor
    )

    # spikes-recall
    spikes_recall, values_spike_list, correct_values_spike_list = compute_spikes_recall(
        values,
        predictions,
        model_lag_size,
        spikes_window_size,
        spikes_std_factor,
        spikes_allowed_time_separation,
        spikes_allowed_over_factor,
        spikes_allowed_under_factor
    )

    # spikes-f1
    if spikes_precision > 0 or spikes_recall > 0:
        spikes_f1 = 2 * spikes_precision * spikes_recall / (spikes_precision + spikes_recall)
    else:
        spikes_f1 = 0

    # visualize
    if visualize:
        visualize_evaluation(
            title,
            id,
            values,
            predictions,
            prediction_spike_list,
            close_prediction_spike_list,
            values_spike_list,
            correct_values_spike_list,
            model_lag_size,
            spikes_window_size,
            output_path,
            **visualize_args
        )

    return over_provisioning, under_provisioning, spikes_precision, spikes_recall, spikes_f1


def compute_evaluation_list(
        title: str,
        traces_ids: List[str],
        traces_values: List[np.ndarray],
        traces_predictions: List[np.ndarray],
        output_path: str,
        model_lag_size: int,
        evaluation_args: Optional[dict] = {},
        visualize: Optional[int] = 2,
        visualize_args: Optional[dict] = {},
        save_metrics: Optional[bool] = True
):
    if visualize > 0:
        set_to_visualize = set(random.sample(traces_ids, min(visualize, len(traces_ids))))
    else:
        set_to_visualize = None

    # move over 0
    traces_values, traces_predictions = move_over_0(traces_values, traces_predictions)

    # compute metrics for all time series
    ids_list = []
    over_provisioning_list = []
    under_provisioning_list = []
    spikes_precision_list = []
    spikes_recall_list = []
    spikes_f1_list = []
    mse_list = []
    for time_series_id_index, time_series_id in enumerate(traces_ids):
        over_provisioning, under_provisioning, spikes_precision, spikes_recall, spikes_f1 = compute_evaluation(
            title,
            traces_ids[time_series_id_index],
            traces_values[time_series_id_index],
            traces_predictions[time_series_id_index],
            output_path,
            model_lag_size,
            **evaluation_args,
            visualize=True if set_to_visualize and time_series_id in set_to_visualize else False,
            **visualize_args
        )
        mse = compute_mse(
            traces_values[time_series_id_index],
            traces_predictions[time_series_id_index],
            model_lag_size
        )

        ids_list.append(time_series_id)
        over_provisioning_list.append(over_provisioning)
        under_provisioning_list.append(under_provisioning)
        spikes_precision_list.append(spikes_precision)
        spikes_recall_list.append(spikes_recall)
        spikes_f1_list.append(spikes_f1)
        mse_list.append(mse)

    # save results per time series
    if save_metrics:
        save_json(f'{output_path}/ids_results', ids_list)
        np.save(f'{output_path}/over_provisioning_results', np.asarray(over_provisioning_list))
        np.save(f'{output_path}/under_provisioning_results', np.asarray(under_provisioning_list))
        np.save(f'{output_path}/spikes_precision_results', np.asarray(spikes_precision_list))
        np.save(f'{output_path}/spikes_recall_results', np.asarray(spikes_recall_list))
        np.save(f'{output_path}/spikes_f1_results', np.asarray(spikes_f1_list))
        np.save(f'{output_path}/mse_results', np.asarray(mse_list))

    # average metrics to output only one value for each
    over_provisioning = sum(over_provisioning_list) / len(over_provisioning_list)
    under_provisioning = sum(under_provisioning_list) / len(under_provisioning_list)
    spikes_precision = sum(spikes_precision_list) / len(spikes_precision_list)
    spikes_recall = sum(spikes_recall_list) / len(spikes_recall_list)
    spikes_f1 = sum(spikes_f1_list) / len(spikes_f1_list)
    mse = sum(mse_list) / len(mse_list)

    return {
        'over-provisioning': over_provisioning,
        'under-provisioning': under_provisioning,
        'spikes_precision': spikes_precision,
        'spikes_recall': spikes_recall,
        'spikes_f1': spikes_f1,
        'mse': mse
    }


def visualize_evaluation(
        title: str,
        id: str,
        values: np.ndarray,
        predictions: np.ndarray,
        prediction_spike_list: List[int],
        close_prediction_spike_list: List[int],
        values_spike_list: List[int],
        correct_values_spike_list: List[int],
        model_lag_size: int,
        spikes_window_size: int,
        output_path: str,
        window_size: Optional[int] = 250,
        max_windows: Optional[int] = 4,
        save_visualization_info: Optional[bool] = False
):
    not_predicted_real_spikes = list(set(values_spike_list).difference(set(correct_values_spike_list)))
    not_related_predicted_spikes = list(set(prediction_spike_list).difference(set(close_prediction_spike_list)))

    if save_visualization_info:
        save_json(
            f'{output_path}/{title}_visualization_info_{id}',
            {
                'title': title,
                'id': id,
                'values': list(values),
                'predictions': list(predictions),
                'close_prediction_spike_list': close_prediction_spike_list,
                'not_predicted_real_spikes': not_predicted_real_spikes,
                'not_related_predicted_spikes': not_related_predicted_spikes,
                'correct_values_spike_list': correct_values_spike_list
            }
        )

    initial_positions = [
        initial_position
        for initial_position in range(model_lag_size + spikes_window_size, values.shape[0], window_size)
        if initial_position + window_size < values.shape[0]
    ]
    initial_positions = random.sample(initial_positions, max_windows)

    for initial_position in initial_positions:
        end_position = initial_position + window_size
        visualize_evaluation_window(
            title,
            id,
            values,
            predictions,
            initial_position,
            end_position,
            close_prediction_spike_list,
            not_predicted_real_spikes,
            not_related_predicted_spikes,
            correct_values_spike_list,
            output_path
        )


def visualize_evaluation_window(
        title: str,
        id: str,
        values: np.ndarray,
        predictions: np.ndarray,
        initial_position: int,
        end_position: int,
        close_prediction_spike_list: List[int],
        not_predicted_real_spikes: List[int],
        not_related_predicted_spikes: List[int],
        correct_values_spike_list: List[int],
        output_path: str
):
    fig, ax = plt.subplots(figsize=(40, 20))

    ax.plot(
        np.arange(initial_position, end_position),
        values[initial_position:end_position],
        color='#7eb3af',
        linewidth=9
    )
    ax.plot(
        np.arange(initial_position, end_position),
        predictions[initial_position:end_position],
        color='#3c474b',
        linewidth=9
    )
    ax.fill_between(
        np.arange(initial_position, end_position),
        values[initial_position:end_position],
        predictions[initial_position:end_position],
        where=[values[index] > predictions[index] for index in range(initial_position, end_position)],
        color='#3c474b',
        alpha=0.4,
        interpolate=True
    )
    ax.fill_between(
        np.arange(initial_position, end_position),
        values[initial_position:end_position],
        predictions[initial_position:end_position],
        where=[values[index] < predictions[index] for index in range(initial_position, end_position)],
        color='#7eb3af',
        alpha=0.4,
        interpolate=True
    )

    point_lists = [
            [
                'green',
                '#c0e0de',
                [
                    (values[time_position], time_position)
                    for time_position in correct_values_spike_list
                    if initial_position < time_position < end_position
                ]
            ],
            [
                'red',
                '#c0e0de',
                [
                    (values[time_position], time_position)
                    for time_position in not_predicted_real_spikes
                    if initial_position < time_position < end_position
                ]
            ],
            [
                'green',
                '#3c474b',
                [
                    (predictions[time_position], time_position)
                    for time_position in close_prediction_spike_list
                    if initial_position < time_position < end_position
                ]
            ],
            [
                'red',
                '#3c474b',
                [
                    (predictions[time_position], time_position)
                    for time_position in not_related_predicted_spikes
                    if initial_position < time_position < end_position
                ]
            ]
        ]
    for point_list in point_lists:
        color_face, color_edge, points = point_list
        for point in points:
            y, x = point
            ax.plot(
                x,
                y,
                marker='o',
                markersize=30,
                markeredgewidth=10,
                markerfacecolor=color_face,
                markeredgecolor=color_edge
            )

    plt.xlabel('time')
    plt.ylabel('CPU consumption')
    handles, labels = ax.get_legend_handles_labels()
    line_1 = Line2D([], [], label='real', color='#7eb3af', linewidth=9)
    line_2 = Line2D([], [], label='pred', color='#3c474b', linewidth=9)
    rec_1 = Patch(label='over-provisioning', color='#7eb3af', alpha=0.4)
    rec_2 = Patch(label='under-provisioning', color='#3c474b', alpha=0.4)
    point_1 = Line2D([], [], label='cor-pred-spikes', marker='o', linewidth=9, markersize=30, markeredgewidth=10, markerfacecolor='green', markeredgecolor='#c0e0de', color='#c0e0de')
    point_2 = Line2D([], [], label='¬cor-pred-spikes', marker='o', linewidth=9, markersize=30, markeredgewidth=10, markerfacecolor='red', markeredgecolor='#c0e0de', color='#c0e0de')
    point_3 = Line2D([], [], label='rel-pred-spikes', marker='o', linewidth=9, markersize=30, markeredgewidth=10, markerfacecolor='green', markeredgecolor='#3c474b', color='#3c474b')
    point_4 = Line2D([], [], label='¬rel-pred-spikes', marker='o', linewidth=9, markersize=30, markeredgewidth=10, markerfacecolor='red', markeredgecolor='#3c474b', color='#3c474b')
    handles.extend([line_1, line_2, rec_1, rec_2, point_1, point_2, point_3, point_4])
    ax.legend(handles=handles, loc='best', shadow=True)

    ax.get_xaxis().set_ticklabels([])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticklabels([])
    ax.get_yaxis().set_ticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.savefig(
        os.path.join(output_path, f'{title}_evaluation_visualization_{id}_window_{initial_position}_{end_position}'),
        bbox_inches='tight'
    )
