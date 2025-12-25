import logging
import random
from bisect import bisect
from typing import List

import numpy as np
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from app.auxiliary_files.data_methods.data_transformations import MeanStdIterativeComputation
from app.auxiliary_files.other_methods.util_functions import \
    print_pretty_json, \
    save_json
from app.auxiliary_files.other_methods.visualize import \
    plot_hist, \
    compare_multiple_lines_matrix, \
    compare_multiple_lines, \
    plot_correlation_matrix, \
    plot_bar

logger = logging.getLogger(__name__)


def perform_charts(config: dict, data_loader: DataLoader, features: List[str], output_path: str, data_source):
    charts = []
    for chart_config in config:
        name = chart_config['name']
        charts.append(eval(name)(chart_config, features, output_path, data_source))

    count = 0
    for (time_series_ids_list, time_series_times_list, time_series_values_list) in data_loader:
        logger.info(f'Iteration {count}')
        for chart in charts:
            chart.process_data(
                time_series_ids_list,
                time_series_times_list,
                time_series_values_list
            )
        count += len(time_series_ids_list)

    for chart in charts:
        logger.info(f'Showing {chart.__class__.__name__} results')
        chart.visualize()


class ChartGeneralLook:
    def __init__(self, config: dict, features: List[str], output_path, data_source):
        self.total_number = config['number']
        self.group_size = config['group_size']
        self.saved_time_series_ids_list = []
        self.saved_time_series_times_list = []
        self.saved_time_series_values_list = []
        self.output_path = output_path
        self.features = config['features']
        self.features_indexes = [features.index(feature) for feature in self.features]
        self.number = 0
        self.count = 0
        self.charts = []
        self.encoded_ids = {}

    def process_data(
            self,
            time_series_ids_list: List[int],
            time_series_times_list: List[np.ndarray],
            time_series_values_list: List[np.ndarray]
    ):
        left = self.total_number - len(self.saved_time_series_ids_list) - self.count
        if left > 0:
            self.saved_time_series_ids_list += time_series_ids_list[:left]
            self.saved_time_series_times_list += time_series_times_list[:left]
            self.saved_time_series_values_list += time_series_values_list[:left]

            if len(self.saved_time_series_ids_list) >= 10:  # for memory reasons
                self.visualize(private=True)
                self.saved_time_series_ids_list = []
                self.saved_time_series_times_list = []
                self.saved_time_series_values_list = []

    def visualize(self, private=False):
        for time_series_index, time_series_id in enumerate(self.saved_time_series_ids_list):
            self.encoded_ids[time_series_id] = self.count
            time_series_times = self.saved_time_series_times_list[time_series_index]
            time_series_values = self.saved_time_series_values_list[time_series_index]
            for feature_index, feature_label in enumerate(self.features):
                feature_index = self.features_indexes[feature_index]
                self.charts.append((
                    [(
                        time_series_values[feature_index, :],
                        time_series_times,
                        feature_label
                    )],
                    self.encoded_ids[time_series_id],
                    feature_label
                ))
            self.count += 1
            self.number += 1
            if self.number >= self.group_size:
                compare_multiple_lines_matrix(
                    False,
                    self.charts,
                    'Time series general look',
                    'values',
                    f'{self.output_path}/time_series_general_look_{self.count - self.number}-{self.count}',
                    ncols=len(self.features)
                )
                self.number = 0
                self.charts = []
        if self.number > 0 and not private:
            compare_multiple_lines_matrix(
                False,
                self.charts,
                'Time series general look',
                'values',
                f'{self.output_path}/time_series_general_look_{self.count - self.number}-{self.count}',
                ncols=len(self.features)
            )
        if not private:
            logger.info(f'Encoded chart ids: {self.encoded_ids}')


class ChartNullValues:
    def __init__(self, config: dict, features: List[str], output_path):
        self.features = features
        self.output_path = output_path
        self.null_values_dict = {}

        for feature in features:
            self.null_values_dict[feature] = {
                'total_values': 0,
                'total_null_values': 0,
                'time_series_with_null_values': set()
            }

    def process_data(
            self,
            time_series_ids_list: List[int],
            time_series_times_list: List[np.ndarray],
            time_series_values_list: List[np.ndarray]
    ):
        for time_series_index, time_series_id in enumerate(time_series_ids_list):
            time_series_values = time_series_values_list[time_series_index]
            for feature_index, feature_label in enumerate(self.features):
                feature_values = time_series_values[feature_index, :]
                null_values = np.count_nonzero(np.isnan(feature_values))
                self.null_values_dict[feature_label]['total_values'] += feature_values.shape[0]
                self.null_values_dict[feature_label]['total_null_values'] += null_values
                if null_values > 0:
                    self.null_values_dict[feature_label]['time_series_with_null_values'].add(time_series_id)

    def visualize(self):
        for feature in self.features:
            # make json serializable
            self.null_values_dict[feature]['time_series_with_null_values'] = len(list(self.null_values_dict[feature]['time_series_with_null_values']))

        print_pretty_json(self.null_values_dict)
        save_json(f'{self.output_path}/null_values', self.null_values_dict)

        plot_bar(
            False,
            f'{self.output_path}/null_values_absolute',
            f'Absolute null values',
            'features',
            'frequency',
            x=[feature for feature in self.features],
            height=[self.null_values_dict[feature]['total_null_values'] for feature in self.features]
        )

        plot_bar(
            False,
            f'{self.output_path}/null_values_percentage',
            f'Percentage null values',
            'features',
            'frequency',
            x=[feature for feature in self.features],
            height=[self.null_values_dict[feature]['total_null_values'] / self.null_values_dict[feature]['total_values'] for feature in self.features]
        )


class ChartMaxMinFeatures:
    def __init__(self, config: dict, features: List[str], output_path, data_source):
        self.features = features
        self.features_max_values = [None] * len(self.features)
        self.features_min_values = [None] * len(self.features)
        self.features_sum_values = [None] * len(self.features)
        self.features_count_values = [None] * len(self.features)

    def process_data(
            self,
            time_series_ids_list: List[int],
            time_series_times_list: List[np.ndarray],
            time_series_values_list: List[np.ndarray]
    ):
        for time_series_index, time_series_id in enumerate(time_series_ids_list):
            time_series_values = time_series_values_list[time_series_index]
            for feature_index, feature_label in enumerate(self.features):
                feature_values = time_series_values[feature_index, :]
                max_value = np.max(feature_values)
                min_value = np.min(feature_values)
                sum_value = np.sum(feature_values)
                count_value = feature_values.shape[0]

                if self.features_max_values[feature_index] is None:
                    self.features_max_values[feature_index] = max_value
                    self.features_min_values[feature_index] = min_value
                    self.features_sum_values[feature_index] = sum_value
                    self.features_count_values[feature_index] = count_value
                else:
                    self.features_max_values[feature_index] = max(max_value, self.features_max_values[feature_index])
                    self.features_min_values[feature_index] = min(min_value, self.features_min_values[feature_index])
                    self.features_sum_values[feature_index] += sum_value
                    self.features_count_values[feature_index] += count_value

    def visualize(self):
        max_values = {feature_label: self.features_max_values[feature_index]
                      for feature_index, feature_label in enumerate(self.features)}
        min_values = {feature_label: self.features_min_values[feature_index]
                      for feature_index, feature_label in enumerate(self.features)}
        average_values = {feature_label: self.features_sum_values[feature_index] / self.features_count_values[feature_index]
                      for feature_index, feature_label in enumerate(self.features)}
        logger.info(f'Max values: {max_values}')
        logger.info(f'Min values: {min_values}')
        logger.info(f'Average values: {average_values}')


class ChartClusterizedTimeSeries:
    def __init__(self, config: dict, features: List[str], output_path, data_source):
        self.chart_params = config['chart_params']
        self.feature = config['feature']
        self.feature_index = features.index(self.feature)
        self.time_series_ids = []
        self.time_series_characteristics = []
        self.output_path = output_path
        self.data_source = data_source

        if 'load' in config:
            self.load = True
            self.load_path = config['load']
        else:
            self.load = False

    def process_data(
            self,
            time_series_ids_list: List[int],
            time_series_times_list: List[np.ndarray],
            time_series_values_list: List[np.ndarray]
    ):
        if self.load:
            return

        import tsfel

        cfg = tsfel.get_features_by_domain()

        for time_series_index, time_series_id in enumerate(time_series_ids_list):
            full_time_series_values = time_series_values_list[time_series_index]
            full_time_series_values = full_time_series_values[self.feature_index]

            characteristics = []

            # extract full time series characteristics
            time_series_values = full_time_series_values

            # length
            length = time_series_values.shape[0]
            characteristics.append(length)

            # amplitude
            max_value = np.max(time_series_values)
            min_value = np.min(time_series_values)
            amplitude = max_value - min_value
            characteristics.append(amplitude)

            # upward / downward spikes
            upward_spikes = 0
            downward_spikes = 0
            mean = np.mean(time_series_values)
            std = np.std(time_series_values)
            for value_index in range(time_series_values.shape[0]):
                value = time_series_values[value_index]
                if abs(mean - value) > (1.5 * std):
                    if value > mean:
                        upward_spikes += 1
                    else:
                        downward_spikes += 1
            characteristics.append(upward_spikes)
            characteristics.append(downward_spikes)

            aux = tsfel.time_series_features_extractor(cfg, time_series_values, fs=1./(60 * 5))
            aux = list(aux.to_numpy()[0])
            characteristics += aux

            self.time_series_ids.append(time_series_id)
            self.time_series_characteristics.append(characteristics)

    def visualize(self):
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.cluster import KMeans
        import seaborn as sns
        import pandas as pd
        import random
        import matplotlib.pyplot as plt

        if not self.load:
            self.time_series_ids = np.asarray(self.time_series_ids)
            self.time_series_characteristics = np.asarray(self.time_series_characteristics)
            np.save(f'{self.output_path}/feature_extraction_ids', self.time_series_ids)
            np.save(f'{self.output_path}/feature_extraction_features', self.time_series_characteristics)
        else:
            self.time_series_ids = np.load(f'{self.load_path}/feature_extraction_ids.npy')
            self.time_series_characteristics = np.load(f'{self.load_path}/feature_extraction_features.npy')

        # standarized
        self.time_series_characteristics = (self.time_series_characteristics - self.time_series_characteristics.mean(axis=0)) / (self.time_series_characteristics.std(axis=0))

        # clusterization
        print('Counts of NaNs', np.count_nonzero(np.isnan(self.time_series_characteristics)))
        self.time_series_characteristics = np.nan_to_num(self.time_series_characteristics)
        pca = PCA(n_components=8)
        pca_result = pca.fit_transform(self.time_series_characteristics)
        cluster_algorithm = KMeans(n_clusters=10)
        clustering_result = cluster_algorithm.fit_predict(pca_result)
        clusters, counts = np.unique(clustering_result, return_counts=True)
        print(f'Number clusters: {clusters.shape[0]}. Clusters: {clusters}. Counts: {counts}')

        clusterized_time_series = {clusters[cluster_index]: [] for cluster_index in range(clusters.shape[0])}
        for time_series_index in range(self.time_series_characteristics.shape[0]):
            time_series_id = self.time_series_ids[time_series_index]
            time_series_cluster = clustering_result[time_series_index]
            clusterized_time_series[time_series_cluster].append(time_series_id)

        visualized_time_series = set()
        for cluster_index in range(clusters.shape[0]):
            examples_per_cluster = 3
            cluster = clusters[cluster_index]
            if len(clusterized_time_series[cluster]) < examples_per_cluster:
                print(f'Warning cluster {cluster} with less samples than examples to show')
                continue
            while examples_per_cluster > 0:
                random_index = random.randint(0, len(clusterized_time_series[cluster]) - 1)
                time_series_id = clusterized_time_series[cluster][random_index]
                if time_series_id not in visualized_time_series:
                    _, time_series_values = self.data_source.load_time_series(time_series_id)
                    time_series_values = time_series_values[self.feature_index]
                    compare_multiple_lines(
                        False,
                        [
                            [
                                time_series_values,
                                np.arange(time_series_values.shape[0]),
                                'feature'
                            ]
                        ],
                        'values',
                        'time',
                        f'Time series example of cluster {cluster}',
                        f'{self.output_path}/time_series_example_cluster_{cluster}_{examples_per_cluster}'
                    )
                    visualized_time_series.add(time_series_id)
                    examples_per_cluster -= 1

        # pca visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.time_series_characteristics)
        #clustering_result = cluster_algorithm.fit_predict(pca_result)
        pca_result = pd.DataFrame(pca_result, columns=["pca-0", "pca-1"])
        pca_result['clusters'] = clustering_result

        plt.figure(figsize=(20, 20))
        sns.scatterplot(
            x="pca-0", y="pca-1",
            hue="clusters",
            palette=sns.color_palette("hls", np.unique(clustering_result).shape[0]),
            data=pca_result,
            legend="full",
            alpha=0.3
        )
        plt.savefig(f'{self.output_path}/time_series_clusterization_pca_2_components')

        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(self.time_series_characteristics)
        #clustering_result = cluster_algorithm.fit_predict(pca_result)

        ax = plt.figure(figsize=(20, 20)).gca(projection='3d')
        ax.scatter(
            xs=pca_result[:, 0],
            ys=pca_result[:, 1],
            zs=pca_result[:, 2],
            c=clustering_result,
            cmap='tab10'
        )
        ax.set_xlabel('pca-0')
        ax.set_ylabel('pca-1')
        ax.set_zlabel('pca-2')
        plt.savefig(f'{self.output_path}/time_series_clusterization_pca_3_components')

        # t-sne visualization
        input_data = self.time_series_characteristics
        if input_data.shape[1] > 50:
            pca = PCA(n_components=40)
            input_data = pca.fit_transform(input_data)
        tsne = TSNE(n_components=2, verbose=1)
        tsne_result = tsne.fit_transform(input_data)
        #clustering_result = cluster_algorithm.fit_predict(tsne_result)
        tsne_result = pd.DataFrame(tsne_result, columns=["tsne-0", "tsne-1"])
        tsne_result['clusters'] = clustering_result

        plt.figure(figsize=(20, 20))
        sns.scatterplot(
            x="tsne-0", y="tsne-1",
            hue="clusters",
            palette=sns.color_palette("hls", np.unique(clustering_result).shape[0]),
            data=tsne_result,
            legend="full",
            alpha=0.3
        )
        plt.savefig(f'{self.output_path}/time_series_clusterization_tsne_2_components')

        tsne = TSNE(n_components=3, verbose=1)
        tsne_result = tsne.fit_transform(input_data)
        #clustering_result = cluster_algorithm.fit_predict(tsne_result)

        ax = plt.figure(figsize=(20, 20)).gca(projection='3d')
        ax.scatter(
            xs=tsne_result[:, 0],
            ys=tsne_result[:, 1],
            zs=tsne_result[:, 2],
            c=clustering_result,
            cmap='tab10'
        )
        ax.set_xlabel('tsne-0')
        ax.set_ylabel('tsne-1')
        ax.set_zlabel('tsne-2')
        plt.savefig(f'{self.output_path}/time_series_clusterization_tsne_3_components')


class ChartExtractTimeSeriesCluster:
    def __init__(self, config: dict, features: List[str], output_path, data_source):
        self.chart_params = config['chart_params']
        self.output_path = output_path
        self.data_source = data_source
        self.load_path = config['load']
        self.feature = config['feature']
        self.feature_index = features.index(self.feature)
        self.clustering_algorithm = config['clustering_algorithm']
        if 'clusters_to_select' in config:
            self.clusters_to_select = set(config['clusters_to_select'])
        else:
            self.clusters_to_select = None

    def process_data(
            self,
            time_series_ids_list: List[int],
            time_series_times_list: List[np.ndarray],
            time_series_values_list: List[np.ndarray]
    ):
        return

    def visualize(self):
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import seaborn as sns
        import pandas as pd
        import random
        import matplotlib.pyplot as plt

        time_series_ids = np.load(f'{self.load_path}/feature_extraction_ids.npy')
        time_series_characteristics = np.load(f'{self.load_path}/feature_extraction_features.npy')

        # standarized
        # time_series_characteristics = (time_series_characteristics - time_series_characteristics.mean(axis=0)) / (time_series_characteristics.std(axis=0))

        # clusterization
        print('Counts of NaNs', np.count_nonzero(np.isnan(time_series_characteristics)))
        time_series_characteristics = np.nan_to_num(time_series_characteristics)

        input_data = time_series_characteristics
        if input_data.shape[1] > 50:
            pca = PCA(n_components=40)
            input_data = pca.fit_transform(input_data)
        tsne = TSNE(n_components=2, verbose=1)
        input_data = tsne.fit_transform(input_data)

        cluster_algorithm = eval(self.clustering_algorithm['name'])(**self.clustering_algorithm['args'])
        clustering_result = cluster_algorithm.fit_predict(input_data)
        clusters, counts = np.unique(clustering_result, return_counts=True)
        print(f'Number clusters: {clusters.shape[0]}. Clusters: {clusters}. Counts: {counts}')

        clusterized_time_series = {clusters[cluster_index]: [] for cluster_index in range(clusters.shape[0])}
        for time_series_index in range(time_series_characteristics.shape[0]):
            time_series_id = time_series_ids[time_series_index]
            time_series_cluster = clustering_result[time_series_index]
            clusterized_time_series[time_series_cluster].append(time_series_id)

        visualized_time_series = set()
        for cluster_index in range(clusters.shape[0]):
            if cluster_index > 10:
                continue
            examples_per_cluster = 3
            cluster = clusters[cluster_index]
            if len(clusterized_time_series[cluster]) < examples_per_cluster:
                print(f'Warning cluster {cluster} with less samples than examples to show')
                continue
            while examples_per_cluster > 0:
                random_index = random.randint(0, len(clusterized_time_series[cluster]) - 1)
                time_series_id = clusterized_time_series[cluster][random_index]
                if time_series_id not in visualized_time_series:
                    _, time_series_values = self.data_source.load_time_series(time_series_id)
                    time_series_values = time_series_values[self.feature_index]
                    compare_multiple_lines(
                        False,
                        [
                            [
                                time_series_values,
                                np.arange(time_series_values.shape[0]),
                                'feature'
                            ]
                        ],
                        'values',
                        'time',
                        f'Time series example of cluster {cluster}',
                        f'{self.output_path}/time_series_example_cluster_{cluster}_{examples_per_cluster}'
                    )
                    visualized_time_series.add(time_series_id)
                    examples_per_cluster -= 1

        # pca visualization
        pca = PCA(n_components=2)
        pca_result_2 = pca.fit_transform(time_series_characteristics)
        #clustering_result = cluster_algorithm.fit_predict(pca_result)
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
        plt.savefig(f'{self.output_path}/time_series_clusterization_pca_2_components')

        pca = PCA(n_components=3)
        pca_result_3 = pca.fit_transform(time_series_characteristics)
        #clustering_result = cluster_algorithm.fit_predict(pca_result)

        ax = plt.figure(figsize=(20, 20)).gca(projection='3d')
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
        plt.savefig(f'{self.output_path}/time_series_clusterization_pca_3_components')

        # t-sne visualization
        input_data = time_series_characteristics
        if input_data.shape[1] > 50:
            pca = PCA(n_components=40)
            input_data = pca.fit_transform(input_data)
        tsne = TSNE(n_components=2, verbose=1)
        tsne_result_2 = tsne.fit_transform(input_data)
        #clustering_result = cluster_algorithm.fit_predict(tsne_result)
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
        plt.savefig(f'{self.output_path}/time_series_clusterization_tsne_2_components')

        tsne = TSNE(n_components=3, verbose=1)
        tsne_result_3 = tsne.fit_transform(input_data)
        #clustering_result = cluster_algorithm.fit_predict(tsne_result)

        ax = plt.figure(figsize=(20, 20)).gca(projection='3d')
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
        plt.savefig(f'{self.output_path}/time_series_clusterization_tsne_3_components')

        if self.clusters_to_select:
            selected_ids = []
            for index in range(clustering_result.shape[0]):
                cluster_index = clustering_result[index]
                if cluster_index not in self.clusters_to_select:
                    clustering_result[index] = -2
                else:
                    selected_ids.append(time_series_ids[index])
            save_json(f'{self.output_path}/selected_ids', {'selected_ids': selected_ids})

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
            plt.savefig(f'{self.output_path}/time_series_clusterization_pca_2_components_selection')

            ax = plt.figure(figsize=(20, 20)).gca(projection='3d')
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
            plt.savefig(f'{self.output_path}/time_series_clusterization_pca_3_components_selection')

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
            plt.savefig(f'{self.output_path}/time_series_clusterization_tsne_2_components_selection')

            ax = plt.figure(figsize=(20, 20)).gca(projection='3d')
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
            plt.savefig(f'{self.output_path}/time_series_clusterization_tsne_3_components_selection')


class ChartExtractTimeSeriesClusterPCA:
    def __init__(self, config: dict, features: List[str], output_path, data_source):
        self.chart_params = config['chart_params']
        self.output_path = output_path
        self.data_source = data_source
        self.load_path = config['load']
        self.feature = config['feature']
        self.number_components = config['number_components']
        self.feature_index = features.index(self.feature)
        self.clustering_algorithm = config['clustering_algorithm']
        if 'clusters_to_select' in config:
            self.clusters_to_select = set(config['clusters_to_select'])
        else:
            self.clusters_to_select = None

    def process_data(
            self,
            time_series_ids_list: List[int],
            time_series_times_list: List[np.ndarray],
            time_series_values_list: List[np.ndarray]
    ):
        return

    def visualize(self):
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        import seaborn as sns
        import pandas as pd
        import random
        import matplotlib.pyplot as plt

        time_series_ids = np.load(f'{self.load_path}/feature_extraction_ids.npy')
        time_series_characteristics = np.load(f'{self.load_path}/feature_extraction_features.npy')

        # NaNs imputation
        print(f'Number of features: {time_series_characteristics.shape[1]}')
        print('Counts of NaNs', np.count_nonzero(np.isnan(time_series_characteristics)))
        time_series_characteristics = np.nan_to_num(time_series_characteristics)

        # standarized
        # time_series_characteristics = (time_series_characteristics - time_series_characteristics.mean(axis=0)) / (time_series_characteristics.std(axis=0))

        # normalization
        sc = StandardScaler()
        time_series_characteristics = sc.fit_transform(time_series_characteristics)

        # show variance
        pca = PCA(n_components=time_series_characteristics.shape[1])
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
        plt.savefig(f'{self.output_path}/pca_features_variance')

        # transformation
        pca = PCA(n_components=self.number_components)
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
        plt.savefig(f'{self.output_path}/pca_selected_features_variance')

        # clusterization
        cluster_algorithm = eval(self.clustering_algorithm['name'])(**self.clustering_algorithm['args'])
        clustering_result = cluster_algorithm.fit_predict(input_data)
        clusters, counts = np.unique(clustering_result, return_counts=True)
        print(f'Number clusters: {clusters.shape[0]}. Clusters: {clusters}. Counts: {counts}')

        clusterized_time_series = {clusters[cluster_index]: [] for cluster_index in range(clusters.shape[0])}
        for time_series_index in range(time_series_characteristics.shape[0]):
            time_series_id = time_series_ids[time_series_index]
            time_series_cluster = clustering_result[time_series_index]
            clusterized_time_series[time_series_cluster].append(time_series_id)


        # visualization
        visualized_time_series = set()
        for cluster_index in range(clusters.shape[0]):
            if cluster_index > 20:
                continue
            examples_per_cluster = 3
            cluster = clusters[cluster_index]
            if len(clusterized_time_series[cluster]) < examples_per_cluster:
                print(f'Warning cluster {cluster} with less samples than examples to show')
                continue
            while examples_per_cluster > 0:
                random_index = random.randint(0, len(clusterized_time_series[cluster]) - 1)
                time_series_id = clusterized_time_series[cluster][random_index]
                if time_series_id not in visualized_time_series:
                    _, time_series_values = self.data_source.load_time_series(time_series_id)
                    time_series_values = time_series_values[self.feature_index]
                    compare_multiple_lines(
                        False,
                        [
                            [
                                time_series_values,
                                np.arange(time_series_values.shape[0]),
                                'feature'
                            ]
                        ],
                        'values',
                        'time',
                        f'Time series example of cluster {cluster}',
                        f'{self.output_path}/time_series_example_cluster_{cluster}_{examples_per_cluster}'
                    )
                    visualized_time_series.add(time_series_id)
                    examples_per_cluster -= 1

        # pca visualization
        pca = PCA(n_components=2)
        pca_result_2 = pca.fit_transform(time_series_characteristics)
        #clustering_result = cluster_algorithm.fit_predict(pca_result)
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
        plt.savefig(f'{self.output_path}/time_series_clusterization_pca_2_components')

        pca = PCA(n_components=3)
        pca_result_3 = pca.fit_transform(time_series_characteristics)
        #clustering_result = cluster_algorithm.fit_predict(pca_result)

        ax = plt.figure(figsize=(20, 20)).gca(projection='3d')
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
        plt.savefig(f'{self.output_path}/time_series_clusterization_pca_3_components')

        # t-sne visualization
        input_data = time_series_characteristics
        if input_data.shape[1] > 50:
            pca = PCA(n_components=40)
            input_data = pca.fit_transform(input_data)
        tsne = TSNE(n_components=2, verbose=1)
        tsne_result_2 = tsne.fit_transform(input_data)
        #clustering_result = cluster_algorithm.fit_predict(tsne_result)
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
        plt.savefig(f'{self.output_path}/time_series_clusterization_tsne_2_components')

        tsne = TSNE(n_components=3, verbose=1)
        tsne_result_3 = tsne.fit_transform(input_data)
        #clustering_result = cluster_algorithm.fit_predict(tsne_result)

        ax = plt.figure(figsize=(20, 20)).gca(projection='3d')
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
        plt.savefig(f'{self.output_path}/time_series_clusterization_tsne_3_components')

        if self.clusters_to_select:
            selected_ids = []
            for index in range(clustering_result.shape[0]):
                cluster_index = clustering_result[index]
                if cluster_index not in self.clusters_to_select:
                    clustering_result[index] = -2
                else:
                    selected_ids.append(time_series_ids[index])
            save_json(f'{self.output_path}/selected_ids', {'selected_ids': selected_ids})

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
            plt.savefig(f'{self.output_path}/time_series_clusterization_pca_2_components_selection')

            ax = plt.figure(figsize=(20, 20)).gca(projection='3d')
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
            plt.savefig(f'{self.output_path}/time_series_clusterization_pca_3_components_selection')

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
            plt.savefig(f'{self.output_path}/time_series_clusterization_tsne_2_components_selection')

            ax = plt.figure(figsize=(20, 20)).gca(projection='3d')
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
            plt.savefig(f'{self.output_path}/time_series_clusterization_tsne_3_components_selection')


class ChartAmplitude:
    def __init__(self, config: dict, features: List[str], output_path, data_source):
        self.chart_params = config['chart_params']
        self.interval_size = config['interval_size']
        self.intervals = config['intervals']
        self.feature = config['feature']
        self.feature_index = features.index(self.feature)
        self.amplitudes = []
        self.intervals_counts = []
        self.output_path = output_path

    def process_data(
            self,
            time_series_ids_list: List[int],
            time_series_times_list: List[np.ndarray],
            time_series_values_list: List[np.ndarray]
    ):
        for time_series_index, time_series_id in enumerate(time_series_ids_list):
            time_series_values = time_series_values_list[time_series_index]
            time_series_values = time_series_values[self.feature_index, :]
            max_value = np.max(time_series_values)
            min_value = np.min(time_series_values)
            amplitude = max_value - min_value
            self.amplitudes.append(amplitude)

            interval_counts = np.zeros(len(self.intervals))
            for value_index in range(time_series_values.shape[0]):
                value = time_series_values[value_index]
                interval_index = bisect(self.intervals, value)
                interval_counts[interval_index] += 1
            number_non_zero = np.count_nonzero(interval_counts)
            self.intervals_counts.append(number_non_zero)

    def visualize(self):
        plot_hist(
            False,
            f'{self.output_path}/amplitudes',
            f'Time series amplitudes of feature {self.feature}',
            'amplitudes',
            'frequency',
            **{'x': self.amplitudes, **self.chart_params}
        )
        counts, bins = np.histogram(self.amplitudes)
        print(f'Amplitudes. Counts: {counts}. Bins {bins}')
        number_intervals = np.divide(self.amplitudes, self.interval_size)
        plot_hist(
            False,
            f'{self.output_path}/number_intervals_raw',
            f'Time series raw number of intervals for feature {self.feature}',
            'number intervals',
            'frequency',
            **{'x': number_intervals, **self.chart_params}
        )
        counts, bins = np.histogram(self.intervals_counts, bins=list(range(1, len(self.intervals) + 1)))
        print(f'Raw number of intervals. Counts: {counts}. Bins {bins}')
        plot_hist(
            False,
            f'{self.output_path}/number_intervals_real',
            f'Time series real number of intervals for feature {self.feature}',
            'number intervals',
            'frequency',
            **{'x': self.intervals_counts, **self.chart_params}
        )
        counts, bins = np.histogram(self.intervals_counts, bins=list(range(1, len(self.intervals) + 1)))
        print(f'Real number of intervals. Counts: {counts}. Bins {bins}')


class ChartLengths:
    def __init__(self, config: dict, features: List[str], output_path, data_source):
        self.output_path = output_path
        self.lengths = []
        self.chart_params = config['chart_params']

    def process_data(
            self,
            time_series_ids_list: List[int],
            time_series_times_list: List[np.ndarray],
            time_series_values_list: List[np.ndarray]
    ):
        for time_series_index, time_series_id in enumerate(time_series_ids_list):
            time_series_times = time_series_times_list[time_series_index]
            self.lengths.append(time_series_times[-1] - time_series_times[0])

    def visualize(self):
        plot_hist(
            False,
            f'{self.output_path}/lengths',
            'Time series lengths',
            'lengths',
            'frequency',
            **{'x': self.lengths, **self.chart_params}
        )


class ChartDistributionValues:
    def __init__(self, config: dict, features: List[str], output_path):
        self.output_path = output_path
        self.total_number = config['number']
        self.chart_params = config['chart_params'] if 'chart_params' in config else None
        self.count = 0
        self.features = features + ['time']
        self.features_values = {}
        if 'log_transform' in config and config['log_transform']:
            self.log_transform = True
            self.log_transform_sum = config['log_transform_sum']
        else:
            self.log_transform = False

    def process_data(
            self,
            time_series_ids_list: List[int],
            time_series_times_list: List[np.ndarray],
            time_series_values_list: List[np.ndarray]
    ):
        left = self.total_number - self.count
        if left > 0:
            self.count += len(time_series_values_list)
            for time_series_index, time_series_values in enumerate(time_series_values_list):
                if self.log_transform:
                    time_series_values += np.ones((time_series_values.shape[0], time_series_values.shape[1])) * np.transpose(np.asarray([self.log_transform_sum]))
                for feature_index, feature_label in enumerate(self.features[:-1]):
                    feature_values = time_series_values[feature_index, :]
                    if self.log_transform:
                        feature_values = np.log(feature_values + 0.00000000001)
                    if feature_label in self.features_values:
                        self.features_values[feature_label] = np.concatenate((self.features_values[feature_label], feature_values))
                    else:
                        self.features_values[feature_label] = feature_values
                if 'time' in self.features_values:
                    self.features_values['time'] = np.concatenate((self.features_values['time'], time_series_times_list[time_series_index]))
                else:
                    self.features_values['time'] = time_series_times_list[time_series_index]

    def visualize(self):
        logger.info(f'Used number of values per feature: {[(feature, self.features_values[feature].shape[0]) for feature in self.features]}')

        for feature in self.features:
            if not np.isnan(self.features_values[feature]).all():
                plot_hist(
                    False,
                    f'{self.output_path}/distribution_values_{feature}',
                    '',
                    'values',
                    'frequency',
                    x=self.features_values[feature],
                    bins=30
                )


class ChartIntervalsDistribution:
    def __init__(self, config: dict, features: List[str], output_path, data_source):
        self.output_path = output_path
        self.chart_params = config['chart_params'] if 'chart_params' in config else None
        self.feature = config['target_feature']
        self.feature_index = features.index(self.feature)
        self.sum = config['targe_feature_sum']
        self.intervals = config['intervals']
        self.intervals_hist = [0] * len(self.intervals)

        if 'log_transform' in config and config['log_transform']:
            self.log_transform = True
        else:
            self.log_transform = False
        if self.log_transform:
            self.intervals = list(np.log(np.asarray(self.intervals) + 0.00000000001))

    def process_data(
            self,
            time_series_ids_list: List[int],
            time_series_times_list: List[np.ndarray],
            time_series_values_list: List[np.ndarray]
    ):
        for time_series_index, time_series_values in enumerate(time_series_values_list):
            feature_values = time_series_values[self.feature_index]
            for value_index in range(feature_values.shape[0]):
                value = feature_values[value_index] + self.sum
                if self.log_transform:
                    value = np.log(value + 0.00000000001)
                interval_index = bisect(self.intervals, value)
                self.intervals_hist[interval_index] += 1

    def visualize(self):
        logger.info(f'Histogram of values: {self.intervals_hist}')
        plot_bar(
            False,
            f'{self.output_path}/distribution_intervals_values_{self.feature}',
            f'Distribution of values in intervals',
            'intervals',
            'frequency',
            x=self.intervals,
            height=self.intervals_hist
        )


class ChartMeanStd:
    def __init__(self, config: dict, features: List[str], output_path):
        self.output_path = output_path
        self.computation = None
        self.features = features

    def process_data(
            self,
            time_series_ids_list: List[int],
            time_series_times_list: List[np.ndarray],
            time_series_values_list: List[np.ndarray]
    ):
        for time_series_values in time_series_values_list:
            if self.computation is None:
                self.computation = MeanStdIterativeComputation(time_series_values)
            else:
                self.computation.update(time_series_values)

    def visualize(self):
        mean, std = self.computation.finalize()
        logger.info(f'Means: {[(feature, mean[feature_index]) for feature_index, feature in enumerate(self.features)]}')
        logger.info(f'Stds: {[(feature, std[feature_index]) for feature_index, feature in enumerate(self.features)]}')


class ChartCorrelation:
    def __init__(self, config: dict, features: List[str], output_path):
        self.output_path = output_path
        self.features = features
        self.count = 0
        self.filtered_features = config['features'] if 'features' in config else None
        if self.filtered_features is not None:
            self.filtered_features_indexes = [features.index(feature) for feature in self.filtered_features]
            self.correlation_matrix = np.zeros((len(self.filtered_features), len(self.filtered_features)))
        else:
            self.filtered_features_indexes = None
            self.correlation_matrix = np.zeros((len(self.features), len(self.features)))

    def process_data(
            self,
            time_series_ids_list: List[int],
            time_series_times_list: List[np.ndarray],
            time_series_values_list: List[np.ndarray]
    ):
        self.count += len(time_series_values_list)
        for time_series_index, time_series_values in enumerate(time_series_values_list):
            if self.filtered_features_indexes is not None:
                time_series_values = np.take(time_series_values, self.filtered_features_indexes, axis=0)
            aux_matrix, _ = spearmanr(np.transpose(time_series_values))
            aux_matrix = np.nan_to_num(aux_matrix)  # Nan outputs when all values of a feature are constant
            self.correlation_matrix = np.add(self.correlation_matrix, aux_matrix)

    def visualize(self):
        self.correlation_matrix = np.divide(self.correlation_matrix, self.count, where=(self.correlation_matrix != 0))
        plot_correlation_matrix(
            False,
            self.correlation_matrix,
            self.filtered_features,
            self.output_path + '/compact_correlation'
        )


class ChartTimeDistanceBetweenSamples:
    def __init__(self, config: dict, features: List[str], output_path):
        self.output_path = output_path
        self.features = features
        self.time_distances_intervals = config['intervals']
        self.time_distances_counts = np.zeros(len(self.time_distances_intervals))

    def process_data(
            self,
            time_series_ids_list: List[int],
            time_series_times_list: List[np.ndarray],
            time_series_values_list: List[np.ndarray]
    ):
        for time_series_times in time_series_times_list:
            for time_index, time_value in enumerate(time_series_times[:-1]):
                time_distance = time_series_times[time_index + 1] - time_value
                if time_distance == 0:
                    position = 0
                else:
                    position = self.__binary_search(self.time_distances_intervals, 0, len(self.time_distances_intervals) - 1, time_distance)
                self.time_distances_counts[position] += 1

    def visualize(self):
        print(self.time_distances_counts)
        plot_bar(
            False,
            f'{self.output_path}/time_distances',
            'Time series time distances between samples',
            'time distances',
            'frequency',
            x=self.time_distances_intervals,
            height=self.time_distances_counts
        )

    def __binary_search(self, arr, low, high, x):
        # Check base case
        if high >= low:
            mid = (high + low) // 2
            # If element is present at the middle itself
            if arr[mid] == x:
                return mid
            # If element is smaller than mid, then it can only
            # be present in left subarray
            elif arr[mid] > x:
                return self.__binary_search(arr, low, mid - 1, x)
            # Else the element can only be present in right subarray
            else:
                return self.__binary_search(arr, mid + 1, high, x)
        else:
            # Element is not present in the array
            return high


def chart_autocorrelation(self, visualize: bool, config: dict, time_series_list: List[np.ndarray],
                          time_series_labels_list: List[str], output_path: str):
    number = config['number']
    if number == -1:
        chosen_time_series_list = time_series_list.copy()
    else:
        chosen_indexes = [random.randint(0, len(time_series_list) - 1) for _ in range(number)]
        chosen_time_series_list = []
        for chosen_index in chosen_indexes:
            chosen_time_series_list.append(time_series_list[chosen_index].copy())

    # normalize time series
    chosen_time_series_list, chosen_time_series_labels_list = self.preprocess([{'name': 'min_max_normalization'}],
                                                                              chosen_time_series_list)

    _type = config['type']

    if _type == 'compact':
        lines = []
        lags = config['lags']
        x = [str(index) for index in range(1, lags)]
        for time_series_index, time_series in enumerate(chosen_time_series_list):
            y = []
            for lag_index, lag in enumerate(range(1, lags)):
                aux1 = time_series[lag:, 0]
                aux2 = time_series[:(len(time_series) - lag), 0]
                if aux1.shape[0] > 0 < aux2.shape[0]:
                    autocorrelation, _ = spearmanr(aux1, aux2)
                    y.append(autocorrelation)
                else:
                    y.append(None)
            lines.append((y, x, 'time series ' + str(time_series_index)))

        compare_multiple_lines(visualize,
                               lines,
                               'autocorrelation',
                               'lags',
                               'Compact lag autocorrelation of target feature per time series',
                               output_path + '/lag_autocorrelation_compact_target'
                               )

        lags = config['lags']
        x = [str(index) for index in range(1, lags)]
        global_y = np.zeros((len(self.features_global), lags - 1))
        for time_series_index, time_series in enumerate(chosen_time_series_list):
            for lag_index, lag in enumerate(range(1, lags)):
                for feature_index in range(len(self.features_global)):
                    autocorrelation, _ = spearmanr(time_series[lag:, 0],
                                                   time_series[:(len(time_series) - lag), feature_index])
                    global_y[feature_index, lag_index] += autocorrelation
        global_y = np.divide(global_y, len(chosen_time_series_list))

        lines = [(global_y[feature_index], x, feature) for feature_index, feature in enumerate(self.features_global)]

        compare_multiple_lines(visualize,
                               lines,
                               'autocorrelation',
                               'lags',
                               'Compact lag autocorrelation with all features with the target one',
                               output_path + '/lag_autocorrelation_compact_features'
                               )

    elif _type == 'split':
        lags = config['lags']
        x = [str(index) for index in range(1, lags)]
        for time_series_index, time_series in enumerate(chosen_time_series_list):
            y = np.zeros((len(self.features_global), lags - 1))
            for lag_index, lag in enumerate(range(1, lags)):
                for feature_index in range(len(self.features_global)):
                    autocorrelation, _ = spearmanr(time_series[lag:, 0],
                                                   time_series[:(len(time_series) - lag), feature_index])
                    y[feature_index, lag_index] = autocorrelation
            lines = [(y[feature_index], x, feature) for feature_index, feature in enumerate(self.features_global)]

            compare_multiple_lines(visualize,
                                   lines,
                                   'autocorrelation',
                                   'lags',
                                   'Split lag autocorrelation with all features with the target one',
                                   output_path + '/lag_autocorrelation_split_features_time_series_' + str(
                                       time_series_index)
                                   )
    else:
        raise Exception('The type ' + _type + ' for chart autocorrelation does not exist')


def chart_preprocess(self, visualize: bool, config: dict, time_series_list: List[np.ndarray],
                     time_series_labels_list: List[str], output_path: str):
    time_series_list, time_series_labels_list = self.random_select_time_series(config['number'], time_series_list,
                                                                               time_series_labels_list)

    time_series_preprocessing_list = [time_series_list.copy()]
    preprocessing_labels = ['original']
    for preprocess_step_config in config['preprocessing_steps']:
        if preprocess_step_config['name'] not in ['filter_by_length', 'select']:
            preprocessing_labels.append(preprocess_step_config['name'])
            time_series_list, time_series_labels_list = self.preprocess([preprocess_step_config], time_series_list,
                                                                        time_series_labels_list)
            time_series_preprocessing_list.append(time_series_list.copy())

    _type = config['type']
    charts = []
    for time_series_index in range(len(time_series_preprocessing_list[0])):  # TODO refactor, error if empty
        for step_index, step_label in enumerate(preprocessing_labels):
            time_series = time_series_preprocessing_list[step_index][time_series_index]
            lines = []
            x = [index for index in range(time_series.shape[0])]
            if _type == 'all':
                for feature_index, feature_label in enumerate(self.features_global):
                    lines.append((time_series[:, feature_index], x, feature_label))
            elif _type == 'target':
                lines.append((time_series[:, 0], x, self.features_global[0]))
            else:
                raise Exception('Type ' + _type + ' not recognized')
            charts.append((lines, 'ts idx ' + str(time_series_labels_list[time_series_index]), step_label))

    # plot in groups
    groups_size = config['groups_size']
    for index in range(0, len(charts), groups_size * len(preprocessing_labels)):
        compare_multiple_lines_matrix(visualize,
                                      charts[index:(index + groups_size * len(preprocessing_labels))],
                                      'Examples of the preprocessing steps',
                                      'preprocessing step',
                                      output_path + '/preprocessing_examples_' + str(index) + '_' + _type,
                                      ncols=len(preprocessing_labels))
