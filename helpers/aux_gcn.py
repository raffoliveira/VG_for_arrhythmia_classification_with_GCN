import os
import pandas as pd
import numpy as np
import igraph as ig
import torch as th
import dgl
import torch.nn.functional as F
import seaborn as sns
import networkx as nx
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from typing import Type, Tuple, Any
from collections import defaultdict, Counter
from dgl.dataloading import GraphDataLoader
from scipy.io import loadmat
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import classification_report, confusion_matrix
from ts2vg import NaturalVG
from ts2vvg.graph import build_graph
from helpers.gcn import GCN2, GCN7, GCN60, GCN120, GCN240


class GCNFunctions:

    def __init__(self) -> None:
        pass

    def segmentation_signals_v1_ii(
        self,
        path: str,
        list_ecgs: list,
        size_beat_before: int,
        size_beat_after: int,
        set_name: str,
    ) -> Tuple[defaultdict, defaultdict]:
        """
        Responsible for segmenting ECG signals into beats"

        Args:
            path: path of ECG signals
            list_ecgs: name of each file of ECG signal
            size_beat_before: amount of points before peak of each beat
            size_beat_after: amount of points after peak of each beat
            set_name: name of set beats

        Returns:
           Two dictionaries: beats v1 and ii
        """

        dict_signals_v1 = defaultdict(list)
        dict_signals_ii = defaultdict(list)

        for file in list_ecgs:

            struct = loadmat(os.path.join(path, set_name, file))
            data = struct["individual"][0][0]
            ecg_v1 = data["signal_r"][:, 1]  # reading lead v1
            ecg_ii = data["signal_r"][:, 0]  # reading lead ii

            if file == "114.mat":
                ecg_v1 = data["signal_r"][:, 0]
                ecg_ii = data["signal_r"][:, 1]

            beat_peaks = data["anno_anns"]  # reading R-peak
            beat_types = data["anno_type"]  # reading type of beat

            for peak, beat_type in zip(beat_peaks, beat_types):

                beat_samples_v1 = []
                beat_samples_ii = []  #

                if (peak - size_beat_before) < 0 or (peak + size_beat_after) > len(
                    ecg_v1
                ):
                    continue

                if beat_type not in "NLRejAaJSVEFP/fUQ":
                    continue

                beat_samples_v1 = ecg_v1[
                    int(peak - size_beat_before): int(peak + size_beat_after)
                ]
                beat_samples_ii = ecg_ii[
                    int(peak - size_beat_before): int(peak + size_beat_after)
                ]

                if beat_type in "NLRej":
                    dict_signals_v1["N"].append(beat_samples_v1)
                    dict_signals_ii["N"].append(beat_samples_ii)
                elif beat_type in "AaJS":
                    dict_signals_v1["S"].append(beat_samples_v1)
                    dict_signals_ii["S"].append(beat_samples_ii)
                elif beat_type in "VE":
                    dict_signals_v1["V"].append(beat_samples_v1)
                    dict_signals_ii["V"].append(beat_samples_ii)

        return dict_signals_v1, dict_signals_ii

    def segmentation_signals_v1_ii_rr(
        self,
        path: str,
        list_ecgs: list,
        size_beat_before: int,
        size_beat_after: int,
        set_name: str
    ) -> Tuple[defaultdict, defaultdict, defaultdict, defaultdict]:
        """
        Responsible for segmenting ECG signals into beats"

        Args:
            path: path of ECG signals
            list_ecgs: name of each file of ECG signal
            size_beat_before: amount of points before peak of each beat
            size_beat_after: amount of points after peak of each beat
            set_name: name of set beats

        Returns:
            Four dictionaries: beats v1, beats ii, pre rr_interval and pos rr_interval
        """

        dict_signals_v1 = defaultdict(list)
        dict_signals_ii = defaultdict(list)
        rr_interval_pos_signals = defaultdict(list)
        rr_interval_pre_signals = defaultdict(list)
        rr_interval_pos = 0
        rr_interval_pre = 0

        for file in list_ecgs:

            struct = loadmat(os.path.join(path, set_name, file))
            data = struct["individual"][0][0]

            beat_peaks = data["anno_anns"]
            beat_types = data["anno_type"]

            ecg_v1 = data["signal_r"][:, 1]
            ecg_ii = data["signal_r"][:, 0]

            if file == '114.mat':
                ecg_v1 = data['signal_r'][:, 0]
                ecg_ii = data['signal_r'][:, 1]

            ecg_ii = self.__standardization(data=ecg_ii)
            ecg_v1 = self.__standardization(data=ecg_v1)

            for it, (peak, beat_type) in enumerate(zip(beat_peaks, beat_types)):

                beat_samples_v1 = []
                beat_samples_ii = []

                if (peak - size_beat_before) < 0 or (peak + size_beat_after) > len(ecg_ii):
                    continue

                if beat_type not in "NLRejAaJSVEFP/fUQ":
                    continue

                beat_samples_ii = ecg_ii[int(peak - size_beat_before): int(peak + size_beat_after)]
                beat_samples_v1 = ecg_v1[int(peak - size_beat_before): int(peak + size_beat_after)]

                if it-1 > 0:
                    rr_interval_pre = (peak - beat_peaks[it-1])[0]
                if it+1 < len(beat_peaks):
                    rr_interval_pos = (beat_peaks[it+1] - peak)[0]

                if beat_type in "NLRej":
                    dict_signals_v1["N"].append(beat_samples_v1)
                    dict_signals_ii["N"].append(beat_samples_ii)
                    rr_interval_pos_signals["N"].append(rr_interval_pos)
                    rr_interval_pre_signals["N"].append(rr_interval_pre)
                elif beat_type in "AaJS":
                    dict_signals_v1["S"].append(beat_samples_v1)
                    dict_signals_ii["S"].append(beat_samples_ii)
                    rr_interval_pos_signals["S"].append(rr_interval_pos)
                    rr_interval_pre_signals["S"].append(rr_interval_pre)
                elif beat_type in "VE":
                    dict_signals_v1["V"].append(beat_samples_v1)
                    dict_signals_ii["V"].append(beat_samples_ii)
                    rr_interval_pos_signals["V"].append(rr_interval_pos)
                    rr_interval_pre_signals["V"].append(rr_interval_pre)

        return dict_signals_v1, dict_signals_ii, rr_interval_pos_signals, rr_interval_pre_signals

    def __standardization(self, data: np.ndarray) -> np.ndarray:
        """
        Scaling centered around mean with a unit standard deviation

        Args:
            data: time serie to scale

        Returns:
           Transformed time serie 
        """
        data = np.nan_to_num(data)
        data = data - np.mean(data)
        data = data / np.std(data)
        return data

    def sampling_windows_beats(self, signals: defaultdict) -> defaultdict:
        """
        Sampling of beats from class N

        Args:
            signals: Dictionary of beats

        Returns:
            Same dictionary with sampled class N
        """

        select_beats = []
        for index, beat in enumerate(signals["N"], 1):
            if (index % 10) == 0:
                select_beats.append(beat)

        signals["N"] = select_beats

        return signals

    def sampling_windows_beats_signals(
        self,
        signals_v1: defaultdict,
        signals_ii: defaultdict,
        rr_interval_pos_signals: defaultdict,
        rr_interval_pre_signals: defaultdict
    ) -> Tuple[defaultdict, defaultdict, defaultdict, defaultdict]:
        """
        Sampling of beats from class N

        Args:
            signals_v1: Dictionary of beats v1
            signals_ii: Dictionary of beats ii
            rr_interval_pos_signals: Dictionary of pos rr_interval
            rr_interval_pre_signals: Dictionary of pre rr_interval

        Returns:
            Same argument dictionaries with sampled class N
        """

        select_n_beats_v1 = []
        select_n_beats_i = []
        select_rr_interval_pos_signals = []
        select_rr_interval_pre_signals = []

        for index, (beat_v1, beat_ii, rr_pos, rr_pre) in enumerate(
                zip(signals_v1["N"], signals_ii["N"], rr_interval_pos_signals["N"], rr_interval_pre_signals["N"]), 1):
            if (index % 10) == 0:
                select_n_beats_v1.append(beat_v1)
                select_n_beats_i.append(beat_ii)
                select_rr_interval_pos_signals.append(rr_pos)
                select_rr_interval_pre_signals.append(rr_pre)

        signals_v1["N"] = select_n_beats_v1
        signals_ii["N"] = select_n_beats_i
        rr_interval_pos_signals["N"] = select_rr_interval_pos_signals
        rr_interval_pre_signals["N"] = select_rr_interval_pre_signals

        return signals_v1, signals_ii, rr_interval_pos_signals, rr_interval_pre_signals

    def resampling_intra_patient(self, dataset_train: defaultdict, dataset_test: defaultdict) -> Tuple[defaultdict, defaultdict]:
        """
        Resampling the datasets, choosing the data in a random way

        Args:
            dataset_train: training data
            dataset_test: testing data

        Returns:
            Resampled datasets
        """

        classes = ['N', 'S', 'V']
        split_num = [45847, 944, 3788]

        for idx, class_ in enumerate(classes):
            total = dataset_train[class_] + dataset_test[class_]
            np.random.shuffle(total)
            dataset_train[class_] = total[:split_num[idx]]
            dataset_test[class_] = total[split_num[idx]:]

        return dataset_train, dataset_test

    def get_beats_features(self, signals_v1: dict, signals_ii: dict) -> dict:
        """
        Extract three features of the beats: time, point of beat ii and point of beat v1

        Args:
            signals_v1: Dictionary of beats v1
            signals_ii: Dictionary of beats ii

        Returns:
            Dictionary with extracted information
        """

        features: dict = {}
        nodes_feat: list = []
        graph_it = 0

        # extract features: time, beat_ii and beat_v1
        for (_, beats_v1), (_, beats_ii) in zip(signals_v1.items(), signals_ii.items()):
            for beat_v1, beat_ii in zip(beats_v1, beats_ii):
                nodes_feat = []
                for i, (j, k) in enumerate(zip(beat_ii, beat_v1)):
                    nodes_feat.append(minmax_scale([i, j, k]))

                features.update({graph_it: nodes_feat})
                graph_it += 1

        return features

    def get_beats_features_avg(
        self,
        signals_v1: defaultdict,
        signals_ii: defaultdict,
        rr_interval_pos_signals: defaultdict,
        rr_interval_pre_signals: defaultdict
    ) -> dict:
        """
        Extract seven features of the beats: time, point of beat ii, point of beat v1, pre_rr_interval, pos_rr_interval,
        difference between beat points v1 and ii, and division the points of beat v1 by the average of points of beat ii

        Args:
            signals_v1: Dictionary of beats v1
            signals_ii: Dictionary of beats ii

        Returns:
            Dictionary with extracted information
        """

        features = {}
        nodes_feat = []
        graph_it = 0

        for (class_, beats_v1), (_, beats_ii) in zip(signals_v1.items(), signals_ii.items()):
            for it, (beat_v1, beat_ii) in enumerate(zip(beats_v1, beats_ii)):
                for i, (j, k) in enumerate(zip(beat_ii, beat_v1)):
                    features_ = [
                        i,
                        j,
                        k,
                        rr_interval_pos_signals[class_][it],
                        rr_interval_pre_signals[class_][it],
                        round(j-k, 3),
                        round(k/np.mean(beat_ii), 3)
                    ]
                    nodes_feat.append(minmax_scale(features_))
                    features.update({graph_it: nodes_feat})
                graph_it += 1
                nodes_feat = []

        return features

    def get_beats_features_dif(
        self,
        signals_v1: defaultdict,
        signals_ii: defaultdict,
        rr_interval_pos_signals: defaultdict,
        rr_interval_pre_signals: defaultdict
    ) -> dict:
        """
        Extract six features of the beats: time, point of beat ii, point of beat v1, pre_rr_interval, pos_rr_interval,
        and difference between beat points v1 and ii

        Args:
            signals_v1: Dictionary of beats v1
            signals_ii: Dictionary of beats ii

        Returns:
            Dictionary with extracted information
        """

        features = {}
        nodes_feat = []
        graph_it = 0

        for (class_, beats_v1), (_, beats_ii) in zip(signals_v1.items(), signals_ii.items()):
            for it, (beat_v1, beat_ii) in enumerate(zip(beats_v1, beats_ii)):
                for i, (j, k) in enumerate(zip(beat_ii, beat_v1)):
                    features_ = [
                        i,
                        j,
                        k,
                        rr_interval_pos_signals[class_][it],
                        rr_interval_pre_signals[class_][it],
                        round(j-k, 3)
                    ]
                    nodes_feat.append(minmax_scale(features_))
                    features.update({graph_it: nodes_feat})
                graph_it += 1
                nodes_feat = []

        return features

    def get_beats_features_rr(
        self,
        signals_v1: defaultdict,
        signals_ii: defaultdict,
        rr_interval_pos_signals: defaultdict,
        rr_interval_pre_signals: defaultdict
    ) -> dict:
        """
        Extract five features of the beats: time, point of beat ii, point of beat v1, pre_rr_interval, pos_rr_interval

        Args:
            signals_v1: Dictionary of beats v1
            signals_ii: Dictionary of beats ii

        Returns:
            Dictionary with extracted information
        """

        features = {}
        nodes_feat = []
        graph_it = 0

        for (class_, beats_v1), (_, beats_ii) in zip(signals_v1.items(), signals_ii.items()):
            for it, (beat_v1, beat_ii) in enumerate(zip(beats_v1, beats_ii)):
                for i, (j, k) in enumerate(zip(beat_ii, beat_v1)):
                    features_ = [
                        i,
                        j,
                        k,
                        rr_interval_pos_signals[class_][it],
                        rr_interval_pre_signals[class_][it]
                    ]
                    nodes_feat.append(minmax_scale(features_))
                    features.update({graph_it: nodes_feat})
                graph_it += 1
                nodes_feat = []

        return features

    def get_beats_features_std(
        self,
        signals_v1: defaultdict,
        signals_ii: defaultdict,
        rr_interval_pos_signals: defaultdict,
        rr_interval_pre_signals: defaultdict
    ) -> dict:
        """
        Extract seven features of the beats: time, point of beat ii, point of beat v1, pre_rr_interval, pos_rr_interval,
        difference between beat points v1 and ii, division the points of beat v1 by the average of points of beat ii and
        division the points of beat v1 by the standard deviation of points of beat ii

        Args:
            signals_v1: Dictionary of beats v1
            signals_ii: Dictionary of beats ii

        Returns:
            Dictionary with extracted information
        """

        features = {}
        nodes_feat = []
        graph_it = 0

        for (class_, beats_v1), (_, beats_ii) in zip(signals_v1.items(), signals_ii.items()):
            for it, (beat_v1, beat_ii) in enumerate(zip(beats_v1, beats_ii)):
                for i, (j, k) in enumerate(zip(beat_ii, beat_v1)):
                    features_ = [
                        i,
                        j,
                        k,
                        rr_interval_pos_signals[class_][it],
                        rr_interval_pre_signals[class_][it],
                        round(j-k, 3),
                        round(k/np.mean(beat_ii), 3),
                        round(k/np.std(beat_ii), 3)
                    ]
                    nodes_feat.append(minmax_scale(features_))
                    features.update({graph_it: nodes_feat})
                graph_it += 1
                nodes_feat = []

        return features

    def get_beats_features_stats(
        self,
        signals_v1: defaultdict,
        signals_ii: defaultdict,
        rr_interval_pos_signals: defaultdict,
        rr_interval_pre_signals: defaultdict
    ) -> dict:
        """
        Extract twenty-two features of the beats: time, point of beat ii, point of beat v1, pre_rr_interval, pos_rr_interval,
        difference between beat points v1 and ii, division the points of beat v1 by the average of points of beat ii,
        division the points of beat v1 by the standard deviation of points of beat ii, entropy, 5-percentile, 
        25-percentile, 75-percentile, 95-percentile, median, mean, standard deviation, variance, rms, kurtosis, 
        skewness, zero_crossings, mean_crossings

        Args:
            signals_v1: Dictionary of beats v1
            signals_ii: Dictionary of beats ii

        Returns:
            Dictionary with extracted information
        """

        features = {}
        nodes_feat = []
        graph_it = 0

        for (class_, beats_v1), (_, beats_ii) in zip(signals_v1.items(), signals_ii.items()):
            for it, (beat_v1, beat_ii) in enumerate(zip(beats_v1, beats_ii)):
                for i, (j, k) in enumerate(zip(beat_ii, beat_v1)):
                    features_ = [
                        i,
                        j,
                        k,
                        rr_interval_pos_signals[class_][it],
                        rr_interval_pre_signals[class_][it],
                        round(j-k, 3),
                        round(k/np.mean(beat_ii), 3),
                        round(k/np.std(beat_ii), 3)
                    ] + self.__get_features(beat=beat_ii)
                    nodes_feat.append(minmax_scale(features_))
                    features.update({graph_it: nodes_feat})
                graph_it += 1
                nodes_feat = []

        return features

    def __calculate_entropy(self, beat: list):
        """
        Calculate the entropy of a beat

        Args:
            beat: list of points

        Returns:
            The entropy
        """
        counter_values = Counter(beat).most_common()
        probabilities = [elem[1]/len(beat) for elem in counter_values]
        entropy = sp.stats.entropy(probabilities)
        return [round(entropy, 3)]

    def __calculate_statistics(self, beat: list):
        """
        Calculate several statistics metrics

        Args:
            beat: list of points

        Returns:
            the metrics 
        """
        n5 = round(np.nanpercentile(beat, 5), 3)
        n25 = round(np.nanpercentile(beat, 25), 3)
        n75 = round(np.nanpercentile(beat, 75), 3)
        n95 = round(np.nanpercentile(beat, 95), 3)
        median = round(np.nanpercentile(beat, 50), 3)
        mean = round(np.nanmean(beat), 3)
        std = round(np.nanstd(beat), 3)
        var = round(np.nanvar(beat), 3)
        rms = round(np.nanmean(np.sqrt(beat**2)), 3)
        kurtosis = round(stats.kurtosis(beat), 3)
        skewness = round(stats.skew(beat), 3)
        return [n5, n25, n75, n95, median, mean, std, var, rms, kurtosis, skewness]

    def __calculate_crossings(self, beat: list):
        """
        Calculate some crossings values

        Args:
            beat: list of points

        Returns:
            crossing values
        """
        # number of times a signal crosses y = 0
        # number of times a signal crosses y = mean(y)
        zero_crossing_indices = np.nonzero(np.diff(np.array(beat) > 0))[0]
        no_zero_crossings = len(zero_crossing_indices)
        mean_crossing_indices = np.nonzero(np.diff(np.array(beat) > np.nanmean(beat)))[0]
        no_mean_crossings = len(mean_crossing_indices)
        return [round(no_zero_crossings, 3), round(no_mean_crossings, 3)]

    def __get_features(self, beat: list):
        """
        Calculate entropy, crossings and statistics metrics

        Args:
            beat: list of points

        Returns:
            the calculated metrics
        """
        entropy = self.__calculate_entropy(beat)
        crossings = self.__calculate_crossings(beat)
        statistics = self.__calculate_statistics(beat)
        return entropy + crossings + statistics

    def convert_beats_in_graphs(self, signals: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert beats into graphs

        Args:
            signals: Dictionary of beats

        Returns:
            Two dataframes, one with edges information and the other with the properties of each generated graph
        """

        aux_graph = {
            "graph_id": [],
            "graph_src": [],
            "graph_dst": [],
            "graph_nodes": [],
            "graph_label": []
        }
        classes = ["N", "S", "V"]
        graph_it = 0

        for class_, beats in signals.items():
            for beat in beats:
                label = classes.index(class_)
                g = NaturalVG(directed=None).build(beat)
                graph = g.as_igraph()
                num_nodes = graph.vcount()

                aux_graph.get("graph_id").extend([graph_it] * len(ig.Graph.get_edgelist(graph)))
                aux_graph.get("graph_src").extend([i[0] for i in ig.Graph.get_edgelist(graph)])
                aux_graph.get("graph_dst").extend([i[1] for i in ig.Graph.get_edgelist(graph)])
                aux_graph.get("graph_nodes").extend([num_nodes] * len(ig.Graph.get_edgelist(graph)))
                aux_graph.get("graph_label").extend([label] * len(ig.Graph.get_edgelist(graph)))

                graph_it += 1
                del graph

        edges = pd.DataFrame(
            {
                "graph_id": aux_graph.get("graph_id"),
                "src": aux_graph.get("graph_src"),
                "dst": aux_graph.get("graph_dst")
            }
        )
        properties = pd.DataFrame(
            {
                "graph_id": aux_graph.get("graph_id"),
                "label": aux_graph.get("graph_label"),
                "num_nodes": aux_graph.get("graph_nodes")
            }
        )
        properties.drop_duplicates(inplace=True)

        return edges, properties

    def convert_beats_in_graphs_vvg(self, signals_v1: dict, signals_ii: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert beats into graphs using VVG

        Args:
            signals_v1: Dictionary of beats V1
            signals_ii: Dictionary of beats ii

        Returns:
            Two dataframes, one with edges information and the other with the properties of each generated graph
        """

        aux_graph = {
            "graph_id": [],
            "graph_src": [],
            "graph_dst": [],
            "graph_nodes": [],
            "graph_label": []
        }
        classes = ["N", "S", "V"]
        graph_it = 0

        for (class_, beats_v1), (_, beats_ii) in zip(signals_v1.items(), signals_ii.items()):
            for beat_v1, beat_ii in zip(beats_v1, beats_ii):
                label = classes.index(class_)
                adjlist_ = build_graph(series=(beat_v1, beat_ii))
                graph = nx.DiGraph(adjlist_)
                num_nodes = nx.number_of_nodes(graph)

                aux_graph.get("graph_id").extend([graph_it] * len(nx.to_edgelist(graph)))
                aux_graph.get("graph_src").extend([i[0] for i in nx.to_edgelist(graph)])
                aux_graph.get("graph_dst").extend([i[1] for i in nx.to_edgelist(graph)])
                aux_graph.get("graph_nodes").extend([num_nodes] * len(nx.to_edgelist(graph)))
                aux_graph.get("graph_label").extend([label] * len(nx.to_edgelist(graph)))

                graph_it += 1
                del graph

        edges = pd.DataFrame(
            {
                "graph_id": aux_graph.get("graph_id"),
                "src": aux_graph.get("graph_src"),
                "dst": aux_graph.get("graph_dst")
            }
        )
        properties = pd.DataFrame(
            {
                "graph_id": aux_graph.get("graph_id"),
                "label": aux_graph.get("graph_label"),
                "num_nodes": aux_graph.get("graph_nodes")
            }
        )
        properties.drop_duplicates(inplace=True)

        return edges, properties

    def __plotting_acc_loss(
        self, acc_values: list, loss_values: list, epochs: int, path: str
    ) -> None:
        """
        Plots the accuracy curve and loss function

        Args:
            acc_values: list of accuracy values
            loss_values: list of loss values
            epochs: value of the number of epochs used during training
            path: path to save the figure
        """

        _, ax = plt.subplots(1, 2, figsize=(12, 6))
        plt.suptitle("Treinamento", fontsize=20)
        ax[0].plot(range(1, epochs + 1), acc_values, "r")
        ax[1].plot(range(1, epochs + 1), loss_values, "b")
        ax[0].set_title("Acurácia")
        ax[1].set_title("Perda")
        ax[0].set_xlabel("Épocas")
        ax[1].set_xlabel("Épocas")
        ax[0].set_ylabel("Acurácia")
        ax[1].set_ylabel("Perda")
        plt.savefig(path, dpi=600)

    def __plotting_confusion_matrix(
        self, true_label: np.array, pred_label: np.array, path: str
    ) -> None:
        """
        Plots the confusion matrix

        Args:
            true_label: array with truth ground values
            pred_label: array with predicted values
            path: path to save the figure
        """

        _, ax = plt.subplots(figsize=(20, 9))
        matrix = confusion_matrix(true_label, pred_label)

        sns.heatmap(
            matrix,
            ax=ax,
            xticklabels=["N", "S", "V"],
            yticklabels=["N", "S", "V"],
            annot=True,
            fmt=".0f",
            cmap="rocket_r",
        )

        ax.set_title("Matriz de Confusão", fontsize=18)
        ax.set_xlabel("Predição", fontsize=16)
        ax.set_ylabel("Verdadeiro", fontsize=16)
        plt.savefig(path, dpi=600)

    def __getting_classification_report(
        self, true_label: np.array, pred_label: np.array, set_name: str, file_path: str
    ):
        """
        Create the file and save the classification report

        Args:
            true_label: array with truth ground values
            pred_label: array with predicted values
            set_name: _description_
            file_path: path to save the file
        """

        with open(file_path, "w") as f:
            f.write(set_name)
            f.write("\n")
            f.write(classification_report(true_label, pred_label, zero_division=0))
            f.write("\n")

    def __divide_into_batches(
        self, dataset: Type[th.utils.data.Dataset], batch_size: int
    ) -> Type[dgl.dataloading.GraphDataLoader]:
        """
        Divide the dataset into batched graph data loader

        Args:
            dataset: Dataset to divide in DGLDataset format
            batch_size: number of batch

        Returns:

        """

        dataloader = GraphDataLoader(
            dataset, batch_size=batch_size, drop_last=False, shuffle=True
        )

        return dataloader

    def __build_gcn_model(self, type_gcn: str, n_features: int, hidden_nodes: int) -> Any:
        """
        Build the GCN model

        Args:
            type_gcn: type of model to build
            n_features: number of features of the first layer
            hidden_nodes: number of nodes in the hidden layers

        Returns:
            built model
        """

        models = {
            "gcn2": GCN2(n_features, hidden_nodes, 3),
            "gcn7": GCN7(n_features, hidden_nodes, 3),
            "gcn60": GCN60(n_features, hidden_nodes, 3),
            "gcn120": GCN120(n_features, hidden_nodes, 3),
            "gcn240": GCN240(n_features, hidden_nodes, 3),
        }

        return models.get(type_gcn)

    def training_and_testing(
        self,
        dataset_train: Type[dgl.data.DGLDataset],
        dataset_val: Type[dgl.data.DGLDataset],
        **kwargs
    ) -> None:
        """
        Trains and evaluates the model.

        Args:
            dataset_train: dataset of training data
            dataset_val: dataset of testing data
            kwargs: auxiliar arguments with keys "epochs", "nodes_hidden_layer", "n_features", "type_gcn", "path"
        """

        aux_var = {
            "pred_train_label": [],
            "true_train_label": [],
            "pred_val_label": [],
            "true_val_label": [],
            "acc_train": [],
            "loss_train": [],
            "num_correct_train": 0,
            "num_vals_train": 0,
            "num_correct_val": 0,
            "num_vals_val": 0,
            "tag": True
        }

        train_loader = self.__divide_into_batches(dataset=dataset_train, batch_size=64)
        val_loader = self.__divide_into_batches(dataset=dataset_val, batch_size=64)

        model = self.__build_gcn_model(
            type_gcn=kwargs.get("type_gcn"),
            n_features=kwargs.get("n_features"),
            hidden_nodes=kwargs.get("nodes_hidden_layer")
        )

        opt = th.optim.Adam(model.parameters(), lr=0.001)

        # training step
        for epoch in range(kwargs.get("epochs")):
            for batched_graph, labels in train_loader:
                pred = model(batched_graph, batched_graph.ndata["attr"].float())
                if aux_var.get("tag"):
                    aux_var.get("pred_train_label").extend(pred.argmax(1))
                    aux_var.get("true_train_label").extend(labels)

                aux_var["num_correct_train"] += (pred.argmax(1) == labels).sum().item()
                aux_var["num_vals_train"] += len(labels)
                loss = F.cross_entropy(pred, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
            aux_var["tag"] = False
            aux_var.get("loss_train").extend(loss)
            aux_var.get("acc_train").extend(aux_var.get("num_correct_train") / aux_var.get("num_vals_train"))
            print(f"epoch_{epoch}...")

        # testing step
        for batched_graph, labels in val_loader:
            pred = model(batched_graph, batched_graph.ndata["attr"].float())
            aux_var["num_correct_val"] += (pred.argmax(1) == labels).sum().item()
            aux_var["num_vals_val"] += len(labels)
            aux_var.get("pred_val_label").extend(pred.argmax(1))
            aux_var.get("true_val_label").extend(labels)

        aux_var["loss_train"] = [float(x.detach().numpy()) for x in aux_var.get("loss_train")]

        # plotting the metrics
        self.__plotting_acc_loss(
            acc_values=aux_var.get("acc_train"),
            loss_values=aux_var.get("loss_train"),
            epochs=kwargs.get("epochs"),
            path=f"{kwargs.get("path")}/acc_loss_{kwargs.get("type_gcn")}.png",
        )
        self.__plotting_confusion_matrix(
            true_label=aux_var.get("true_val_label"),
            pred_label=aux_var.get("pred_val_label"),
            path=f"{kwargs.get("path")}/confusion_matrix_{kwargs.get("type_gcn")}.png",
        )
        self.__getting_classification_report(
            true_label=aux_var.get("true_val_label"),
            pred_label=aux_var.get("pred_val_label"),
            set_name="VALIDATION",
            file_path=f"{kwargs.get("path")}/report_{kwargs.get("type_gcn")}.txt",
        )

    def training(
        self,
        dataset_train: Type[dgl.data.DGLDataset],
        model_name: str,
        **kwargs
    ) -> None:
        """
        Trains the model.

        Args:
            dataset_train: dataset of training data
            model_name: name of model to save it
            kwargs: auxiliar arguments with keys "epochs", "nodes_hidden_layer", "n_features", "type_gcn", "path"
        """

        aux_var = {
            "pred_train_label": [],
            "true_train_label": [],
            "acc_train": [],
            "loss_train": [],
            "num_correct_train": 0,
            "num_vals_train": 0,
            "tag": True
        }

        train_loader = self.__divide_into_batches(dataset=dataset_train, batch_size=64)

        model = self.__build_gcn_model(
            type_gcn=kwargs.get("type_gcn"),
            n_features=kwargs.get("n_features"),
            hidden_nodes=kwargs.get("nodes_hidden_layer")
        )

        opt = th.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(kwargs.get("epochs")):
            for batched_graph, labels in train_loader:
                pred = model(batched_graph, batched_graph.ndata["attr"].float())
                if aux_var.get("tag"):
                    aux_var.get("pred_train_label").extend(pred.argmax(1))
                    aux_var.get("true_train_label").extend(labels)

                aux_var["num_correct_train"] += (pred.argmax(1) == labels).sum().item()
                aux_var["num_vals_train"] += len(labels)
                loss = F.cross_entropy(pred, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
            aux_var["tag"] = False
            aux_var.get("loss_train").extend(loss)
            aux_var.get("acc_train").extend(aux_var.get("num_correct_train") / aux_var.get("num_vals_train"))
            print(f"epoch_{epoch}...")

        th.save(model.state_dict(), f"{kwargs.get("path")}/{model_name}.pth")

        aux_var["loss_train"] = [float(x.detach().numpy()) for x in aux_var.get("loss_train")]

        self.__plotting_acc_loss(
            acc_values=aux_var.get("acc_train"),
            loss_values=aux_var.get("loss_train"),
            epochs=kwargs.get("epochs"),
            path=f"{kwargs.get("path")}/acc_loss_{kwargs.get("type_gcn")}.png",
        )

    def testing(
        self,
        dataset_val: Type[dgl.data.DGLDataset],
        model_name: str,
        **kwargs
    ) -> None:
        """
        Trains and evaluates the model.

        Args:
            dataset_val: dataset of testing data
            model_name: name of model to save it
            kwargs: auxiliar arguments with keys "nodes_hidden_layer", "n_features", "type_gcn", "path"
        """

        aux_var = {
            "pred_val_label": [],
            "true_val_label": [],
            "num_correct_val": 0,
            "num_vals_val": 0
        }

        val_loader = self.__divide_into_batches(dataset=dataset_val, batch_size=64)

        model = self.__build_gcn_model(
            type_gcn=kwargs.get("type_gcn"),
            n_features=kwargs.get("n_features"),
            hidden_nodes=kwargs.get("nodes_hidden_layer")
        )
        model.load_state_dict(th.load(f"./Models/{model_name}.pth"))
        model.eval()

        for batched_graph, labels in val_loader:
            pred = model(batched_graph, batched_graph.ndata["attr"].float())
            aux_var["num_correct_val"] += (pred.argmax(1) == labels).sum().item()
            aux_var["num_vals_val"] += len(labels)
            aux_var.get("pred_val_label").extend(pred.argmax(1))
            aux_var.get("true_val_label").extend(labels)

        self.__plotting_confusion_matrix(
            true_label=aux_var.get("true_val_label"),
            pred_label=aux_var.get("pred_val_label"),
            path=f"{kwargs.get("path")}/confusion_matrix_{kwargs.get("type_gcn")}.png",
        )
        self.__getting_classification_report(
            true_label=aux_var.get("true_val_label"),
            pred_label=aux_var.get("pred_val_label"),
            set_name="VALIDATION",
            file_path=f"{kwargs.get("path")}/report_{kwargs.get("type_gcn")}.txt",
        )
