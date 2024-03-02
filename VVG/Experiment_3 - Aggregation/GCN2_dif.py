from typing import Type, Tuple
import os
import numpy as np
import pandas as pd
import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from dgl.data import DGLDataset
from dgl.nn import GraphConv
from dgl.dataloading import GraphDataLoader
from collections import defaultdict
from scipy.io import loadmat
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import minmax_scale


def segmentation_signals(path: str, list_ecgs: list, size_beat_before: int, size_beat_after: int, set_name: str
                         ) -> Tuple[defaultdict, defaultdict, defaultdict, defaultdict]:

    dict_signals_V1 = defaultdict(list)  # dict to load beats
    dict_signals_II = defaultdict(list)  # dict to load beats
    rr_interval_pos_signals = defaultdict(list)  # dict to load rr interval
    rr_interval_pre_signals = defaultdict(list)  # dict to load rr interval
    rr_interval_pos = 0
    rr_interval_pre = 0

    for file in list_ecgs:

        struct = loadmat(os.path.join(path, set_name, file))  # loading the original file
        data = struct["individual"][0][0]  # loading info of the signal

        beat_peaks = data["anno_anns"]  # reading R-peak
        beat_types = data["anno_type"]  # reading type of beat

        ecg_V1 = data["signal_r"][:, 1]  # reading lead V1
        ecg_II = data["signal_r"][:, 0]  # reading lead II

        if file == '114.mat':
            ecg_V1 = data['signal_r'][:, 0]
            ecg_II = data['signal_r'][:, 1]

        ecg_II = normalize(ecg_II)
        ecg_V1 = normalize(ecg_V1)

        for it, (peak, beat_type) in enumerate(zip(beat_peaks, beat_types)):

            beat_samples_V1 = []  # list to save samples of beat V1
            beat_samples_II = []  # list to save samples of beat II
            # half_beat = int(size_beat/2) #half of size beat

            # if the position is before the begining or
            # if the position is after the ending
            # do nothing
            if (peak - size_beat_before) < 0 or (peak + size_beat_after) > len(ecg_II):
                continue

            # if type of beat is different than this list, do nothing
            if beat_type not in "NLRejAaJSVEFP/fUQ":
                continue

            # taking the samples of beat window
            beat_samples_II = ecg_II[int(peak - size_beat_before): int(peak + size_beat_after)]
            beat_samples_V1 = ecg_V1[int(peak - size_beat_before): int(peak + size_beat_after)]

            # calculate the rr_interval using
            if it-1 > 0:
                rr_interval_pre = (peak - beat_peaks[it-1])[0]
            if it+1 < len(beat_peaks):
                rr_interval_pos = (beat_peaks[it+1] - peak)[0]

            # taking the type of beat and saving in dict
            if beat_type in "NLRej":
                dict_signals_V1["N"].append(beat_samples_V1)
                dict_signals_II["N"].append(beat_samples_II)
                rr_interval_pos_signals["N"].append(rr_interval_pos)
                rr_interval_pre_signals["N"].append(rr_interval_pre)
            elif beat_type in "AaJS":
                dict_signals_V1["S"].append(beat_samples_V1)
                dict_signals_II["S"].append(beat_samples_II)
                rr_interval_pos_signals["S"].append(rr_interval_pos)
                rr_interval_pre_signals["S"].append(rr_interval_pre)
            elif beat_type in "VE":
                dict_signals_V1["V"].append(beat_samples_V1)
                dict_signals_II["V"].append(beat_samples_II)
                rr_interval_pos_signals["V"].append(rr_interval_pos)
                rr_interval_pre_signals["V"].append(rr_interval_pre)

    return dict_signals_V1, dict_signals_II, rr_interval_pos_signals, rr_interval_pre_signals


def normalize(data: np.ndarray) -> np.ndarray:
    data = np.nan_to_num(data)  # removing NaNs and Infs
    data = data - np.mean(data)
    data = data / np.std(data)
    return data


def sampling_windows_10_beats(signals_V1: defaultdict,
                              signals_II: defaultdict,
                              rr_interval_pos_signals: defaultdict,
                              rr_interval_pre_signals: defaultdict) -> (
                                  Tuple[defaultdict, defaultdict, defaultdict, defaultdict]):

    select_N_beats_V1 = []
    select_N_beats_II = []
    select_rr_interval_pos_signals = []
    select_rr_interval_pre_signals = []

    for index, (beat_V1, beat_II, rr_interval_pos, rr_interval_pre) in enumerate(zip(signals_V1['N'], signals_II['N'],
                                                                                     rr_interval_pos_signals['N'], rr_interval_pre_signals['N']), 1):
        if (index % 10) == 0:
            select_N_beats_V1.append(beat_V1)
            select_N_beats_II.append(beat_II)
            select_rr_interval_pos_signals.append(rr_interval_pos)
            select_rr_interval_pre_signals.append(rr_interval_pre)

    signals_V1["N"] = select_N_beats_V1
    signals_II["N"] = select_N_beats_II
    rr_interval_pos_signals["N"] = select_rr_interval_pos_signals
    rr_interval_pre_signals["N"] = select_rr_interval_pre_signals

    return signals_V1, signals_II, rr_interval_pos_signals, rr_interval_pre_signals


def get_beats_features(signals_V1: defaultdict, signals_II: defaultdict, rr_interval_pos_signals: defaultdict,
                       rr_interval_pre_signals: defaultdict) -> dict:

    features = {}
    nodes_feat = []
    graph_it = 0

    # iterate in dict of beats
    for (class_, beats_V1), (_, beats_II) in zip(signals_V1.items(), signals_II.items()):
        for it, (beat_V1, beat_II) in enumerate(zip(beats_V1, beats_II)):
            [nodes_feat.append(minmax_scale(
                [i, j, k, rr_interval_pos_signals[class_][it], rr_interval_pre_signals[class_][it], round(j-k, 3)]))
                for i, (j, k) in enumerate(zip(beat_II, beat_V1))]
            features.update({graph_it: nodes_feat})
            graph_it += 1
            nodes_feat = []

    return features


def projection_vectors_VVG(series_a: np.ndarray, series_b: np.ndarray, norm_a: float) -> float:
    # calculate the projection from series_a to series_b
    return np.dot(series_a, series_b) / norm_a


def criteria_VVG(series_a: np.ndarray, series_b: np.ndarray, series_c: np.ndarray, time_a: int, time_b: int, time_c: int,
                 norm_a: float) -> bool:
    # calculate the visibility graph of the three series
    # series_a, series_b, series_c are the three series
    # time_a, time_b, time_c are the time of the three series
    # returns True if the visibility graph of the three series is possible, False otherwise
    proj_aa = norm_a  # norm of vector_a
    proj_ab = projection_vectors_VVG(series_a, series_b, norm_a)
    proj_ac = projection_vectors_VVG(series_a, series_c, norm_a)
    time_frac = (time_b - time_c) / (time_b - time_a)
    vg = proj_ab + (proj_aa - proj_ab) * time_frac
    return proj_ac < vg


def vector_visibility_graph_VVG(series_a: np.ndarray, series_b: np.ndarray) -> dict:
    # calculate the visibility graph of the two series
    # adjacency_list is the adjacency list of the visibility graph
    # series_a, series_b are the two series
    # returns the adjacency list of the visibility graph
    norm_a = float(np.linalg.norm(series_a))
    adjacency_list = defaultdict(list)
    all_samples = np.column_stack((series_a, series_b))

    for i, sample_i in enumerate(all_samples):
        for s, sample_s in enumerate(all_samples[i + 1:], start=i + 1):
            if s == i + 1:
                adjacency_list[i].append(s)
                adjacency_list[s].append(i)
            else:
                for t, sample_t in enumerate(all_samples[i + 1:s], start=i + 1):
                    if criteria_VVG(sample_i, sample_s, sample_t, i, s, t, norm_a):
                        adjacency_list[i].append(s)
                        adjacency_list[s].append(i)
                        break
    return adjacency_list


def convert_beats_in_graphs(signals_V1: defaultdict, signals_II: defaultdict) -> Tuple[pd.DataFrame, pd.DataFrame]:

    graph_id = []
    graph_src = []
    graph_dst = []
    graph_nodes = []
    graph_label = []
    classes = ["N", "S", "V"]
    graph_it = 0

    for (class_, beats_V1), (_, beats_II) in zip(signals_V1.items(), signals_II.items()):
        for beat_V1, beat_II in zip(beats_V1, beats_II):
            label = classes.index(class_)
            adjlist_ = vector_visibility_graph_VVG(beat_V1, beat_II)
            G = nx.DiGraph(adjlist_)
            source_nodes_ids = [i[0] for i in nx.to_edgelist(G)]
            destination_nodes_ids = [i[1] for i in nx.to_edgelist(G)]
            num_nodes = nx.number_of_nodes(G)

            graph_id.extend([graph_it] * len(nx.to_edgelist(G)))
            graph_src.extend(source_nodes_ids)
            graph_dst.extend(destination_nodes_ids)
            graph_nodes.extend([num_nodes] * len(nx.to_edgelist(G)))
            graph_label.extend([label] * len(nx.to_edgelist(G)))
            del G
            graph_it += 1

    edges = pd.DataFrame({"graph_id": graph_id, "src": graph_src, "dst": graph_dst})
    properties = pd.DataFrame({"graph_id": graph_id, "label": graph_label, "num_nodes": graph_nodes})
    properties.drop_duplicates(inplace=True)

    del graph_id, graph_src, graph_dst, graph_nodes, graph_label

    return edges, properties


def plotting_acc_loss(acc_values: list, loss_values: list, epochs: int) -> None:

    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    plt.suptitle("Treinamento", fontsize=20)
    ax[0].plot(range(1, epochs+1), acc_values, "r")
    ax[1].plot(range(1, epochs+1), loss_values, "b")
    ax[0].set_title("Acurácia")
    ax[1].set_title("Perda")
    ax[0].set_xlabel("Épocas")
    ax[1].set_xlabel("Épocas")
    ax[0].set_ylabel("Acurácia")
    ax[1].set_ylabel("Perda")
    plt.savefig(f"./Images2/acc_loss_gcn2_dif.png", dpi=600)

    return None


def plotting_confusion_matrix(true_label: np.array, pred_label: np.array) -> None:

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
    plt.savefig("./Images2/confusion_matrix_gcn2_dif.png", dpi=600)

    return None


def getting_classification_report(true_label: np.array, pred_label: np.array, set_name: str):

    m = confusion_matrix(true_label, pred_label)
    n = round(((m[1][0]+m[2][0])/((m[1][0]+m[2][0]) + (m[1][1]+m[1][2]+m[2][1]+m[2][2])))*100, 2)
    s = round(((m[0][1]+m[2][1])/((m[0][1]+m[2][1]) + (m[0][0]+m[0][2]+m[2][0]+m[2][2])))*100, 2)
    v = round(((m[0][2]+m[1][2])/((m[0][2]+m[1][2]) + (m[0][0]+m[0][1]+m[1][0]+m[1][1])))*100, 2)
    pond = round(((4423/9480)*n) + ((1837/9480)*s) + ((3220/9480)*v), 2)

    with open("./Images2/report_gcn2_dif.txt", "w") as f:
        f.write(set_name)
        f.write("\n")
        f.write(classification_report(true_label, pred_label, zero_division=0))
        f.write("\n")
        f.write(f'N: {n}\n')
        f.write(f'S: {s}\n')
        f.write(f'V: {v}\n')
        f.write(f'pond: {pond}')

    return None


def divide_into_batches(dataset: Type[th.utils.data.Dataset]) -> Type[dgl.dataloading.GraphDataLoader]:

    dataloader = GraphDataLoader(dataset, batch_size=64, drop_last=False, shuffle=True)

    return dataloader


def training(dataset_train: Type[dgl.data.DGLDataset], n_features: int, model_name: str) -> None:

    epochs = 150
    pred_train_label = []
    true_train_label = []
    acc_train = []
    loss_train = []
    num_correct_train = 0
    num_vals_train = 0
    tag = True

    train_loader = divide_into_batches(dataset_train)
    model = GCN(n_features, 50, 3)  # (n_nodes_features, n_nodes_hidden_layer, n_classes)
    opt = th.optim.Adam(model.parameters(), lr=0.001)

    # executing the training step

    for epoch in range(epochs):
        for batched_graph, labels in train_loader:
            pred = model(batched_graph, batched_graph.ndata["attr"].float())
            if tag:
                pred_train_label.extend(pred.argmax(1))
                true_train_label.extend(labels)
            num_correct_train += (pred.argmax(1) == labels).sum().item()
            num_vals_train += len(labels)
            loss = F.cross_entropy(pred, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
        tag = False
        loss_train.append(loss)
        acc_train.append(num_correct_train / num_vals_train)
        print(f"epoch_{epoch}...")

    th.save(model.state_dict(), f"./Models/{model_name}.pth")

    loss_train = [float(x.detach().numpy()) for x in loss_train]

    plotting_acc_loss(acc_train, loss_train, epochs)


def testing(dataset_test: Type[dgl.data.DGLDataset], n_features: int, model_name: str) -> None:

    pred_val_label = []
    true_val_label = []
    num_correct_val = 0
    num_vals_val = 0

    test_loader = divide_into_batches(dataset_test)

    model = GCN(n_features, 50, 3)  # (n_nodes_features, n_nodes_hidden_layer, n_classes)
    model.load_state_dict(th.load(f"./Models/{model_name}.pth"))
    model.eval()

    for batched_graph, labels in test_loader:
        pred = model(batched_graph, batched_graph.ndata["attr"].float())
        num_correct_val += (pred.argmax(1) == labels).sum().item()
        num_vals_val += len(labels)
        pred_val_label.extend(pred.argmax(1))
        true_val_label.extend(labels)

    plotting_confusion_matrix(true_val_label, pred_val_label)
    getting_classification_report(true_val_label, pred_val_label, "VALIDATION")


class SyntheticDataset(DGLDataset):
    def __init__(self, attr_edges, attr_properties, attr_features):
        self.attr_edges = attr_edges
        self.attr_properties = attr_properties
        self.attr_features = attr_features
        self.graphs: list = []
        self.labels: list = []

        super().__init__(name="synthetic")

    def process(self):

        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in self.attr_properties.iterrows():
            label_dict[row["graph_id"]] = row["label"]
            num_nodes_dict[row["graph_id"]] = row["num_nodes"]

        # For the edges, first group the table by graph IDs.
        edges_group = self.attr_edges.groupby("graph_id")

        # For each graph ID...
        for graph_id in edges_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id["src"].to_numpy()
            dst = edges_of_id["dst"].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]

            # Create a graph and add it to the list of graphs and labels.
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            g.ndata["attr"] = th.tensor(np.array(self.attr_features[graph_id]))
            g = dgl.add_self_loop(g)
            self.graphs.append(g)
            self.labels.append(label)
            del g

        # Convert the label list to tensor for saving.
        self.labels = th.LongTensor(self.labels)

    # get a graph and its label by index
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    # number of graphs in the dataset
    def __len__(self):
        return len(self.graphs)


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')


if __name__ == "__main__":

    PATH = "../../../Data"
    FILES_TRAIN = os.listdir(os.path.join(PATH, "Train"))
    FILES_TEST = os.listdir(os.path.join(PATH, "Test"))
    MODE = "Test"
    N_FEATURES = 6
    MODEL_NAME = "model2_dif"

    print("segmentating...")
    train_signals_V1, train_signals_II, train_rr_interval_pos_signals, train_rr_interval_pre_signals = (
        segmentation_signals(PATH, FILES_TRAIN, 100, 180, 'Train'))
    test_signals_V1, test_signals_II, test_rr_interval_pos_signals, test_rr_interval_pre_signals = (
        segmentation_signals(PATH, FILES_TEST, 100, 180, 'Test'))

    if MODE == 'Train':
        print("sampling...")
        train_signals_V1, train_signals_II, train_rr_interval_pos_signals, train_rr_interval_pre_signals = (
            sampling_windows_10_beats(train_signals_V1, train_signals_II, train_rr_interval_pos_signals,
                                      train_rr_interval_pre_signals))

        print("extracting attributes...")
        train_features = get_beats_features(train_signals_V1, train_signals_II, train_rr_interval_pos_signals,
                                            train_rr_interval_pre_signals)

        print("converting beats into graphs...")
        train_edges, train_properties = convert_beats_in_graphs(train_signals_V1, train_signals_II)

        print("creating dataset...")
        dataset_train = SyntheticDataset(train_edges, train_properties, train_features)

        print("training")
        training(dataset_train, N_FEATURES, MODEL_NAME)
    else:
        print("sampling...")
        test_signals_V1, test_signals_II, test_rr_interval_pos_signals, test_rr_interval_pre_signals = (
            sampling_windows_10_beats(test_signals_V1, test_signals_II, test_rr_interval_pos_signals,
                                      test_rr_interval_pre_signals))

        print("extracting attributes...")
        test_features = get_beats_features(test_signals_V1, test_signals_II, test_rr_interval_pos_signals,
                                           test_rr_interval_pre_signals)

        print("converting beats into graphs...")
        test_edges, test_properties = convert_beats_in_graphs(test_signals_V1, test_signals_II)

        print("creating dataset...")
        dataset_test = SyntheticDataset(test_edges, test_properties, test_features)

        print("testing")
        testing(dataset_test, N_FEATURES, MODEL_NAME)
