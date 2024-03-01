import os
import numpy as np
import pandas as pd
import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import igraph as ig

from ts2vg import NaturalVG
from dgl.data import DGLDataset
from dgl.nn import SAGEConv
from dgl.dataloading import GraphDataLoader
from collections import defaultdict
from scipy.io import loadmat
from sklearn.metrics import classification_report, confusion_matrix
from typing import Type, Tuple
from sklearn.preprocessing import minmax_scale


def segmentation_signals(path: str, list_ecgs: list, size_beat_before: int, size_beat_after: int, set_name: str
                         ) -> Tuple[defaultdict, defaultdict]:

    dict_signals_V1 = defaultdict(list)  # dict to load beats V1
    dict_signals_II = defaultdict(list)  # dict to load beats II

    for file in list_ecgs:

        struct = loadmat(os.path.join(path, set_name, file))  # loading the original file
        data = struct["individual"][0][0]  # loading info of the signal
        ecg_V1 = data["signal_r"][:, 1]  # reading lead V1
        ecg_II = data["signal_r"][:, 0]  # reading lead II

        if file == '114.mat':
            ecg_V1 = data['signal_r'][:, 0]
            ecg_II = data['signal_r'][:, 1]

        beat_peaks = data["anno_anns"]  # reading R-peak
        beat_types = data["anno_type"]  # reading type of beat

        for (peak, beat_type) in zip(beat_peaks, beat_types):

            beat_samples_V1 = []  # list to save samples of beat V1
            beat_samples_II = []  # list to save samples of beat II
            # half_beat = int(size_beat/2) #half of size beat

            # if the position is before the begining or
            # if the position is after the ending
            # do nothing

            if (peak - size_beat_before) < 0 or (peak + size_beat_after) > len(ecg_V1):
                continue

            # if type of beat is different than this list, do nothing
            if beat_type not in "NLRejAaJSVEFP/fUQ":
                continue

            # taking the samples of beat window
            beat_samples_V1 = ecg_V1[int(peak - size_beat_before): int(peak + size_beat_after)]
            beat_samples_II = ecg_II[int(peak - size_beat_before): int(peak + size_beat_after)]

            # taking the type of beat and saving in dict
            if beat_type in "NLRej":
                dict_signals_V1["N"].append(beat_samples_V1)
                dict_signals_II["N"].append(beat_samples_II)
            elif beat_type in "AaJS":
                dict_signals_V1["S"].append(beat_samples_V1)
                dict_signals_II["S"].append(beat_samples_II)
            elif beat_type in "VE":
                dict_signals_V1["V"].append(beat_samples_V1)
                dict_signals_II["V"].append(beat_samples_II)

    return dict_signals_V1, dict_signals_II


def sampling_windows_10_beats(signals: defaultdict) -> defaultdict:

    select_N_beats = []

    for index, beat in enumerate(signals['N'], 1):
        if (index % 10) == 0:
            select_N_beats.append(beat)

    signals["N"] = select_N_beats

    return signals


def get_beats_features(signals_V1: dict, signals_II: dict) -> dict:

    features = {}
    nodes_feat = []
    graph_it = 0

    # iterate in dict of beats, rr-interval
    for (_, beats_V1), (_, beats_II) in zip(signals_V1.items(), signals_II.items()):
        for beat_V1, beat_II in zip(beats_V1, beats_II):
            nodes_feat = []
            [nodes_feat.append(minmax_scale([i, j, k])) for i, (j, k) in enumerate(zip(beat_II, beat_V1))]
            features.update({graph_it: nodes_feat})
            graph_it += 1

    return features


def convert_beats_in_graphs(signals_II: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:

    graph_id = []
    graph_src = []
    graph_dst = []
    graph_nodes = []
    graph_label = []
    classes = ["N", "S", "V"]
    graph_it = 0

    for class_, beats in signals_II.items():
        for beat in beats:
            label = classes.index(class_)
            g = NaturalVG(directed=None).build(beat)
            G = g.as_igraph()
            source_nodes_ids = [i[0] for i in ig.Graph.get_edgelist(G)]
            destination_nodes_ids = [i[1] for i in ig.Graph.get_edgelist(G)]
            num_nodes = G.vcount()
            graph_id.extend([graph_it] * len(ig.Graph.get_edgelist(G)))
            graph_src.extend(source_nodes_ids)
            graph_dst.extend(destination_nodes_ids)
            graph_nodes.extend([num_nodes] * len(ig.Graph.get_edgelist(G)))
            graph_label.extend([label] * len(ig.Graph.get_edgelist(G)))
            graph_it += 1
            del G

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
    plt.savefig(f"./Images/acc_loss_gcn60.png", dpi=600)

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

    ax.set_title(f"Matriz de Confusão", fontsize=18)
    ax.set_xlabel("Predição", fontsize=16)
    ax.set_ylabel("Verdadeiro", fontsize=16)
    plt.savefig(f"./Images/confusion_matrix_gcn60.png", dpi=600)

    return None


def getting_classification_report(true_label: np.array, pred_label: np.array, set_name: str):

    with open(f"./Images/report_gcn60.txt", "w") as f:
        f.write(set_name)
        f.write("\n")
        f.write(classification_report(true_label, pred_label, zero_division=0))
        f.write("\n")

    return None


def divide_into_batches(dataset: Type[th.utils.data.Dataset]) -> Type[dgl.dataloading.GraphDataLoader]:

    dataloader = GraphDataLoader(dataset, batch_size=64, drop_last=False, shuffle=True)

    return dataloader


def training(dataset_train: Type[dgl.data.DGLDataset], dataset_val: Type[dgl.data.DGLDataset]) -> None:

    epochs = 150
    pred_train_label = []
    true_train_label = []
    pred_val_label = []
    true_val_label = []
    acc_train = []
    loss_train = []
    num_correct_train = 0
    num_vals_train = 0
    num_correct_val = 0
    num_vals_val = 0
    tag = True

    train_loader = divide_into_batches(dataset_train)
    val_loader = divide_into_batches(dataset_val)

    model = GCN(3, 60, 3)  # (n_nodes_features, n_nodes_hidden_layer, n_classes)

    # creating the optimizer
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

    for batched_graph, labels in val_loader:
        pred = model(batched_graph, batched_graph.ndata["attr"].float())
        num_correct_val += (pred.argmax(1) == labels).sum().item()
        num_vals_val += len(labels)
        pred_val_label.extend(pred.argmax(1))
        true_val_label.extend(labels)

    loss_train = [float(x.detach().numpy()) for x in loss_train]

    # plotting the metrics
    plotting_acc_loss(acc_train, loss_train, epochs)
    plotting_confusion_matrix(true_val_label, pred_val_label)
    getting_classification_report(true_val_label, pred_val_label, "VALIDATION")

    return None


class SyntheticDataset(DGLDataset):
    def __init__(self, attr_edges, attr_properties, attr_features):
        self.attr_edges = attr_edges
        self.attr_properties = attr_properties
        self.attr_features = attr_features
        self.graphs = []
        self.labels = []

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
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats - 10, "mean")
        self.conv3 = SAGEConv(h_feats - 10, h_feats - 25, "mean")
        self.conv4 = SAGEConv(h_feats - 25, num_classes, "mean")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        h = F.relu(h)
        h = self.conv4(g, h)
        g.ndata["h"] = h
        return dgl.mean_nodes(g, "h")


if __name__ == '__main__':

    PATH = "../../../Data"
    FILES = os.listdir(os.path.join(PATH, 'Train'))
    FILES_VAL = ['109.mat', '114.mat', '207.mat', '223.mat']
    FILES_TRAIN = list(set(FILES)-set(FILES_VAL))

    print("segmentating...")
    train_signals_V1, train_signals_II = segmentation_signals(PATH, FILES_TRAIN, 100, 180, 'Train')
    val_signals_V1, val_signals_II = segmentation_signals(PATH, FILES_VAL, 100, 180, 'Train')

    print("sampling...")
    train_signals_V1 = sampling_windows_10_beats(train_signals_V1)
    train_signals_II = sampling_windows_10_beats(train_signals_II)
    val_signals_V1 = sampling_windows_10_beats(val_signals_V1)
    val_signals_II = sampling_windows_10_beats(val_signals_II)

    print("extracting attributes...")
    train_features = get_beats_features(train_signals_V1, train_signals_II)
    val_features = get_beats_features(val_signals_V1, val_signals_II)

    print("converting beats into graphs...")
    train_edges, train_properties = convert_beats_in_graphs(train_signals_II)
    val_edges, val_properties = convert_beats_in_graphs(val_signals_II)

    print("creating dataset...")
    dataset_train = SyntheticDataset(train_edges, train_properties, train_features)
    dataset_val = SyntheticDataset(val_edges, val_properties, val_features)

    print("training...")
    training(dataset_train, dataset_val)
