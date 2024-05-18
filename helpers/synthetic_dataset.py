import torch as th
import dgl
import pandas as pd
import numpy as np
from dgl.data import DGLDataset


class SyntheticDataset(DGLDataset):
    """
    Create a graph dataset for graph classification.
    The graph dataset should inherit the dgl.data.DGLDataset class
    """

    def __init__(
        self,
        attr_edges: pd.DataFrame,
        attr_properties: pd.DataFrame,
        attr_features: dict,
    ):
        self.attr_edges = attr_edges
        self.attr_properties = attr_properties
        self.attr_features = attr_features
        self.graphs = []
        self.labels = []

        super().__init__(name="synthetic")

    def process(self):
        """
        load and process raw data from disk
        """

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
