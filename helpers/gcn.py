from dgl.nn import GraphConv, SAGEConv
import torch.nn.functional as F
import torch.nn as nn
import dgl


class GCN2(nn.Module):
    """
    Create a graph convolutional neural network with two layers
    """

    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN2, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata["h"] = h

        return dgl.mean_nodes(g, "h")


class GCN7(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN7, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = GraphConv(h_feats, h_feats - 10)
        self.conv3 = SAGEConv(h_feats - 10, h_feats - 20, "mean")
        self.conv4 = SAGEConv(h_feats - 20, h_feats - 30, "mean")
        self.conv5 = SAGEConv(h_feats - 30, h_feats - 40, "mean")
        self.conv6 = GraphConv(h_feats - 40, h_feats - 45)
        self.conv7 = GraphConv(h_feats - 45, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        h = F.relu(h)
        h = self.conv4(g, h)
        h = F.relu(h)
        h = self.conv5(g, h)
        h = F.relu(h)
        h = self.conv6(g, h)
        h = F.relu(h)
        h = self.conv7(g, h)
        g.ndata["h"] = h
        return dgl.mean_nodes(g, "h")


class GCN60(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN60, self).__init__()
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


class GCN120(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN120, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats - 80, "mean")
        self.conv3 = SAGEConv(h_feats - 80, h_feats - 100, "mean")
        self.conv4 = SAGEConv(h_feats - 100, num_classes, "mean")

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


class GCN240(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN240, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats - 100, "mean")
        self.conv3 = SAGEConv(h_feats - 100, h_feats - 200, "mean")
        self.conv4 = SAGEConv(h_feats - 200, num_classes, "mean")

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
