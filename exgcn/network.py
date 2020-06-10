"""
Definition of network architecture
"""
import dgl
import torch as to
import torch.nn as nn
from exgcn.constants import DEVICE, CFG


def aggregate(edges):
    return {'m': edges.data['polar']}


def combine(nodes):  # = dgl.function.sum('m', 'h')
    return {'h': to.mean(nodes.mailbox['m'], dim=1)}


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, graph, ndata):
        with graph.local_scope():
            graph.ndata['h'] = ndata
            graph.update_all(aggregate, combine)
            return graph.ndata['h']


# This is a stupid architecture and just an example!
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer()
        self.layer2 = GCNLayer()
        self.fc = nn.Linear(2, CFG.n_classes)

    def forward(self, g):
        g.update_all(aggregate, combine)
        h = g.ndata['h']

        h = self.layer1(g, h)
        h = self.layer2(g, h)
        h = to.relu(h)

        # readout
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.fc(hg)


def save_wgts(gcn: nn.Module, filepath: str = 'weights.pt'):
    to.save(gcn.state_dict(), filepath)


def load_wgts(gcn: nn.Module, filepath: str = 'weights.pt'):
    gcn.load_state_dict(to.load(filepath))


def create_network():
    gcn = GCN()
    gcn.to(DEVICE)
    return gcn
