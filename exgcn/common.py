"""
Common functions
"""
import dgl

import numpy as np
import torch as to
import torch.nn.functional as tf

from dgl import DGLGraph
from exgcn.typedefs import (CCoords, PCoords, Edges, GraphSample, GraphSamples,
                            Sample, Samples, BatchGen, Network, Batch)
from exgcn.constants import CFG, DEVICE
from nutsflow import (Chunk, Unzip, MapCol, nut_function, nut_processor,
                      Head, Print)
from nutsml import ReadLabelDirs, SplitRandom


def to_numpy(x: to.Tensor) -> np.array:
    return x.detach().cpu().numpy()


def probabilities(logits) -> to.Tensor:
    return to.softmax(logits, 1)


def accuracy(pred_logits: to.Tensor, target: to.Tensor) -> float:
    probs = probabilities(pred_logits)
    preds = to.max(probs, 1)[1].view_as(target)
    acc = (target == preds).sum().item() / len(target) * 100.0
    return acc


def view_graph(graph: DGLGraph):
    import matplotlib.pyplot as plt
    import networkx as nx
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    gx = graph.to_networkx(node_attrs=['xy'], edge_attrs=['polar'])
    pos = [(y, -x) for n, (x, y) in gx.nodes.data('xy')]
    nx.draw(gx, pos=pos,
            with_labels=True, font_size=9,
            node_size=300,
            node_color=[[0.7, 0.8, 0.9]],
            edge_color=[0, 0, 0, 0.4],
            arrows=True,
            ax=ax)
    plt.show()


def to_dgl_graph(xy: CCoords, polar: PCoords, edges: Edges) -> DGLGraph:
    g = DGLGraph()
    g.add_nodes(len(xy))
    g.add_edges(*zip(*edges))
    g.ndata['xy'] = np.array(xy, dtype=np.float32)
    g.edata['polar'] = np.array(polar, dtype=np.float32)
    return g


@nut_function
def LoadGraph(sample: Sample) -> GraphSample:
    import json
    filepath, label = sample
    with open(filepath) as f:
        g = json.load(f)
        return to_dgl_graph(g['xys'], g['polars'], g['edges']), label


@nut_processor
def MakeBatch(samples: GraphSamples, batchsize: int) -> BatchGen:
    for batch in samples >> Chunk(batchsize):
        graphs, targets = batch >> Unzip()
        graph_batch = dgl.batch(graphs).to(DEVICE)
        tar_batch = to.tensor(targets).to(DEVICE)
        yield graph_batch, tar_batch


@nut_function
def EvalBatch(batch: Batch, net: Network) -> float:
    graphs, targets = batch
    pred_logits = net(graphs)
    return accuracy(pred_logits, targets)


@nut_function
def PredBatch(batch: Batch, net: Network) -> (np.array, np.array, np.array):
    graphs, targets = batch
    pred_logits = net(graphs)
    probs = probabilities(pred_logits)
    preds = to.max(probs, 1)[1].view_as(targets)
    return to_numpy(targets), to_numpy(preds), to_numpy(probs)


@nut_function
def TrainBatch(batch: Batch, net: Network, optimizer) -> (float, float):
    graphs, labels = batch
    pred_logits = net(graphs)
    loss = tf.cross_entropy(pred_logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), accuracy(pred_logits, labels)


def load_samples() -> (Samples, Samples, Samples):
    read = ReadLabelDirs(CFG.datadir, '*.json')
    label2int = MapCol(1, int)
    return read >> label2int >> SplitRandom(ratio=CFG.ratios)


if __name__ == '__main__':
    train_samples, val_samples, test_samples = load_samples()
    train_samples >> LoadGraph() >> MakeBatch(2) >> Print() >> Head(3)
    # for graph, label in train_samples >> LoadGraph():
    #    view_graph(graph)
