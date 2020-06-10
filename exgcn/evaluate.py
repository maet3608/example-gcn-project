"""
Evaluate network on test data
"""

import torch as to

from typing import List
from exgcn.constants import CFG, DEVICE
from exgcn.typedefs import Network, Samples, IVec
from exgcn.network import create_network, load_wgts
from exgcn.common import LoadGraph, MakeBatch, PredBatch, load_samples
from nutsflow import Cache, Unzip, Flatten, Collect


def evaluate(cfg: CFG, net: Network, testdata: Samples) -> (IVec, IVec, float):
    net.eval()
    with to.no_grad():
        tars, preds, probs = (testdata >> LoadGraph() >>
                              MakeBatch(cfg.batchsize) >>
                              PredBatch(net) >> Unzip())
        tars = tars >> Flatten() >> Collect()
        preds = preds >> Flatten() >> Collect()
        acc = 100.0 * [t == p for t, p in zip(tars, preds)].count(True) / len(
            tars)
    return tars, preds, acc


if __name__ == '__main__':
    print('creating network ...')
    net = create_network()
    print('loading weights ...')
    load_wgts(net)

    print('loading samples ...')
    testdata = load_samples()[1]

    print('evaluating on', DEVICE, '...')
    tars, preds, acc = evaluate(CFG, net, testdata)
    print('targets    ', tars)
    print('predictions', preds)
    print('test accuracy:', acc)
