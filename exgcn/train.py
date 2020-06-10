"""
Network training
"""
import time

import numpy as np
import torch as to

from exgcn.typedefs import Network, Samples
from exgcn.constants import CFG, DEVICE
from exgcn.common import (MakeBatch, TrainBatch, EvalBatch, load_samples,
                          LoadGraph)
from exgcn.network import create_network, save_wgts
from nutsflow import nut_function, Unzip, Mean, Shuffle, Cache, Take
from nutsml import PlotLines


def validate(cfg: CFG, net: Network, valdata: Samples,
             val_cache: Cache) -> float:
    net.eval()
    with to.no_grad():
        val_acc = (valdata >> LoadGraph() >> val_cache >>
                   MakeBatch(cfg.batchsize) >> EvalBatch(net) >> Mean())
    return val_acc


def train(cfg: CFG, net: Network, traindata: Samples, valdata: Samples):
    plotlines = PlotLines((0, 1, 2), layout=(3, 1), figsize=(8, 12),
                          titles=('loss', 'train-acc', 'val-acc'))
    train_cache = Cache(cfg.cachedir + '/train', cfg.cacheclear)
    val_cache = Cache(cfg.cachedir + '/val', cfg.cacheclear)
    optimizer = to.optim.Adam(net.parameters(), lr=cfg.lr)

    max_val = 0
    for epoch in range(cfg.n_epochs):
        start = time.time()
        net.train()
        losses, accs = (traindata >> LoadGraph() >> train_cache >>
                        Shuffle(100) >> MakeBatch(cfg.batchsize) >>
                        TrainBatch(net, optimizer) >> Unzip())
        loss, acc = np.mean(losses), np.mean(accs)

        if cfg.verbose:
            msg = "{:d}..{:d}  {:s} : {:.4f}  ({:.1f}% {:.1f}%)"
            elapsed = time.strftime("%M:%S", time.gmtime(time.time() - start))
            print(msg.format(epoch, cfg.n_epochs, elapsed, loss, acc, max_val))

        val_acc = validate(cfg, net, valdata, val_cache)
        if val_acc >= max_val:
            max_val = val_acc
            save_wgts(net)

        if cfg.verbose > 1:
            plotlines((loss, acc, max_val))


if __name__ == '__main__':
    print('creating network ...')
    net = create_network()

    print('loading samples ...')
    samplesets = load_samples()
    print("#samples", list(map(len, samplesets)))

    print('training on', DEVICE, '...')
    train(CFG, net, samplesets[0], samplesets[1])
