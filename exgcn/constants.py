"""
Constants
"""

import torch as to
from nutsml.config import Config

DEVICE = to.device('cuda' if to.cuda.is_available() else 'cpu')

CFG = Config(
    n_epochs=50,
    batchsize=4,
    lr=0.1,
    n_classes=2,
    ratios=(0.1, 0.4, 0.5),
    verbose=2,
    datadir='../data',
    cachedir='../cache',
    cacheclear=True,
)
