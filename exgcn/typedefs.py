"""
Type definitions
"""

import torch.nn as nn
from dgl import DGLGraph
from typing import Iterable, Tuple, List, Generator

CCoords = Iterable[Tuple[float, float]]
PCoords = Iterable[Tuple[float, float]]
Edges = Iterable[Tuple[int, int]]
Network = nn.Module
IVec = List[int]
Sample = Tuple[str, int]
Samples = Iterable[Sample]
GraphSample = Tuple[DGLGraph, int]
GraphSamples = Iterable[GraphSample]
BatchGen = Generator
Batch = Tuple
