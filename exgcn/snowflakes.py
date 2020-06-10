"""
Create fake data; in this case graph representations of snowflakes.
"""

import json

from math import cos, sin, radians
from exgcn.common import view_graph, to_dgl_graph


class Turtle(object):
    def __init__(self, alpha: float = 0, x: float = 0, y: float = 0):
        self.phi = radians(alpha)
        self.x = x
        self.y = y
        self.xys = [(x, y)]
        self.polars = []
        self.edges = []

    def n_nodes(self):
        return len(self.xys) - 1

    def move(self, length):
        self.x += round(length * cos(self.phi), 5)
        self.y += round(length * sin(self.phi), 5)

    def turn(self, alpha):
        self.phi += radians(alpha)

    def save(self, filepath: str):
        data = {'xys': self.xys, 'polars': self.polars, 'edges': self.edges}
        with open(filepath, 'w') as f:
            json.dump(data, f)


def mk_snowflake(turtle: Turtle, length: float, branches: int, depth: int):
    angle = 360.0 / branches
    if depth > 0:
        u = turtle.n_nodes()
        for i in range(branches):
            turtle.move(length)
            turtle.xys.append((turtle.x, turtle.y))
            turtle.edges.append((turtle.n_nodes(), u))
            turtle.polars.append((angle, length))
            mk_snowflake(turtle, length / 3, branches, depth - 1)
            turtle.move(-length)
            turtle.turn(angle)


def gen_snowflakes(n: int, branches: int):
    from random import randint
    for n in range(n):
        angle = randint(0, 360 // branches)
        length = randint(10, 100)
        depth = randint(3, 5)
        turtle = Turtle(angle)
        mk_snowflake(turtle, length, branches, depth)
        yield turtle


def write_dataset(n: int):
    from exgcn.constants import CFG
    from os.path import join
    from os import mkdir
    for label, branches in {'0': 5, '1': 6}.items():
        mkdir(join(CFG.datadir, label))
        for i, flake in enumerate(gen_snowflakes(n, branches)):
            filepath = join(CFG.datadir, label, 'flake_%03d.json' % i)
            print('saving:', filepath)
            flake.save(filepath)


def show_example():
    turtle = Turtle(0)
    mk_snowflake(turtle, 10, 5, 2)
    graph = to_dgl_graph(turtle.xys, turtle.polars, turtle.edges)
    view_graph(graph)


if __name__ == '__main__':
    #show_example()
    write_dataset(30)
