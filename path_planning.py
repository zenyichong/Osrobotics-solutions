#!/usr/bin/env python
import numpy as np
import pylab as pl
import sys
from roadmap import PRM

sys.path.append('osr_examples/scripts/')
import environment_2d


NUM_NODES = 350    # 350
MAX_NEIGHBOURS = 8  # 10
RADIUS = 2     # 2


def main():
    pl.ion()
    np.random.seed(4)
    env = environment_2d.Environment(10, 6, 5)
    pl.clf()
    env.plot()
    q = env.random_query()
    if q is not None:
        x_start, y_start, x_goal, y_goal = q
        env.plot_query(x_start, y_start, x_goal, y_goal)

    graph = PRM(env, NUM_NODES, MAX_NEIGHBOURS, RADIUS)
    graph.random_sampling()
    graph.find_closest_neighbours()
    path = graph.solve_query()
    graph.plot_nodes()
    graph.plot_edges()
    graph.plot_path(path)
    # pl.savefig(f'random_seed_{i}.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
