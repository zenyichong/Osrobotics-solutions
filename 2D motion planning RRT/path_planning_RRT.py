#!/usr/bin/env python
import numpy as np
import pylab as pl
import sys
from random_tree import Treenode, RRT

sys.path.append('osr_examples/scripts/')
import environment_2d


MAX_RADIUS = 0.9
MAXREP = 800


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

    start = Treenode(env.start[0], env.start[1])
    goal = Treenode(env.goal[0], env.goal[1])
    start_tree = RRT(env, start, MAX_RADIUS)
    goal_tree = RRT(env, goal, MAX_RADIUS)
    midpoint = start_tree.bidirectional_RRT(goal_tree, MAXREP)
    start_tree.plot_tree()
    goal_tree.plot_tree()
    if midpoint is not None:
        goal_tree.plot_path(midpoint)
        start_tree.plot_path(midpoint)


if __name__ == '__main__':
    main()
