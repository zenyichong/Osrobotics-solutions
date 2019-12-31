#!/usr/bin/env python
import numpy as np
import pylab as pl


class Treenode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.children = []

    def get_euclidean_distance(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class RRT:
    def __init__(self, env, start, max_radius):
        self.env = env
        self.width = env.size_x
        self.height = env.size_y
        self.treenodes = set([start])
        self.treenode_parents = {}
        self.radius = max_radius

    def bidirectional_RRT(self, other, maxrep):
        """
        Simultaneously builds two trees, one rooted at the start, the other
        rooted at the goal. Returns the node where both trees meet if a path
        is found within maxrep iterations, otherwise None
        """
        for _ in range(maxrep):
            q_rand = self.random_config()
            self.extend(q_rand)
            other.extend(q_rand)
            if q_rand in self.treenodes and q_rand in other.treenodes:
                return q_rand
        return None

    def random_config(self):
        """ Returns a random configuration within Cfree """
        while True:
            x = np.random.random() * self.width
            y = np.random.random() * self.height
            if not self.env.check_collision(x, y):
                return Treenode(x, y)

    def extend(self, q_rand):
        """
        Takes a tree and a target configuration q_rand as input.
        Attempts to grow the tree towards q_rand.
        """
        q_near, distance = self.nearest_neighbour(q_rand)
        q_new = self.steer(q_near, q_rand, distance)
        if q_new:
            self.treenodes.add(q_new)
            self.treenode_parents[q_new] = q_near
            q_near.children.append(q_new)

    def nearest_neighbour(self, q_rand):
        """ Returns the nearest node to q_rand within the tree. """
        nearest = None
        min_distance = float('inf')
        for node in self.treenodes:
            distance = node.get_euclidean_distance(q_rand)
            if distance < min_distance:
                nearest = node
                min_distance = distance
        return nearest, min_distance

    def valid_segment(self, node1, node2):
        """
        Checks for collisions in the straight line joining two nodes.
        Returns True for no collisions, False otherwise.
        """
        x_coords = np.linspace(node1.x, node2.x, 302)[1:-1]
        y_coords = np.linspace(node1.y, node2.y, 302)[1:-1]
        for x, y in zip(x_coords, y_coords):
            if self.env.check_collision(x, y):
                return False
        return True

    def steer(self, q_near, q_rand, distance):
        """
        Determines whether it is possible to extend the tree towards q_rand
        via q_near.
        Returns q_near if distance <= max_radius, q_new(point on the line between
        q_near and q_rand) if distance > max_radius, otherwise None if extension of
        tree is impossible.
        """
        if distance <= self.radius:
            q_new = q_rand
        else:
            scale = self.radius / distance
            q_new_x = q_near.x + scale * (q_rand.x - q_near.x)
            q_new_y = q_near.y + scale * (q_rand.y - q_near.y)
            q_new = Treenode(q_new_x, q_new_y)

        if self.valid_segment(q_near, q_new):
            return q_new
        return None

    def plot_tree(self):
        x_coords = [node.x for node in self.treenodes]
        y_coords = [node.y for node in self.treenodes]
        pl.scatter(x_coords, y_coords, s=8, marker='.', color='cyan')
        for node in self.treenodes:
            for child in node.children:
                pl.plot([node.x, child.x],
                        [node.y, child.y], 'c', linewidth=0.3)

    def plot_path(self, midpoint):
        current = midpoint
        parent = self.treenode_parents.get(current, None)
        while parent is not None:
            pl.plot([parent.x, current.x],
                    [parent.y, current.y], 'k', linewidth=1)
            current = parent
            parent = self.treenode_parents.get(current, None)
