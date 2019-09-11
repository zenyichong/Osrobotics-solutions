#!/usr/bin/env python
import numpy as np
import pylab as pl


class Node:
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.neighbours = []
        self.id = id


class PRM:
    def __init__(self, env, num_nodes, max_neighbours, radius):
        self.env = env
        self.width = env.size_x
        self.height = env.size_y
        self.start = Node(env.start[0], env.start[1], 'start')
        self.goal = Node(env.goal[0], env.goal[1], 'goal')
        self.nodes = []
        self.num_nodes = num_nodes
        self.max_neighbours = max_neighbours
        self.radius = radius

    def random_sampling(self):
        """
        Takes n random samples within the configuration space
        and assigns them as valid nodes if they are in Cfree.
        """
        n = 0
        while n < self.num_nodes:
            x = np.random.random() * self.width
            y = np.random.random() * self.height
            if not self.env.check_collision(x, y):
                self.nodes.append(Node(x, y, n))
                n += 1

    def find_closest_neighbours(self):
        """
        Finds a maximum of k closest neighbours for each node and
        sorts them according to their euclidean distance.
        """
        for node in self.nodes:
            distances = []
            for other in self.nodes:
                distance = self.calc_distance(node, other)
                if distance <= self.radius:
                    if self.valid_segment(node, other):
                        distances.append((other, distance))
            distances = sorted(distances, key=lambda x: x[1])
            node.neighbours = distances[:self.max_neighbours]

    def solve_query(self):
        """
        Connects the start and goal nodes to the PRM and solves for
        the shortest distance.
        Returns path if valid path can be found, False otherwise.
        """
        candidates = []
        for node in self.nodes:
            dist = self.calc_distance(self.start, node)
            if dist < self.radius:
                if self.valid_segment(self.start, node):
                    candidates.append((node, dist))
        source = sorted(candidates, key=lambda x: x[1])[0][0]

        candidates = []
        for node in self.nodes:
            dist = self.calc_distance(self.goal, node)
            if dist < self.radius:
                if self.valid_segment(self.goal, node):
                    candidates.append((node, dist))
        dest = sorted(candidates, key=lambda x: x[1])[0][0]

        path = self.shortest_path(source, dest)
        if not path:
            print('Path not found.')
        else:
            path = [self.start] + path + [self.goal]
            self.print_path(path)
        return path

    def shortest_path(self, source, dest):
        """
        Helper function:
        Uses Dijkstra's algorithm to find the shortest path
        (not considering start and goal nodes).
        """
        distance = [float('Inf')] * self.num_nodes
        parent = [-1] * self.num_nodes
        visited = [False] * self.num_nodes
        distance[source.id] = 0

        while True:
            if (sum(visited) == self.num_nodes) or visited[dest.id]:
                break
            low = float('Inf')
            min_id = -1
            for i in range(self.num_nodes):
                if not visited[i]:
                    if (min_id == -1) or (distance[i] < low):
                        low = distance[i]
                        min_id = i
            visited[min_id] = True

            for elem in self.nodes[min_id].neighbours:
                other, dist = elem
                curr_dist = distance[min_id] + dist
                if (curr_dist < distance[other.id]):
                    distance[other.id] = curr_dist
                    parent[other.id] = min_id

        index = dest.id
        path = []
        while True:
            if index == source.id:
                break
            if parent[index] == -1:
                return False
            path.append(self.nodes[index])
            index = parent[index]
        return path[::-1]      # reverses order to get path from start to goal

    def valid_segment(self, node1, node2):
        """
        Helper function:
        Checks for collisions in the straight line joining two nodes.
        Returns True for no collisions, False otherwise.
        """
        x_coords = np.linspace(node1.x, node2.x, 102)[1:-1]
        y_coords = np.linspace(node1.y, node2.y, 102)[1:-1]
        for x, y in zip(x_coords, y_coords):
            if self.env.check_collision(x, y):
                return False
        return True

    def calc_distance(self, node1, node2):
        return ((node2.x - node1.x) ** 2 + (node2.y - node1.y) ** 2) ** 0.5

    def print_path(self, path):
        """
        Helper function:
        Prints the nodes and their coordinates in the shortest path in order.
        """
        for i, node in enumerate(path, 1):
            print(f'{i: <2}. Node '
                  + f'({node.id})'.ljust(7)
                  + f': X: {node.x:6f}  Y: {node.y:6f}')

    def plot_nodes(self):
        x_coords = [node.x for node in self.nodes]
        y_coords = [node.y for node in self.nodes]
        pl.scatter(x_coords, y_coords, s=6, marker='.', color='cyan')

    def plot_edges(self):
        for n1 in self.nodes:
            for n2 in n1.neighbours:
                pl.plot([n1.x, n2[0].x], [n1.y, n2[0].y], 'c', linewidth=0.2)

    def plot_path(self, path):
        if not path:
            return None
        for i in range(len(path) - 1):
            n1, n2 = path[i], path[i + 1]
            pl.plot([n1.x, n2.x], [n1.y, n2.y], 'k', linewidth=1)
