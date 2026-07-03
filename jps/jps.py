import numpy as np
import matplotlib.pyplot as plt
import heapq
import random


import numpy as np
import matplotlib.pyplot as plt
import heapq
import random
import math


class JPS:
    def __init__(self, grid):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        self.visited = set()

    def in_bounds(self, x, y):
        return 0 <= x < self.rows and 0 <= y < self.cols

    def is_passable(self, x, y):
        return self.grid[x, y] == 0

    def jump(self, x, y, dx, dy, goal):
        nx, ny = x + dx, y + dy
        if not self.in_bounds(nx, ny) or not self.is_passable(nx, ny):
            return None
        if (nx, ny) == goal:
            return (nx, ny)

        if dx != 0 and dy != 0:
            if (self.in_bounds(nx - dx, ny) and not self.is_passable(nx - dx, ny) and
                self.in_bounds(nx - dx, ny + dy) and self.is_passable(nx - dx, ny + dy)):
                return (nx, ny)
            if (self.in_bounds(nx, ny - dy) and not self.is_passable(nx, ny - dy) and
                self.in_bounds(nx + dx, ny - dy) and self.is_passable(nx + dx, ny - dy)):
                return (nx, ny)
        else:
            if dx != 0:
                if (self.in_bounds(nx, ny + 1) and not self.is_passable(nx, ny + 1) and
                    self.in_bounds(nx + dx, ny + 1) and self.is_passable(nx + dx, ny + 1)):
                    return (nx, ny)
                if (self.in_bounds(nx, ny - 1) and not self.is_passable(nx, ny - 1) and
                    self.in_bounds(nx + dx, ny - 1) and self.is_passable(nx + dx, ny - 1)):
                    return (nx, ny)
            if dy != 0:
                if (self.in_bounds(nx + 1, ny) and not self.is_passable(nx + 1, ny) and
                    self.in_bounds(nx + 1, ny + dy) and self.is_passable(nx + 1, ny + dy)):
                    return (nx, ny)
                if (self.in_bounds(nx - 1, ny) and not self.is_passable(nx - 1, ny) and
                    self.in_bounds(nx - 1, ny + dy) and self.is_passable(nx - 1, ny + dy)):
                    return (nx, ny)

        if dx != 0 and dy != 0:
            if self.jump(nx, ny, dx, 0, goal) or self.jump(nx, ny, 0, dy, goal):
                return (nx, ny)

        return self.jump(nx, ny, dx, dy, goal)

    def find_neighbors(self, node):
        x, y = node
        neighbors = []
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny) and self.is_passable(nx, ny):
                neighbors.append((dx, dy))
        return neighbors

    def heuristic(self, a, b):
        return math.sqrt(abs(a[0] - b[0])**2 + abs(a[1] - b[1])**2 )

    def find_path(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (0 + self.heuristic(start, goal), 0, start, [start]))
        self.visited.clear()

        while open_list:
            f, g, current, path = heapq.heappop(open_list)
            if current == goal:
                return path, len(self.visited)

            self.visited.add(current)

            for dx, dy in self.find_neighbors(current):
                jump_point = self.jump(current[0], current[1], dx, dy, goal)
                if jump_point and jump_point not in self.visited:
                    new_g = g + self.heuristic(current, jump_point)
                    heapq.heappush(open_list, (
                        new_g + self.heuristic(jump_point, goal),
                        new_g,
                        jump_point,
                        path + [jump_point]
                    ))
        return None, len(self.visited)
