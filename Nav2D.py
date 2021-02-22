import torch
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib as plt
from copy import deepcopy


class Navigate2D:
    def __init__(self, size, n_obs, d_obs, goal_dist_min):
        self.size = size
        self.n_obs = n_obs
        self.d_obs = d_obs
        self.goal_dist_min = goal_dist_min
        self.factorized_state_dim = [n_obs + 2, 2]
        self.grid_state_dim = [3, size, size]
        self.action_dim = 4
        self.scale = 10

        self.actions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=int)

        self.factorized = None
        self.grid = None
        self.dist = None
        self.buffer = None

    def reset(self):
        grid = np.zeros((3, self.size, self.size), dtype=np.uint8)
        obs = np.zeros((self.n_obs, 2), dtype=np.uint8)
        for i in range(self.n_obs):
            center = np.random.randint(0, self.size, 2)
            obs[i] = center
            minX = np.maximum(center[0] - self.d_obs, 1)
            minY = np.maximum(center[1] - self.d_obs, 1)
            maxX = np.minimum(center[0] + self.d_obs, self.size - 1)
            maxY = np.minimum(center[1] + self.d_obs, self.size - 1)
            grid[0, minX:maxX, minY:maxY] = 1

        free_idx = np.argwhere(grid[0, :, :] == 0)
        start = free_idx[np.random.randint(0, free_idx.shape[0], 1), :].squeeze()
        while True:
            finish = free_idx[np.random.randint(0, free_idx.shape[0], 1), :].squeeze()
            if (
                not np.all(start == finish)
                and np.linalg.norm(start - finish) >= self.goal_dist_min
            ):
                break
        grid[1, start[0], start[1]] = self.scale
        grid[2, finish[0], finish[1]] = self.scale

        self.factorized = np.concatenate([start[None, ...], finish[None, ...], obs])
        self.grid = grid
        self.dist = np.linalg.norm(start - finish)
        self.buffer = []

    def step(self, action):
        old_grid = self.grid.copy()
        old_factorized = self.factorized.copy()
        pos = self.factorized[0]
        target = self.factorized[1]
        new_pos = pos + self.actions[action]
        reward = -1

        if (
            np.all(new_pos >= 0)
            and np.all(new_pos < self.size)
            and not self.grid[0, new_pos[0], new_pos[1]]
        ):
            self.dist = np.linalg.norm(new_pos - target)
            self.grid[1, pos[0], pos[1]] = 0.0
            self.grid[1, new_pos[0], new_pos[1]] = self.scale
            self.factorized[0] = new_pos
            if np.all(new_pos == target):
                reward = 0

        self.buffer.append([old_grid, old_factorized, action, reward])
        return reward

    def state(self, factorized=False):
        if factorized:
            return self.factorized.copy()
        else:
            return self.grid.copy()

    def her(self, factorized=False):
        goal_grid = self.grid[1, :, :]
        goal_factorized = self.factorized[0]
        for i in range(len(self.buffer)):
            self.buffer[-i - 1][0][2, :, :] = goal_grid
            self.buffer[-i - 1][1][1] = goal_factorized
            if i == 0 or np.all(self.buffer[-i][1][0] == goal_factorized):
                self.buffer[-1 - i][3] = 0
            else:
                self.buffer[-i - 1][3] = -1

        if factorized:
            ret = [[f, a, r] for g, f, a, r in self.buffer]
        else:
            ret = [[g, a, r] for g, f, a, r in self.buffer]
        self.buffer = None
        return ret

    def render(self):
        plot = imshow(self.grid)
        return plot
