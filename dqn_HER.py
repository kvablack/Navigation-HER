import torch
import numpy as np
import torch.nn.functional as F
from collections import deque
from Models import ConvNet, MLP
import random
import time
import tqdm
from log_utils import logger, mean_val
from copy import deepcopy
from Nav2D import Navigate2D


class DQN_HER:
    def __init__(self, h):
        self.env = Navigate2D(
            h["grid_size"],
            h["num_obstacles"],
            h["obstacle_diameter"],
            h["min_goal_dist"],
        )
        self.factorized = h["use_factorized_state"]
        self.max_episode_length = h["max_episode_length"]
        if self.factorized:
            self.model = MLP(self.env.factorized_state_dim, self.env.action_dim).cuda()
        else:
            self.model = ConvNet(self.env.grid_state_dim, self.env.action_dim).cuda()
            self.calc_norm(h["image_norm_episodes"])
        self.target_model = deepcopy(self.model).cuda()
        self.gamma = h["gamma"]
        self.ddqn = h["use_ddqn"]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=h["learning_rate"]
        )

        self.batch_size = h["batch_size"]
        self.epsilon_high = h["epsilon_high"]
        self.epsilon_low = h["epsilon_low"]
        self.epsilon_decay = h["epsilon_decay"]
        self.steps_between_model_swap = h["steps_between_model_swap"]

        self.replay_buffer = deque(maxlen=h["replay_buffer_size"])
        self.image_mean = 0
        self.image_std = 0
        self.steps_until_model_swap = 0
        self.steps = 0
        self.epsilon = self.epsilon_high

    def run_episode(self):
        self.env.reset()
        state = self.env.state(factorized=self.factorized)
        sum_r = 0
        total_loss = 0
        min_dist = self.env.dist

        for t in range(self.max_episode_length):
            self.steps += 1
            self.epsilon = self.epsilon_low + (self.epsilon_high - self.epsilon_low) * (
                np.exp(-1.0 * self.steps / self.epsilon_decay)
            )
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.env.action_dim)
            else:
                with torch.no_grad():
                    Q = self.model(
                        self.norm(
                            torch.tensor(state, dtype=torch.float).cuda().unsqueeze(0)
                        )
                    ).squeeze(0)
                    action = torch.argmax(Q).item()
            reward = self.env.step(action)
            new_state = self.env.state(factorized=self.factorized)
            sum_r += reward
            min_dist = min(min_dist, self.env.dist)

            self.replay_buffer.append([state, action, reward, new_state])
            loss = self.update_model()
            total_loss += loss
            state = new_state

            self.steps_until_model_swap += 1
            if self.steps_until_model_swap > self.steps_between_model_swap:
                self.target_model.load_state_dict(self.model.state_dict())
                self.steps_until_model_swap = 0
                print("updated target model")

            if reward == 0:
                break

        if reward != 0:
            her = self.env.her(factorized=self.factorized)
            for (state, action, reward), (new_state, _, _) in zip(her, her[1:]):
                self.replay_buffer.append([state, action, reward, new_state])
            self.replay_buffer.append(
                her[-1] + [self.env.state(factorized=self.factorized)]
            )

        avg_loss = total_loss / t
        return sum_r, avg_loss, min_dist

    def calc_norm(self, num_episodes):
        states = []
        for _ in tqdm.tqdm(range(num_episodes)):
            self.env.reset()
            for _ in range(self.max_episode_length):
                state = self.env.state(factorized=False)
                self.env.step(np.random.randint(self.env.action_dim))
                states.append(state)

        states = torch.tensor(states, dtype=torch.float).cuda()
        self.image_mean = states.mean(dim=0)
        self.image_std = states.std(dim=0)

    def norm(self, state):
        if self.factorized:
            half_size = self.env.size / 2
            return (state - half_size) / half_size
        else:
            return state - self.image_mean

    def update_model(self):
        self.optimizer.zero_grad()
        K = min(len(self.replay_buffer), self.batch_size)
        samples = random.sample(self.replay_buffer, K)

        states, actions, rewards, new_states = zip(*samples)
        states = torch.tensor(states, dtype=torch.float).cuda()
        actions = torch.tensor(actions, dtype=torch.long).cuda()
        rewards = torch.tensor(rewards, dtype=torch.float).cuda()
        new_states = torch.tensor(new_states, dtype=torch.float).cuda()

        states = self.norm(states)
        new_states = self.norm(new_states)

        if self.ddqn:
            model_next_acts = self.model(new_states).detach().max(dim=1)[1]
            target_q = rewards + self.gamma * self.target_model(new_states).gather(
                1, model_next_acts.unsqueeze(1)
            ).squeeze() * (rewards == -1)
        else:
            target_q = rewards + self.gamma * self.target_model(new_states).max(dim=1)[
                0
            ].detach() * (rewards == -1)
        policy_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        L = F.smooth_l1_loss(policy_q, target_q)
        L.backward()
        self.optimizer.step()
        return L.detach().item()
