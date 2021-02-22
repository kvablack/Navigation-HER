import torch
import numpy as np
import os
from dqn_HER import DQN_HER
import time
import json
from collections import deque, defaultdict
import pickle

HYPERPARAMETERS = {
    "grid_size": 20,
    "num_obstacles": 15,
    "obstacle_diameter": 2,
    "min_goal_dist": 10,
    "gamma": 0.99,
    "replay_buffer_size": 500000,
    "use_ddqn": True,
    "epochs": 15000,
    "use_factorized_state": True,
    "learning_rate": 0.0001,
    "max_episode_length": 50,
    "batch_size": 16,
    "epsilon_high": 0.9,
    "epsilon_low": 0.1,
    "epsilon_decay": 2000,
    "steps_between_model_swap": 3000,
    "image_norm_episodes": 5000,
}


def train(h, save_dir):
    os.makedirs(save_dir, exist_ok=False)
    with open(os.path.join(save_dir, "hyperparameters.json"), "w") as f:
        json.dump(h, f)
    alg = DQN_HER(h)
    successes = deque(maxlen=100)
    results = defaultdict(list)

    for i in range(h["epochs"]):
        start = time.time()
        total_reward, avg_loss, min_dist = alg.run_episode()
        finish = time.time()
        successes.append(min_dist == 0)
        success_rate = np.mean(successes)
        print(
            f"done: {i} of {h['epochs']}. loss: {avg_loss:.2f}. success rate: {success_rate:.2f}. time: {finish - start:.3f}. steps: {alg.steps}"
        )
        results["total_reward"].append(total_reward)
        results["min_dist"].append(min_dist)
        results["steps"].append(alg.steps)
        results["avg_loss"].append(avg_loss)
        if i % 100 == 0:
            torch.save(alg.model.state_dict(), os.path.join(save_dir, "model.pt"))
            if not h["use_factorized_state"]:
                torch.save(alg.image_mean, os.path.join(save_dir, "image_mean.pt"))
            with open(os.path.join(save_dir, "results.pickle"), "wb") as f:
                pickle.dump(results, f)


train(HYPERPARAMETERS, "test_factorized")
