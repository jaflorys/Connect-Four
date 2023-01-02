import gc
import os
import numpy as np
import shutil

from connect_four_env import ConnectFourEnv
from gym import Env
from stable_baselines3.dqn.dqn import DQN
from typing import DefaultDict
from utils import play_matches


def main(*, settings):
    num_steps = settings["num_steps"]
    c = settings["c"]

    # Determine available training epochs
    models_path = os.path.join(os.getcwd(), "models")
    if not os.path.exists(models_path):
        print("No available models.")
        exit()
    if len(models_path) <= 1:
        print("Must have more than one policy.")
        exit()

    policy_file_paths = [
        os.path.join(models_path, path, "end_model") for path in os.listdir(models_path)
    ]

    # Create policy map, reward estimates, and counts
    policy_map = DefaultDict(int)
    for i, policy_name in enumerate(policy_file_paths):
        policy_map[i] = policy_name

    num_policies = len(policy_file_paths)
    Q_t = np.ones(num_policies) * np.inf
    N_t = np.zeros(num_policies)

    rewards = []
    actions = []
    for step in np.arange(num_steps):
        t = step + 1
        print(
            "Epoch " + str(t) + " of " + str(num_steps), end="\r", flush=True,
        )

        # Select policy with highest UCB
        action, _ = compute_ucb(t=t, Q=Q_t, N=N_t, c=c)
        policy_path = policy_file_paths[action]
        policy = DQN.load(policy_file_paths[action])

        # Select a random opposing policy (non-self)
        valid = False
        while not valid:
            opponent_policy_path = np.random.choice(policy_file_paths, 1)[0]
            if not opponent_policy_path == policy_path:
                valid = True
        opponent_policy = [opponent_policy_path]
        env = ConnectFourEnv(
            rows=6,
            cols=7,
            move_first=None,
            opponent_models=opponent_policy,
            deterministic_opponent=True,
            max_model_history=None,
            probability_switch_model=0,
            switch_method=None,
            id=1,
        )

        num_wins = play_matches(
            player_policy=policy, deterministic_player=True, env=env, num_games=1
        )
        reward = 0
        reward = 1 if num_wins[0] == 1 else (-1 if num_wins[0] == 1 else 0)

        Q_t[action] = (
            reward
            if Q_t[action] == np.inf
            else Q_t[action] + 1 / t * (reward - Q_t[action])
        )
        N_t[action] += 1

        rewards.append(reward)
        actions.append(action)

        env.close()
        gc.collect()

    pct_action = N_t / np.sum(N_t)

    print("\n")
    print("Estimated action-reward values:")
    for action in policy_map:
        print(str(action) + ": " + "{:.3f}".format(Q_t[action]))
    print("\n")
    print("Percent actions selected: ")
    for action in policy_map:
        print(str(action) + ": " + "{:.2%}".format(pct_action[action]))

    # Select all policies with above median performance
    median_value = np.median(Q_t)
    best_policy_folder = "ucb_best_policies"
    if os.path.exists(best_policy_folder):
        shutil.rmtree(best_policy_folder)
    os.mkdir(best_policy_folder)

    # Copy the best policies into the folder
    idx = 0
    for i, q in enumerate(Q_t):
        if q >= median_value:
            policy_dir = best_policy_folder + "////" + str(idx)
            os.mkdir(policy_dir)
            copy_file_path = os.path.join(policy_dir, "end_model" + ".zip")
            shutil.copy(policy_file_paths[i] + ".zip", copy_file_path)
            idx += 1


def compute_ucb(t: int, Q: np.ndarray, N: np.ndarray, c: float):
    ucb = Q + c * np.sqrt(np.log(t) * 1.0 / (N + 1e-9))
    action = np.argmax(ucb)
    return action, ucb


if __name__ == "__main__":
    settings = {
        "num_steps": 10000,
        "c": 2,
    }
    main(settings=settings)

