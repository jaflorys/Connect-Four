from gc import callbacks
from pyclbr import Function
from tkinter import N
import numpy as np
from stable_baselines3.common.env_checker import check_env
from connect_four_dual import ConnectFour
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from copy import deepcopy
import os
from typing import Union
from gym import Env

from learning_schedules import constant_schedule
from itertools import product


def make_env(
    n_rows: int, n_cols: int, move_first: bool, opponent_model: Union[DQN, None]
):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = ConnectFour(n_rows, n_cols, move_first, opponent_model)
        return env

    return _init


def play_episode(model: DQN, env: Env):
    states = []
    rewards = []
    state = env.reset()
    states.append(deepcopy(env.state))

    done = False
    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, info = env.step(action)
        states.append(deepcopy(env.state))
        rewards.append(reward)

    return rewards, states, info


def self_play(*, settings: dict, id: str):

    # Unpack settings
    n_rows = settings["n_rows"]
    n_cols = settings["n_cols"]
    move_first = settings["move_first"]
    self_model = settings["self_model"]
    opponent_model = settings["opponent_model"]
    net_arch = settings["net_arch"]
    initial_learning_rate = settings["initial_learning_rate"]
    exploration_initial_eps = settings["exploration_initial_eps"]
    exploration_final_eps = settings["exploration_final_eps"]
    exploration_fraction = settings["exploration_fraction"]
    n_cpus = settings["n_cpus"]
    total_timesteps = settings["total_timesteps"]
    learning_schedule = constant_schedule(learning_rate=initial_learning_rate)

    policy_kwargs = {
        "net_arch": net_arch,
    }

    # Create opponent model
    if not opponent_model == None:
        opponent_model = DQN.load(settings["opponent_model"])

    dummy_env = ConnectFour(
        n_rows=n_rows,
        n_cols=n_cols,
        move_first=move_first,
        opponent_model=opponent_model,
    )

    check_env(
        env=dummy_env, warn=True,
    )

    # Now create vectorized environments for training and evaluation
    train_env = SubprocVecEnv(
        [
            make_env(
                n_rows=n_rows,
                n_cols=n_cols,
                move_first=True,
                opponent_model=opponent_model,
            )
            for i in range(n_cpus)
        ]
    )

    eval_env = SubprocVecEnv(
        [
            make_env(
                n_rows=n_rows,
                n_cols=n_cols,
                move_first=True,
                opponent_model=opponent_model,
            )
            for i in range(n_cpus)
        ]
    )

    # Create self model if it is 'None'
    if self_model == None:
        self_model = DQN(
            policy="MlpPolicy",
            learning_rate=learning_schedule,
            env=train_env,
            verbose=1,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            tensorboard_log=os.path.join("./log", id),
            policy_kwargs=policy_kwargs,
        )
    else:
        self_model = DQN(
            policy="MlpPolicy",
            learning_rate=learning_schedule,
            env=train_env,
            verbose=1,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            tensorboard_log=os.path.join("./log", id),
            policy_kwargs=policy_kwargs,
        )
        self_model.load(settings["self_model"])

    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=os.path.join("./models", id),
        eval_freq=2000,
        log_path=os.path.join("./logs", id),
        deterministic=True,
        render=False,
    )
    self_model.learn(
        total_timesteps=total_timesteps, log_interval=4, callback=eval_callback
    )
    self_model.save(os.path.join("./models", id, "end_model"))

    train_env.close()
    eval_env.close()
    dummy_env.close()
    if not opponent_model == None:
        del opponent_model
    return self_model


def main():

    settings = {
        "num_training_sets": 3,
        "continue_training": False,
        "n_rows": 6,
        "n_cols": 7,
        "net_arch": [1024, 1024],
        "initial_learning_rate": 1e-4,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "exploration_fraction": 0.95,
        "n_cpus": 4,
        "total_timesteps": 10 ** 4,  # 0.5e6,
        "learning_schedule": constant_schedule,
    }

    continue_training = settings["continue_training"]
    num_training_sets = settings["num_training_sets"]

    # If continue training find last epoch completed
    start_idx = 0
    last_model = None
    if continue_training:
        model_files = os.listdir(os.path.join(os.getcwd(), "./models"))
        model_prefixes = list(set([int(file.split("_")[0]) for file in model_files]))
        model_prefixes.sort(reverse=True)
        last_model = model_prefixes[0]
        fully_complete = True if str(last_model) + "_" + "1" in model_files else False
        start_idx = last_model + 1 if fully_complete else last_model

    # Get set of all runs to be done
    epochs = np.arange(start_idx, start_idx + num_training_sets + 1)
    modes = np.arange(2)
    runs = product(epochs, modes)

    if continue_training and fully_complete:
        last_id = str(last_model) + "_1"
    elif continue_training and not fully_complete:
        last_id = str(last_model - 1) + "_1"
    else:
        last_id = None

    for run in runs:
        epoch, mode = run
        """
        Skip training step if:
        (1) continuing training and not fully complete, and
        (2) on incomplete epoch
        (3) on completed mode
        """
        skip_epoch = (
            (continue_training)
            and (not fully_complete)
            and (epoch == start_idx)
            and (mode == 0)
        )
        if not skip_epoch:
            id = str(epoch) + "_" + str(mode)
            settings["move_first"] = True if mode == 0 else False

            if epoch == 0:
                settings["self_model"] = None
                settings["opponent_model"] = None
            else:
                settings["self_model"] = os.path.join("./models", last_id, "end_model")
                settings["opponent_model"] = settings["self_model"]
            breakpoint()
            model = self_play(settings=settings, id=id)
            del model
            if mode == 1:
                last_id = id

    """
    R = []
    I = []
    S = []
    for i in np.arange(1000):
        rewards, states, info = play_episode(
            model=model,
            env=ConnectFour(
                n_rows=n_rows, n_cols=n_cols, move_first=True, opponent_model=None
            ),
        )
        R.append(np.mean(rewards))
        I.append(info["done_state"])
        S.append(states)
    """


if __name__ == "__main__":
    main()
