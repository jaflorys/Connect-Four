import copy
import gc
import os
import learning_schedules
import numpy as np
import shutil

from connect_four_env import ConnectFour
from gym import Env
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from typing import Union


def make_env(
    n_rows: int,
    n_cols: int,
    move_first: Union[None, bool],
    deterministic_opponent: bool,
    opponent_models: Union[None, list],
    max_model_history: Union[None, int],
    probability_switch_model: float,
    switch_method: Union[None, str],
    id: int,
):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = ConnectFour(
            n_rows,
            n_cols,
            move_first,
            deterministic_opponent,
            opponent_models,
            max_model_history,
            probability_switch_model=probability_switch_model,
            switch_method=switch_method,
            id=id,
        )
        return env

    return _init


def play_episode(model: DQN, env: Env):
    states = []
    rewards = []
    state = env.reset()
    states.append(copy.deepcopy(env.state))

    done = False
    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, info = env.step(action)
        states.append(copy.deepcopy(env.state))
        rewards.append(reward)

    return rewards, states, info


def self_play(*, settings: dict, id: str):

    """Trains an DQN policy against randomly selected past policies.

    Trains an agent using deep-Q learning (DQN) against randomly selected
    policies. Vectorized training and evaluation environments are created
    and have file paths to randomly load existing policies as opponents
    against the policy being trained. The function returns the trained
    policy. 


    Args:
        settings: An dictionary of training settings.
        id: A integer that identifies the current policy
        being trained. Lower integers indicated older policies.

    Returns:
        A trained DQN model.

    Raises:
        IOError: An error occurred accessing the smalltable.
    """

    # Unpack settings
    n_rows = settings["n_rows"]
    n_cols = settings["n_cols"]
    self_model = settings["self_model"]
    opponent_models = settings["opponent_model"]
    max_model_history = settings["max_model_history"]
    probability_switch_model = settings["probability_switch_model"]
    switch_method = settings["switch_method"]
    net_arch = settings["net_arch"]
    initial_learning_rate = settings["initial_learning_rate"]
    exploration_initial_eps = settings["exploration_initial_eps"]
    exploration_final_eps = settings["exploration_final_eps"]
    exploration_fraction = settings["exploration_fraction"]
    num_cpus_train = settings["num_cpus_train"]
    num_cpus_eval = settings["num_cpus_eval"]
    total_timesteps = settings["total_timesteps"]
    learning_schedule = learning_schedules.linear_schedule(
        initial_learning_rate=initial_learning_rate
    )

    policy_kwargs = {
        "net_arch": net_arch,
    }

    dummy_env = ConnectFour(
        n_rows=n_rows,
        n_cols=n_cols,
        move_first=None,
        deterministic_opponent=True,
        opponent_models=opponent_models,
        max_model_history=max_model_history,
        probability_switch_model=1,
        switch_method=switch_method,
        id=-1,
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
                move_first=None,
                deterministic_opponent=True,
                opponent_models=opponent_models,
                max_model_history=max_model_history,
                probability_switch_model=probability_switch_model,
                switch_method=switch_method,
                id=i,
            )
            for i in range(num_cpus_train)
        ]
    )

    eval_env = SubprocVecEnv(
        [
            make_env(
                n_rows=n_rows,
                n_cols=n_cols,
                move_first=None,
                deterministic_opponent=True,
                opponent_models=opponent_models,
                max_model_history=max_model_history,
                probability_switch_model=probability_switch_model,
                switch_method=switch_method,
                id=i,
            )
            for i in range(num_cpus_eval)
        ]
    )

    """train_env = ConnectFour(
        n_rows=n_rows,
        n_cols=n_cols,
        move_first=None,
        deterministic_opponent=True,
        opponent_models=opponent_models,
        max_model_history=max_model_history,
        probability_switch_model=probability_switch_model,
        switch_method=switch_method,
        id=1,
    )

    eval_env = ConnectFour(
        n_rows=n_rows,
        n_cols=n_cols,
        move_first=None,
        deterministic_opponent=True,
        opponent_models=opponent_models,
        max_model_history=max_model_history,
        probability_switch_model=probability_switch_model,
        switch_method=switch_method,
        id=1,
    )"""

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
    breakpoint()
    train_env.close()
    eval_env.close()
    dummy_env.close()
    return self_model


def main():

    settings = {
        "num_training_sets": 1,
        "continue_training": True,
        "n_rows": 6,
        "n_cols": 7,
        "max_model_history": None,
        "probability_switch_model": 1.0,
        "switch_method": "ucb",  # "inverse_history_length",
        "net_arch": [1024, 1024],
        "initial_learning_rate": 1e-4,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "exploration_fraction": 0.95,
        "num_cpus_train": 2,
        "num_cpus_eval": 1,
        "total_timesteps": 2.0e6,
    }

    continue_training = settings["continue_training"]
    num_training_sets = settings["num_training_sets"]

    # If continue training find last epoch completed
    start_idx = 0
    last_model = None
    num_cpus = settings["num_cpus_train"]
    if continue_training:
        # Find the last fully-complete epoch
        model_files = os.listdir(os.path.join(os.getcwd(), "./models"))
        model_prefixes = [int(file) for file in model_files]
        model_prefixes.sort(reverse=True)
        last_model = model_prefixes[0]
        start_idx = last_model + 1

        # Copy model files for each cpu to avoid collisions
        for model_file in model_files:
            for i in np.arange(num_cpus):
                model_file_orig = os.path.join(
                    os.getcwd(), "./models", model_file, "end_model.zip"
                )
                model_file_cpu = os.path.join(
                    os.getcwd(), "./models", model_file, "end_model_" + str(i) + ".zip"
                )
                if not os.path.exists(model_file_cpu):
                    shutil.copy(model_file_orig, model_file_cpu)

    # Get set of all runs to be done
    epochs = np.arange(start_idx, start_idx + num_training_sets)

    if continue_training:
        last_id = str(last_model)
    else:
        last_id = None

    opponent_models = []
    if continue_training:
        opponent_models = [
            os.path.join("./models", str(prefix), "end_model")
            for prefix in model_prefixes
        ]

    for epoch in epochs:
        id = str(epoch)

        if epoch == 0:
            settings["self_model"] = None
            settings["opponent_model"] = None
        else:
            self_model_path = os.path.join("./models", last_id, "end_model")
            settings["self_model"] = self_model_path
            settings["opponent_model"] = opponent_models

        model = self_play(settings=settings, id=id)
        del model

        opponent_models.append(os.path.join("./models", id, "end_model"))
        last_id = id

        # Copy model file for use by each environment
        model_file_orig = os.path.join(os.getcwd(), "./models", id, "end_model.zip")
        for i in np.arange(num_cpus):
            model_file_cpu = os.path.join(
                os.getcwd(), "./models", id, "end_model_" + str(i) + ".zip"
            )
            shutil.copyfile(model_file_orig, model_file_cpu)

        gc.collect()


if __name__ == "__main__":
    main()
