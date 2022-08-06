import numpy as np
import os

from connect_four_env import ConnectFour
from gym import Env
from stable_baselines3 import DQN


def get_move_first_input():
    while True:
        print("Do you want to go first? (y/n/q)")
        usr_input = input().lower()
        if usr_input[0] == "q":
            exit()
        elif usr_input[0] == "y":
            return True
        elif usr_input[0] == "n":
            return False


def get_training_epoch(*, max_epoch: int, model_files: list):
    while True:
        print("Which training epoch? (0-" + str(max_epoch) + ")")
        prefix = input().lower()
        if prefix[0] == "q":
            exit()
        if prefix in model_files:
            return prefix


def get_player_move(*, env: Env):
    while True:
        print("Which column do you wish to drop your piece (1-" + str(env.n_cols) + ")")
        while True:
            usr_input = input().lower()
            if usr_input.isnumeric():
                is_integer = (float(usr_input) - int(usr_input)) == 0
                if is_integer:
                    usr_input = int(usr_input) - 1
                    if usr_input in env.available_columns:
                        return usr_input


def main():
    move_first = get_move_first_input()

    # Determine available training epochs
    models_path = os.path.join(os.getcwd(), "models")
    if not os.path.exists(models_path):
        print("No available models.")
        exit()
    model_files = os.listdir(models_path)
    available_epochs = list([int(f) for f in model_files])
    available_epochs.sort()
    max_epoch = np.max(available_epochs)

    # Get which degree of trained model to load
    prefix = get_training_epoch(max_epoch=max_epoch, model_files=model_files)
    model_file_path = os.path.join(models_path, prefix, "end_model")

    # Instantiate environment
    env = ConnectFour(
        n_rows=6,
        n_cols=7,
        move_first=move_first,
        deterministic_opponent=True,
        opponent_models=[model_file_path],
        max_model_history=None,
        probability_switch_model=0,
        id=1,
    )
    done = False
    while not done:
        print(env.state)
        _, _, done, info = env.step(get_player_move(env=env))
    print("Game over: ")
    print(env.state)
    done_state = info["done_state"]
    if done_state == 1:
        print("YOU WIN!!!")
    elif done_state == -1:
        print("YOU LOSE!!!")
    else:
        print("TIE GAME!!!")


if __name__ == "__main__":
    main()
