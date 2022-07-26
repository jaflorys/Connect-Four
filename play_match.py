import numpy as np
import os

from stable_baselines3 import DQN
from connect_four_play import ConnectFour


def main():
    is_valid = False
    go_first = None
    while not is_valid:
        print("Do you want to go first? (y/n/q)")
        usr_input = input()
        if usr_input[0] == "q":
            exit()
        go_first = True if usr_input[0] == "y" else False
        is_valid = True if usr_input[0] == "y" or usr_input[0] == "n" else False

    # Determine available training epochs
    models_path = os.path.join(os.getcwd(), "models")
    if not os.path.exists(models_path):
        print("No available models.")
        exit()
    model_files = os.listdir(models_path)
    available_epochs = list(set([int(f.split("_")[0]) for f in model_files]))
    available_epochs.sort()
    max_epoch = np.max(available_epochs)

    # Get which degree of trained model to load
    is_valid = False
    while not is_valid:
        print("Which training epoch? (0-" + str(max_epoch) + ")")
        prefix = input()
        suffix = "1"
        model_file_name = prefix + "_" + suffix
        model_file_path = os.path.join(models_path, model_file_name, "end_model")
        is_valid = True if model_file_name in model_files else False

    # Now instantiate policy agent
    model = DQN.load(model_file_path)

    # Instantiate environment
    env = ConnectFour(n_rows=6, n_cols=7, move_first=not go_first, self_model=model)
    done = False
    while not done:
        print(env.state)
        col = env.get_users_move()
        _, _, done, _ = env.step(col)

    print("Game over: ")
    print(env.state)


if __name__ == "__main__":
    main()
