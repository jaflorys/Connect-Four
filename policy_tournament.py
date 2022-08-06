import gc
import os
import numpy as np


from connect_four_env import ConnectFour
from stable_baselines3.dqn.dqn import DQN
from utils import play_matches


def main(*, settings: dict):

    total_random_matches = settings["total_random_matches"]
    games_per_match = settings["games_per_match"]
    deterministic_player = settings["deterministic_player"]
    deterministic_opponent = settings["deterministic_opponent"]

    # Determine available training epochs
    models_path = os.path.join(os.getcwd(), "models")
    if not os.path.exists(models_path):
        print("No available models.")
        exit()

    policy_idxs = []
    policy_map = {}
    policy_file_paths = []

    for i, path in enumerate(os.listdir(models_path)):
        policy_idxs.append(i)
        policy_map[i] = path
        policy_file_paths.append(os.path.join(models_path, path, "end_model"))

    num_policies = len(policy_idxs)
    total_wins = np.zeros((num_policies, num_policies))
    total_matches = np.zeros((num_policies, num_policies))

    for i in np.arange(total_random_matches):
        print(
            "Match " + str(i + 1) + " of " + str(total_random_matches),
            end="\r",
            flush=True,
        )
        # Randomly select two policies
        idx_1, idx_2 = np.random.choice(policy_idxs, 2, replace=False)
        policy_path_1, policy_path_2 = (
            policy_file_paths[idx_1],
            policy_file_paths[idx_2],
        )
        player = DQN.load(policy_path_1)
        env = ConnectFour(
            n_rows=6,
            n_cols=7,
            move_first=None,
            opponent_models=[policy_path_2],
            deterministic_opponent=deterministic_opponent,
            max_model_history=None,
            probability_switch_model=0,
            id=1,
        )

        num_wins = play_matches(
            player_policy=player,
            deterministic_player=deterministic_player,
            env=env,
            num_games=games_per_match,
        )
        total_wins[idx_1, idx_2] += num_wins[0]
        total_wins[idx_2, idx_1] += num_wins[1]
        total_matches[idx_1, idx_2] += games_per_match
        total_matches[idx_2, idx_1] += games_per_match
        env.close()
        del player
        del env
        gc.collect()

    win_probabilities = total_wins / (total_matches - np.eye(num_policies))
    print(win_probabilities)


if __name__ == "__main__":
    settings = {
        "total_random_matches": 2000,
        "games_per_match": 1,
        "deterministic_player": True,
        "deterministic_opponent": True,
    }
    main(settings=settings)

