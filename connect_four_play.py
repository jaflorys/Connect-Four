from tabnanny import check
from gym import Env
from gym.spaces import Discrete, MultiDiscrete
import numpy as np
import random
from typing import Union

from utils import check_connect_four
from stable_baselines3.dqn.dqn import DQN


class ConnectFour(Env):
    def __init__(
        self, n_rows: int, n_cols: int, move_first: bool, self_model: Union[None, DQN]
    ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.self_model = self_model
        self.action_space = Discrete(self.n_cols)
        self.observation_space = MultiDiscrete(
            [3 for i in np.arange(self.n_rows * self.n_cols)]
        )
        # States:
        # 0: Empty place
        # -1: Black (player)
        # 1: Red (opponent)
        self.state = np.zeros((self.n_rows, self.n_cols))
        self.available_columns = set(np.arange(self.n_cols))
        self.piece = 1  # if move_first else 2
        self.usr_piece = 2  # if move_first else 1

        # If self is going second, user makes first move
        if not move_first:
            col = self.get_users_move()
            # Opponent makes first move
            self.drop_piece(col=col, player=self.usr_piece)

        # Interrogate model to find best move
        col, _ = self.self_model.predict(self.state.flatten(), deterministic=True)
        # If column not valid, choose random column
        if not col in self.available_columns:
            col = np.random.choice(list(self.available_columns), 1)[0]

        self.drop_piece(col=col, player=self.piece)

    def step(self, col):
        reward = 0
        self.drop_piece(col=col, player=self.usr_piece)

        # Determine whether user won game
        playerWon = check_connect_four(state=self.state, player=self.usr_piece)

        if playerWon:
            # Return win reward
            done = True
            return self.state.flatten(), reward, done, {"done_state": 1}

        if self.all_positions_full():
            # Return tie reward
            done = True
            return self.state.flatten(), reward, done, {"done_state": 0}

        # If user has not won, self moves
        # Interrogate model to find best move
        col, _ = self.self_model.predict(self.state.flatten(), deterministic=True)
        # If column not valid, choose random column
        if not col in self.available_columns:
            col = np.random.choice(list(self.available_columns), 1)[0]

        self.drop_piece(col=col, player=self.piece)

        # Check whether opponent won
        selfWon = check_connect_four(state=self.state, player=self.piece)

        if selfWon:
            # return lose reward
            done = True
            return self.state.flatten(), reward, done, {"done_state": -1}

        if self.all_positions_full():
            # Return tie reward
            done = True
            return self.state.flatten(), reward, done, {"done_state": 0}

        # If no conditions met, game continues
        done = False
        return self.state.flatten(), reward, done, {}

    def reset(self):
        self.state = np.zeros((self.n_rows, self.n_cols))
        self.available_columns = set(np.arange(self.n_cols))
        return self.state.flatten()

    def all_positions_full(self):
        return len(self.available_columns) == 0

    def drop_piece(self, *, col: int, player: int):
        # Determine which positions in column are filled
        filled_positions = np.abs(self.state[:, col])
        row = -1
        if np.sum(filled_positions) == 0:
            # If column is empty, lands at last row
            row = self.n_rows - 1
            self.state[self.n_rows - 1, col] = player
        else:
            # Otherwise lands at first unfilled position
            row = np.argmax(
                [
                    filled_positions[i] - filled_positions[i - 1]
                    for i in np.arange(1, self.n_rows)
                ]
            )
            self.state[row, col] = player

        if row == 0:
            # If column full, remove from set of available columns
            self.available_columns.remove(col)

    def get_users_move(self):
        valid = False
        while not valid:
            print("Enter a column to drop your piece.")
            usr_input = input()
            try:
                col = int(usr_input)
                valid_col = col in self.available_columns
                if valid_col:
                    return col
            except:
                print("Not a valid input")
