from tabnanny import check
from gym import Env
from gym.spaces import Discrete, MultiDiscrete
import numpy as np
import random

from utils import check_connect_four


class ConnectFour(Env):
    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = 6
        self.n_cols = 7
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

    def step(self, col):
        reward = 0
        if not col in self.available_columns:
            # If row is full, give penality and randomly choose available column
            reward -= 1
            col = np.random.choice(list(self.available_columns), 1)[0]
        # Place player piece
        self.drop_piece(col=col, player=1)

        # Determine whether player won game
        playerWon = check_connect_four(state=self.state, player=1)

        if playerWon:
            # Return win reward
            reward += 1
            done = True
            return self.state.flatten(), reward, done, {"done_state": 1}

        if self.all_positions_full():
            # Return tie reward
            reward += 0
            done = True
            return self.state.flatten(), reward, done, {"done_state": 0}

        # If player has not win and board is not full, opponent chooses random column
        if not self.all_positions_full():
            # Opponent picks random column and places piece
            col = np.random.choice(list(self.available_columns), 1)[0]
            self.drop_piece(col=col, player=2)

        # Check whether opponent won
        opponentWon = check_connect_four(state=self.state, player=2)

        if opponentWon:
            # return lose reward
            reward += -1
            done = True
            return self.state.flatten(), reward, done, {"done_state": -1}

        if self.all_positions_full():
            # Return tie reward
            reward += 0
            done = True
            return self.state.flatten(), reward, done, {"done_state": 0}

        # If no conditions met, game continues
        reward += -0.1
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
